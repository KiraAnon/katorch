"""编译配方（Recipe）—— 一套「怎么把模型编成一张图」的决策集合。

为什么要有这一层：

    KAtorch 不是只服务一种生图流，也不是只服务一种精度。
    fp16 有 fp16 的最优结构，int16 有 int16 的最优结构，w8a16 又是另一种。
    但它们共享同一份 lowering 正文 —— 差异全部收在这里。

一个配方决定七件事：

    1. act_dtype      激活精度（全图统一，不混）
    2. weight_dtype   权重精度
    3. reduce_as_fc   大维度归约走原生还是 FC(ones)
    4. softmax_impl   原生 SOFTMAX 还是展开成 5 个算子
    5. norm_island    是否在归一化周围插 DEQUANTIZE/QUANTIZE
    6. banned_ops     这套配方下不能用的算子
    7. note           给人看的说明

新增一套配方 = 加一个 Recipe 常量，不改 lowering。

兼容性：
    历史上用环境变量控制（KA_REDUCE_AS_FC / KA_NORM_ISLAND / KA_SOFTMAX）。
    `Recipe.resolve()` 会读它们并覆盖，所以老脚本不设任何变量时行为不变。
"""

import os
from dataclasses import dataclass, field, replace

# Neuron 类型码（与 host.c 的 OT.type 一致）
T_FP16 = 8
T_INT16 = 7
T_INT8_PC = 11     # QUANT8_SYMM_PER_CHANNEL


@dataclass(frozen=True)
class Recipe(object):
    name: str
    act_dtype: int = T_FP16
    weight_dtype: str = "fp16"          # 'fp16' | 'int8_pc' | 'int16'
    reduce_as_fc: bool = False          # 大维度归约是否改用 FC(ones)
    reduce_native_max_dim: int = 0      # 原生归约支持的最大归约维；0 = 不限
    softmax_impl: str = "unroll5"       # 'unroll5' | 'native'
    norm_island: bool = False           # 归一化周围插 DQ/Q
    banned_ops: frozenset = field(default_factory=frozenset)
    note: str = ""

    # ---- 决策查询（lowering 只调这些，不直接看字段）----

    def reduce_impl(self, axis):
        """返回 'fc' 或 'native'。axis 是调用方传进来的原始值（-1 表示最后一维）。"""
        if self.reduce_as_fc and axis == -1:
            return "fc"
        return "native"

    def use_island(self, axis=None, ndim=None):
        """是否在这处插 FP16 岛。ndim 给定时只在最后一维生效（历史行为）。"""
        if not self.norm_island:
            return False
        if axis is None:
            return True
        return ndim is not None and axis == ndim - 1

    def check_op(self, code, name=""):
        if code in self.banned_ops:
            raise ValueError("配方 %s 禁用算子 %s(%d)" % (self.name, name or "?", code))


# ---------- 内置配方 ----------

# 当前使用的配方：全 fp16，归约走原生，softmax 展开，不插岛。
# ★ 这一套必须和重构前的行为逐字节一致。
FP16 = Recipe(
    name="fp16",
    act_dtype=T_FP16,
    weight_dtype="fp16",
    reduce_as_fc=False,
    reduce_native_max_dim=0,
    softmax_impl="unroll5",
    norm_island=False,
    note="全 fp16。归约用原生算子（fp16 下完全正确），softmax 必须展开成 5 个算子。",
)

# 照玑绘的路线：全 int16 激活 + int8 per-channel 权重，零 DQ/Q。
# 图形态参考 aot/mt6897/ditc1c_w8_w16s4rs.npug：
#   MEAN 只在 128 维上归约（3072 维会溢出/失准）
#   SOFTMAX 走原生（int16 输入 + 输出 scale 1/32768）
#   DEQUANTIZE / QUANTIZE 出现 0 次
INT16 = Recipe(
    name="int16",
    act_dtype=T_INT16,
    weight_dtype="int8_pc",
    reduce_as_fc=True,
    reduce_native_max_dim=128,
    softmax_impl="native",
    norm_island=False,
    note="单一 int16 贯穿 + int8 per-channel 权重。归约按维度分流，softmax 走原生。",
)

# int16 主路 + int8 权重，但归约仍走 FC(ones)（不依赖原生小维度归约）。
W8A16 = Recipe(
    name="w8a16",
    act_dtype=T_INT16,
    weight_dtype="int8_pc",
    reduce_as_fc=True,
    reduce_native_max_dim=0,
    softmax_impl="native",
    norm_island=False,
    note="int16 主路 + int8 权重。与旧 W8A16 的区别：旧版是 fp16 主路 + 岛，会多出 36 个 DQ/Q。",
)

REGISTRY = {r.name: r for r in (FP16, INT16, W8A16)}


def resolve(name=None):
    """按 显式参数 > KA_RECIPE > 默认 fp16 选配方，再让老环境变量覆盖。

    环境变量优先级最高，保证现有脚本行为不变。
    """
    key = name or os.environ.get("KA_RECIPE") or "fp16"
    base = REGISTRY.get(key)
    if base is None:
        raise ValueError("未知配方 %r，可选：%s" % (key, ", ".join(sorted(REGISTRY))))

    over = {}
    if os.environ.get("KA_REDUCE_AS_FC"):
        over["reduce_as_fc"] = True
    if os.environ.get("KA_NORM_ISLAND"):
        over["norm_island"] = True
    if os.environ.get("KA_SOFTMAX") == "native":
        over["softmax_impl"] = "native"
    return replace(base, **over) if over else base


def describe(r):
    return ("%s: act=%s weight=%s reduce=%s softmax=%s island=%s"
            % (r.name,
               {T_FP16: "fp16", T_INT16: "int16"}.get(r.act_dtype, r.act_dtype),
               r.weight_dtype,
               "fc" if r.reduce_as_fc else "native",
               r.softmax_impl,
               "on" if r.norm_island else "off"))
