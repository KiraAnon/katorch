"""算子选择器：按实测性能表挑「同一个运算最便宜的写法」。

背景
  MDLA 上同一个数学运算常有多种表达，成本差得很远：
      BMM[N,K] adj_y=1  28.10 ms
      FC  [N,K]         28.83 ms
      BMM[K,N]          49.24 ms      —— 同一个乘法，慢 1.75 倍
  这些数字是 tools/bench_ops.py 在本机实测出来的，写进 tools/op_table.json。

  但把最优选择硬编码进 lowering 有个问题：换芯片、换尺寸、驱动升级，最优解可能变。
  所以选择逻辑放在这里 —— 查表决定，表变了行为就变，不用改 lowering。

表结构（tools/op_table.json）
    [
      {"group": "matmul", "name": "BMM_NK_adj01", "status": "ok", "ms": 28.10},
      {"group": "matmul", "name": "FC",           "status": "ok", "ms": 28.83},
      ...
    ]

用法
    from op_select import prefer
    impl = prefer("matmul", ["FC", "BMM_NK_adj01", "BMM_KN_adj00"])
    # -> "BMM_NK_adj01"（表里最快的；表缺失或不可用时按 candidates 顺序取第一个）

设计原则
  · 表读不到、某项缺失、或该项 status 不是 ok —— 一律退回保守默认，不抛异常。
    编译器不该因为缺一张性能表就跑不起来。
  · 结果带缓存，一个进程内只读一次文件。
  · 环境变量 KA_OPSEL=0 可以强制关闭，回到硬编码的行为（排查用）。
"""
import json
import os

_ENV = "KA_OPSEL"
_TABLE_NAME = "op_table.json"

_cache = None
_loaded = False


def _candidates_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, _TABLE_NAME),                        # 编译器/op_table.json
        os.path.join(here, "..", "tools", _TABLE_NAME),         # tools/op_table.json
        os.path.join(here, "..", "tools", "op_table.json"),
    ]


def load_table():
    """读性能表。返回 {(group, name): ms}，失败返回空 dict。"""
    global _cache, _loaded
    if _loaded:
        return _cache
    _loaded = True
    _cache = {}
    if os.environ.get(_ENV) == "0":
        return _cache
    for p in _candidates_paths():
        if not os.path.exists(p):
            continue
        try:
            rows = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for r in rows:
            if r.get("status") != "ok" or not r.get("ms"):
                continue
            _cache[(r.get("group"), r.get("name"))] = float(r["ms"])
        break
    return _cache


def cost(group, name):
    """某个写法的实测毫秒数；没有记录返回 None。"""
    return load_table().get((group, name))


# 只有比默认写法快这么多才换。
# 门槛存在的理由：单算子基准测不出包装成本。实测过 BMM_NK_adj01 单算子比 FC
# 快 2.5%，但它在真实图里要多包 3 个 reshape 把 2D 变 3D，整块反而慢 3.5%。
# 所以小幅优势一律不换 —— 收益抵不过上下文开销和引入的复杂度。
MIN_GAIN = 0.15


def prefer(group, candidates, default=None, min_gain=None):
    """在 candidates 里挑表中最快的那个写法。

    candidates: 按「保守优先级」排列，第一个是最稳的默认写法。
    min_gain:   至少要快这个比例才换（默认 MIN_GAIN=15%）。

    表里一个都没命中时返回 default 或 candidates[0]。
    """
    base = default if default is not None else candidates[0]
    tbl = load_table()
    if not tbl:
        return base
    base_ms = tbl.get((group, base))
    best, best_ms = None, None
    for name in candidates:
        ms = tbl.get((group, name))
        if ms is None:
            continue
        if best_ms is None or ms < best_ms:
            best, best_ms = name, ms
    if best is None:
        return base
    if base_ms is not None and best != base:
        gain = (base_ms - best_ms) / base_ms
        if gain < (MIN_GAIN if min_gain is None else min_gain):
            return base            # 优势太小，保持默认
    return best


def report():
    """给自检用：打印当前表里有什么。"""
    tbl = load_table()
    if not tbl:
        return "（没有可用的性能表，全部走保守默认）"
    lines = ["已加载算子性能表："]
    by_group = {}
    for (g, n), ms in tbl.items():
        by_group.setdefault(g, []).append((ms, n))
    for g in sorted(by_group):
        items = sorted(by_group[g])
        lines.append("  %-8s %s" % (g, "  ".join("%s=%.2fms" % (n, ms) for ms, n in items)))
    return "\n".join(lines)
