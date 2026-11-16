"""NPT lowering —— torch.fx 图 → Neuron 图描述

用法：
    low = Lowering("/tmp/out")
    x = low.input((T, HID))
    y = low.lower_module(qwen3_mlp, [x])      # 自动抓内部图并转换
    low.output(y)
    print(low.graph_text())                    # 图描述
"""

import os
import numpy as np
import torch
import torch.fx as fx
import torch.nn.functional as F

from ka_island import IslandIn, IslandOut
# 编译配方：一套「怎么把模型编成一张图」的决策集合（见 recipe.py）。
# 不传就用默认 fp16 配方，行为与重构前完全一致。
from recipe import resolve as resolve_recipe


class _IslandTracer(fx.Tracer):
    """自定义 Tracer：把 FP16 岛边界当成叶子模块，否则 FX 会把它内联成一个恒等、不留节点。"""

    def is_leaf_module(self, m, qn):
        if isinstance(m, (IslandIn, IslandOut)):
            return True
        return super().is_leaf_module(m, qn)
from shape_infer import Graph as SIGraph
from shape_infer import FC, ADD, MUL, DIV, SUB, TRANSPOSE, RESHAPE, CONCAT, CAST
from shape_infer import EXP, REDUCE_MAX, REDUCE_SUM, MAXIMUM, MINIMUM, LOGISTIC, TANH, MEAN, NEG, CONV, SOFTMAX
from shape_infer import BATCH_MATMUL, SQRT, RSQRT, ABS, RELU, LOG, SIN

T_F16, T_I32 = 8, 4


class Unsupported(Exception):
    pass


class Lowering(object):
    def __init__(self, outdir, recipe=None):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        # 配方：决定激活精度、权重精度、归约/softmax 实现、是否插 FP16 岛。
        # 优先级：显式参数 > KA_RECIPE > fp16；老环境变量（KA_REDUCE_AS_FC 等）最高。
        self.recipe = recipe or resolve_recipe()
        self.si = SIGraph()
        self.opds = []
        self.ops = []
        self.ins = []
        self.outs = []
        self.act = None
        self.tag = 0
        self._imm = {}
        self.const_data = {}
        self.node_ids = {}      # FX node.name -> operand id（标定用）
        self.node_extra = {}    # FX node.name -> 该节点创建的所有 operand

    # ---------- 基础构件 ----------
    def _tag(self, p):
        self.tag += 1
        return "%s%d" % (p, self.tag)

    def _opd(self, oid, ty, shape, tail):
        self.opds.append("OPD %d %d %d %s 0 0 %s"
                         % (oid, ty, len(shape) if shape else 0,
                            " ".join(str(v) for v in shape), tail))

    def input(self, shape):
        oid = self.si.operand(tuple(shape), "in")
        self._opd(oid, self.recipe.act_dtype, shape, "i")
        self.ins.append(oid)
        return oid

    def output(self, oid):
        self.outs.append(oid)

    def scalar(self, v):
        k = int(v)
        if k in self._imm:
            return self._imm[k]
        oid = self.si.operand((1,), "a")
        self._opd(oid, T_I32, (1,), "a %d" % k)
        self._imm[k] = oid
        return oid

    def scalar0(self, v):
        """nd=0 的 INT32 标量。CONV_2D 的 pad/stride/act 参数必须用这种。"""
        k = int(v)
        key = ("s0", k)
        if key in self._imm:
            return self._imm[key]
        oid = self.si.operand((), "a")
        self._opd(oid, 1, (), "a %d" % k)   # type=1 是 INT32 标量，CONV 参数必须用它
        self._imm[key] = oid
        return oid

    def const(self, arr, nd=None):
        arr = np.asarray(arr, dtype=np.float16)
        if nd is not None:
            arr = arr.reshape(nd)
        path = os.path.join(self.outdir, self._tag("c") + ".bin")
        arr.tofile(path)
        oid = self.si.operand(tuple(arr.shape), "w")
        self._opd(oid, T_F16, arr.shape, "w " + path)
        self.const_data[oid] = arr
        return oid

    def const_i32(self, arr):
        arr = np.asarray(arr, dtype=np.int32)
        path = os.path.join(self.outdir, self._tag("i") + ".bin")
        arr.tofile(path)
        oid = self.si.operand(tuple(arr.shape), "w")
        self._opd(oid, T_I32, arr.shape, "w " + path)
        return oid

    def op(self, code, ins, **kw):
        if os.environ.get("NPT_DEBUG"):
            print("      op %d ins=%s shapes=%s kw=%s"
                  % (code, ins, [self.si.shapes[i] for i in ins], kw))
        oid = self.si.op(code, ins, **kw)
        shape = self.si.shapes[oid]
        self._opd(oid, self.recipe.act_dtype, shape, "o")
        self.ops.append("OP %d %d %s 1 %d"
                        % (code, len(ins), " ".join(str(i) for i in ins), oid))
        return oid

    def act0(self):
        if self.act is None:
            self.act = self.scalar(0)
        return self.act

    # ---------- 算子展开 ----------
    def linear(self, x, mod):
        w = mod.weight.detach().float().numpy()
        b = (mod.bias.detach().float().numpy() if mod.bias is not None
             else np.zeros(w.shape[0], np.float32))
        xsh = self.si.shapes[x]
        if xsh[-1] != w.shape[-1]:
            raise Unsupported("linear 维度不符: x%s w%s" % (xsh, w.shape))
        if len(xsh) != 2:
            x_flat = self.op(RESHAPE, [x, self.const_i32([xsh[0], int(np.prod(xsh[1:]))])],
                             shape=[xsh[0], int(np.prod(xsh[1:]))])
        else:
            x_flat = x
        out = self.op(FC, [x_flat, self.const(w), self.const(b), self.act0()])
        if len(xsh) != 2:
            out = self.op(RESHAPE, [out, self.const_i32(list(xsh[:-1]) + [w.shape[0]])],
                          shape=list(xsh[:-1]) + [w.shape[0]])
        return out

    def linear_op(self, x, wid, bid):
        """F.linear 的 lowering：权重/bias 必须是常量 operand。"""
        wd = self.const_data.get(wid)
        if wd is None:
            raise Unsupported("F.linear 的权重必须是常量")
        xsh = self.si.shapes[x]
        if len(xsh) != 2:
            flat = [xsh[0], int(np.prod(xsh[1:]))]
            x2 = self.reshape(x, flat)
        else:
            x2 = x
        if bid is None:
            bid = self.const(np.zeros(wd.shape[0], np.float16))
        out = self.op(FC, [x2, wid, bid, self.act0()])
        if len(xsh) != 2:
            out = self.reshape(out, list(xsh[:-1]) + [int(wd.shape[0])])
        return out

    def silu(self, x):
        s = self.op(LOGISTIC, [x])
        return self.op(MUL, [x, s, self.act0()])

    def conv2d(self, x, w, b, pad=(1, 1, 1, 1), stride=(1, 1), act=0):
        """x: NHWC [N,H,W,C]，w: OHWC [O,KH,KW,C]，b: [O]。
        pad/stride/act 必须是 nd=0 的 INT32 标量，否则 adapter 转换失败。
        """
        ins = [x, w, b,
               self.scalar0(pad[0]), self.scalar0(pad[1]),
               self.scalar0(pad[2]), self.scalar0(pad[3]),
               self.scalar0(stride[0]), self.scalar0(stride[1]),
               self.scalar0(act)]
        return self.op(CONV, ins, pad=tuple(pad), stride=tuple(stride))

    def mul(self, a, b):
        return self.op(MUL, [a, b, self.act0()])

    def add(self, a, b):
        return self.op(ADD, [a, b, self.act0()])

    def rmsnorm(self, x, mod, eps=1e-6):
        sh = self.si.shapes[x]
        sq = self.op(MUL, [x, x, self.act0()])
        axis = self.scalar(len(sh) - 1)
        ms = self.op(SQRT, [self.op(ADD, [self.op(REDUCE_SUM, [sq, axis, self.act0()], axis=len(sh)-1),
                                       self.act0()], shape=sh[:-1] + [1])])
        raise Unsupported("rmsnorm 先用简化式，下一轮补")

    # ---------- FX 走图 ----------
    def lower_module(self, mod, inputs, auto_output=True):
        gm = fx.GraphModule(mod, _IslandTracer().trace(mod))
        ret = self._walk(gm, inputs)
        if auto_output:
            if isinstance(ret, list):
                for o in ret:
                    self.output(o)
            else:
                self.output(ret)
        return ret

    def _walk(self, gm, inputs):
        env = {}
        ph = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(ph) != len(inputs):
            raise Unsupported("输入个数不符: %d vs %d" % (len(ph), len(inputs)))
        for n, oid in zip(ph, inputs):
            env[n] = oid
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                a = node.args[0]
                if isinstance(a, (tuple, list)):
                    return [env[x] if isinstance(x, fx.Node) else x for x in a]
                return env[a]
            _n0 = len(self.opds)
            env[node] = self._node(node, env, gm)
            self.node_ids[node.name] = env[node]
            # 记录该节点内部创建的所有 operand（matmul 等会造出多个），
            # 标定时它们都该拿到这个节点的值，否则会落到兜底尺度上。
            self.node_extra[node.name] = [int(self.opds[i].split()[1])
                                          for i in range(_n0, len(self.opds))]
            if os.environ.get("NPT_DEBUG"):
                oid = env[node]
                sh = self.si.shapes.get(oid) if oid is not None else None
                print("   %-22s %-14s -> id=%s %s" % (node.name, node.op, oid, sh))
        raise Unsupported("没有 output 节点")

    def _args(self, node, env):
        out = []
        for a in node.args:
            if isinstance(a, fx.Node):
                out.append(env[a])
            elif isinstance(a, (tuple, list)):
                out.extend(env[x] for x in a if isinstance(x, fx.Node))
            else:
                out.append(a)
        return out

    def _shape(self, oid):
        return self.si.shapes[oid]

    # ---- 具体算子 ----
    def rmsnorm_seq(self, x, weight):
        """x: [..., D]  weight: [D]  ->  x/sqrt(mean(x^2)+eps)*w"""
        sh = self._shape(x)
        axis = len(sh) - 1
        sq = self.op(MUL, [x, x, self.act0()])
        mn = self._reduce(sq, MEAN, axis)
        eps = self.const(np.array([1e-6], np.float16), nd=(1,))
        ad = self.op(ADD, [mn, eps, self.act0()])
        rq = self.op(RSQRT, [ad])
        out = self.op(MUL, [x, rq, self.act0()])
        if weight is not None:
            out = self.op(MUL, [out, weight, self.act0()])
        return out

    def transpose(self, x, d0, d1):
        sh = list(self._shape(x))
        nd = len(sh)
        if d0 < 0: d0 += nd
        if d1 < 0: d1 += nd
        if d0 > d1: d0, d1 = d1, d0
        perm = list(range(nd)); perm[d0], perm[d1] = perm[d1], perm[d0]
        # 实测 MDLA 的 TRANSPOSE 直接支持 2D/3D/4D（3D 数值与 numpy 逐元素一致），
        # 过去给 3D 包一层 4D 桥接是两个多余的 RESHAPE。
        return self.op(TRANSPOSE, [x, self.const_i32(perm)], perm=perm)

    def transpose_n(self, x, perm):
        """任意 perm 的转置。实测 MDLA 的 TRANSPOSE 接受 2D/3D/4D 原生维度，
        不需要 4D 桥接（旧实现给 3D 包了两层 RESHAPE，纯属浪费算子）。"""
        sh = list(self._shape(x))
        if len(perm) != len(sh):
            raise Unsupported("transpose_n: perm 长度 %d != 维度 %d" % (len(perm), len(sh)))
        return self.op(TRANSPOSE, [x, self.const_i32(perm)], perm=perm)

    def reshape(self, x, shp):
        shp = list(shp)
        if -1 in shp:
            cur = self._shape(x)
            total = int(np.prod(cur))
            known = 1
            for v in shp:
                if v != -1:
                    known *= v
            shp = [total // known if v == -1 else v for v in shp]
        return self.op(RESHAPE, [x, self.const_i32(shp)], shape=shp)

    def _fc_reduce(self, x, mean=False):
        """用 FC（全 1 权重）实现最后一维的 sum/mean。

        只为 INT8 量化链路准备：MDLA 的 INT16 归约算子（MEAN/REDUCE_SUM/REDUCE_MAX）
        实测数值是坏的，而 FC 在 W8A16 下精度 0.4%。fp16 链路默认不走这条。
        """
        sh = list(self._shape(x))
        n = int(sh[-1])
        m = int(np.prod(sh[:-1])) if len(sh) > 1 else 1
        x2 = x
        if len(sh) != 2:
            x2 = self.op(RESHAPE, [x, self.const_i32([m, n])], shape=(m, n))
        w = np.full((1, n), (1.0 / n) if mean else 1.0, np.float16)
        b = np.zeros(1, np.float16)
        o = self.op(FC, [x2, self.const(w), self.const(b), self.act0()])
        tgt = tuple(sh[:-1]) + (1,)
        if len(tgt) != 2:
            o = self.op(RESHAPE, [o, self.const_i32(list(tgt))], shape=tgt)
        return o

    def _reduce(self, x, code, axis):
        """按配方选：原生归约算子，还是 FC（全 1 权重）版。

        fp16 配方走原生（完全正确）；int16 配方走 FC —— MDLA 的 INT16 归约
        （MEAN/REDUCE_SUM/REDUCE_MAX）实测是坏的，而 FC 在 W8A16 下精度 0.4%。
        """
        if self.recipe.reduce_impl(axis) == "fc":
            return self._fc_reduce(x, mean=(code == MEAN))
        ax = self.scalar(axis)
        return self.op(code, [x, ax, self.act0()], axis=axis)

    def softmax5(self, x, axis=-1):
        sh = self._shape(x)
        if axis < 0:
            axis += len(sh)
        # 归一化/softmax 这类对精度敏感、又依赖归约算子的部分，整体放进 FP16 岛：
        # MDLA 的 INT16 归约（MEAN/REDUCE_MAX/REDUCE_SUM）实测是坏的，
        # 而 FP16 下它们完全正确。边界用 DEQUANTIZE(6) / QUANTIZE(72)。
        island = self.recipe.use_island(axis=axis, ndim=len(sh))
        if island:
            x = self.op(6, [x])          # INT16 -> FP16
        ax = self.scalar(axis)
        mx = self.op(REDUCE_MAX, [x, ax, self.act0()], axis=axis)
        sb = self.op(SUB, [x, mx, self.act0()])
        ex = self.op(EXP, [sb])
        sm = self.op(REDUCE_SUM, [ex, ax, self.act0()], axis=axis)
        out = self.op(DIV, [ex, sm, self.act0()])
        if island:
            out = self.op(72, [out])     # FP16 -> INT16
        return out

    def scalar_f32(self, v):
        """FLOAT32 标量（type 0, nd=0）。SOFTMAX 的 beta 必须用这个类型——
        换成 type 3 (TENSOR_FLOAT32) 实测编译直接 -516。

        ★ 必须走 w 文件，不能用立即数：`a 1` 写进去的是整数 1 的位模式
        (0x00000001)，而 type 0 要的是 float 1.0 (0x3F800000)。用立即数时
        beta 变成 1.4e-45 ≈ 0，softmax 退化成均匀分布（实测输出恒为 1/64）。
        """
        k = ("f32", float(v))
        if k in self._imm:
            return self._imm[k]
        arr = np.array([float(v)], np.float32)
        path = os.path.join(self.outdir, self._tag("f") + ".bin")
        arr.tofile(path)
        oid = self.si.operand((), "w")
        self._opd(oid, 0, (), "w " + path)
        self._imm[k] = oid
        return oid

    def softmax_native(self, x, axis=-1, beta=1.0):
        """原生 SOFTMAX（opcode 25）：数据 + float32 beta 标量，一个算子。

        实测 2D/3D/4D 都支持，最后一维不要求是 2 的幂（100x100 也编得过）。
        相比手写的 softmax5 少 4 个算子。仅用于 fp16 图；
        INT16 岛模式（KA_NORM_ISLAND）下量化参数未验证，仍走 softmax5。
        """
        if float(beta) != 1.0:
            raise Unsupported("SOFTMAX beta != 1.0 未验证")
        return self.op(SOFTMAX, [x, self.scalar_f32(beta)], axis=axis)

    def softmax(self, x, axis=-1):
        """默认走手写展开 softmax5 —— 这是 fp16 图唯一正确的做法。

        ★ 实测结论（别被「有原生算子」骗了）
          原生 SOFTMAX(25) 只对 **INT16 量化输入** 正确：
            int16 输入 + 输出 scale 1/32768  -> 与 torch 逐元素吻合，行和 1.0
          对 **fp16 输入**，它恒输出均匀分布 1/N，完全不看数据：
            输入 [1..8] / [0..0,1] / 全负，输出都一样 [0.125 ×8]
          所以 fp16 图必须展开成 REDUCE_MAX/SUB/EXP/REDUCE_SUM/DIV 五个算子。
          当年 optable.py 测出的「SOFTMAX 0.05%」是展开版的成绩，不是原生 25。

        KA_SOFTMAX=native 可以在确认输入是 int16 时强制走原生（省 4 个算子）。
        """
        if self.recipe.softmax_impl == "native":
            return self.softmax_native(x, axis)
        return self.softmax5(x, axis)
    def matmul(self, a, b):
        ash, bsh = list(self._shape(a)), list(self._shape(b))
        # 2D @ 2D
        if len(ash) == 2 and len(bsh) == 2:
            if os.environ.get("NPT_DEBUG"):
                print("    matmul 2D@2D: a=%s b=%s" % (ash, bsh))
            a3 = self.reshape(a, [1, ash[0], ash[1]])
            b3 = self.reshape(b, [1, bsh[0], bsh[1]])
            o = self.op(BATCH_MATMUL, [a3, b3, self.scalar(0), self.scalar(0)], adj_x=0, adj_y=0)
            osh = self._shape(o)
            return self.reshape(o, [osh[1], osh[2]])
        # 3D @ 2D: [B,M,K] @ [K,N]  -> 用 FC（BMM 的 batch 维必须相等，不能广播）
        if len(ash) == 3 and len(bsh) == 2:
            if os.environ.get("NPT_DEBUG"):
                bd0 = self.const_data.get(b)
                print("    matmul 3D@2D: a=%s b=%s bd_shape=%s" % (ash, bsh, None if bd0 is None else bd0.shape))
            bd = self.const_data.get(b)
            if bd is None:
                raise Unsupported("3D@2D matmul 右操作数不是常量: a=%s b=%s kinds=%s" %
                                  (ash, bsh, [getattr(self.si, "kinds", {}).get(i) for i in (a, b)]))
            bm = ash[0] * ash[1]
            a2 = self.reshape(a, [bm, ash[2]])
            W = np.ascontiguousarray(np.asarray(bd, np.float16).T)   # y = x @ B  =>  W = B^T
            zb = np.zeros(W.shape[0], np.float16)
            out2 = self.op(FC, [a2, self.const(W), self.const(zb), self.act0()])
            return self.reshape(out2, [ash[0], ash[1], bsh[1]])
        # 3D @ 3D
        if len(ash) == 3 and len(bsh) == 3:
            return self.op(BATCH_MATMUL, [a, b, self.scalar(0), self.scalar(0)], adj_x=0, adj_y=0)
        raise Unsupported("matmul %s x %s" % (ash, bsh))

    def _node(self, node, env, gm):
        if node.op == "call_module":
            mod = gm.get_submodule(node.target)
            x = env[node.args[0]]
            # FP16 norm 岛的边界标记（见 ka_island.py）
            if isinstance(mod, IslandIn):
                if not self.recipe.norm_island:
                    return x
                return self.op(6, [x])     # DEQUANTIZE  INT16 -> FP16
            if isinstance(mod, IslandOut):
                if not self.recipe.norm_island:
                    return x
                return self.op(72, [x])    # QUANTIZE    FP16 -> INT16
            if isinstance(mod, torch.nn.Linear):
                if os.environ.get("NPT_DEBUG"):
                    print("   linear %-22s in=%s" % (node.target, self.si.shapes[x]))
                return self.linear(x, mod)
            raise Unsupported("call_module %s" % type(mod).__name__)

        if node.op == "get_attr":
            obj = None
            try:
                obj = gm.get_parameter(node.target)
            except AttributeError:
                try: obj = gm.get_buffer(node.target)
                except AttributeError: obj = None
            if obj is None:
                raise Unsupported("get_attr %s" % node.target)
            return self.const(obj.detach().float().numpy())

        if node.op == "call_function":
            fn = node.target
            nm = getattr(fn, "__name__", "")
            a = self._args(node, env)
            if nm == "getattr":
                attr = node.args[1] if len(node.args) > 1 else None
                if attr == "T":
                    # 常量转置直接折叠，不生成算子（这样 matmul 的右操作数仍是常量）
                    cd = self.const_data.get(a[0])
                    if cd is not None:
                        return self.const(np.ascontiguousarray(np.asarray(cd).T))
                    sh = self._shape(a[0])
                    perm = [1, 0] if len(sh) == 2 else list(range(len(sh)))[::-1]
                    if len(perm) in (2, 4):
                        return self.op(TRANSPOSE, [a[0], self.const_i32(perm)], perm=perm)
                    return self.transpose_n(a[0], perm)
                return a[0]      # 其它属性（如 .dtype）直接透传
            if nm == "linear":
                b = a[2] if len(a) > 2 else None
                if b is not None and not isinstance(b, int):
                    b = None
                return self.linear_op(a[0], a[1], b)
            if nm in ("silu",):
                return self.silu(a[0])
            if nm == "conv2d":
                # F.conv2d(x_nchw, w_oihw, b, stride, padding)
                # 设备端是 NHWC/O,H,W,C，这里自动转换
                x = a[0]
                w = a[1]
                bias = a[2] if len(a) > 2 and a[2] is not None else None
                stride = node.kwargs.get("stride", 1)
                padding = node.kwargs.get("padding", 0)
                if isinstance(stride, int): stride = (stride, stride)
                if isinstance(padding, int): padding = (padding, padding)
                pad4 = (padding[1], padding[1], padding[0], padding[0])   # l,r,t,b
                # 权重必须是常量才能预转置
                wd = self.const_data.get(w)
                if wd is None:
                    raise Unsupported("conv2d: 权重不是常量")
                w_ohwc = np.ascontiguousarray(np.transpose(wd, (0, 2, 3, 1)))  # O,C,KH,KW -> O,KH,KW,C
                if os.environ.get("NPT_DEBUG"):
                    print("    conv2d: wd=%s -> w_ohwc=%s  x=%s" % (wd.shape, w_ohwc.shape, self._shape(x)))
                x_nhwc = self.transpose_n(x, [0, 2, 3, 1])
                y = self.conv2d(x_nhwc, self.const(w_ohwc), bias, pad4, stride)
                return self.transpose_n(y, [0, 3, 1, 2])

            if nm == "conv2d_nhwc":
                # (x, w, b, pad, stride, act) —— pad/stride 是 python 元组，
                # 不在 a 里（_args 只收集张量），直接从 node.args 取
                pad = node.args[3] if len(node.args) > 3 else (1, 1, 1, 1)
                stride = node.args[4] if len(node.args) > 4 else (1, 1)
                act = node.args[5] if len(node.args) > 5 else 0
                return self.conv2d(a[0], a[1], a[2], pad, stride, act)
            if nm == "tanh":
                return self.op(TANH, [a[0]])
            if nm == "exp":
                return self.op(EXP, [a[0]])
            if nm == "island_in":
                if not self.recipe.norm_island:
                    return a[0]
                return self.op(6, [a[0]])      # DEQUANTIZE  INT16 -> FP16
            if nm == "island_out":
                if not self.recipe.norm_island:
                    return a[0]
                return self.op(72, [a[0]])     # QUANTIZE    FP16 -> INT16
            if nm == "sqrt":
                return self.op(SQRT, [a[0]])
            if nm == "rsqrt":
                return self.op(RSQRT, [a[0]])
            if nm == "sigmoid":
                return self.op(LOGISTIC, [a[0]])
            if nm == "maximum":
                return self.op(MAXIMUM, [self._as_operand(a[0]), self._as_operand(a[1])])
            if nm == "minimum":
                return self.op(MINIMUM, [self._as_operand(a[0]), self._as_operand(a[1])])
            if nm == "abs":
                # abs(x) = maximum(x, -x)
                x = self._as_operand(a[0])
                return self.op(MAXIMUM, [x, self.op(NEG, [x])])
            if nm == "relu":
                x = self._as_operand(a[0])
                return self.op(MAXIMUM, [x, self.const(np.array([0.0], np.float16), nd=(1,))])
            if nm == "neg":
                return self.op(NEG, [a[0]])
            if nm in ("div", "truediv"):
                return self.op(DIV, [self._as_operand(a[0]), self._as_operand(a[1])])
            if nm == "log":
                return self.op(LOG, [a[0]])
            if nm == "sin":
                return self.op(SIN, [a[0]])
            if nm in ("mul", "add", "sub"):
                x, y = self._as_operand(a[0]), self._as_operand(a[1])
                code = {"mul": MUL, "add": ADD, "sub": SUB}[nm]
                return self.op(code, [x, y, self.act0()])
            if nm == "rsqrt":
                return self.op(RSQRT, [a[0]])
            if nm == "softmax":
                dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else -1)
                return self.softmax(a[0], dim)
            if nm == "matmul":
                return self.matmul(a[0], a[1])
            if nm == "cat":
                tensors = [env[t] for t in node.args[0]]
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
                cur = tensors[0]
                axis = self.scalar(dim)
                for t in tensors[1:]:
                    cur = self.op(CONCAT, [cur, t, axis], axis=dim)
                return cur
            if nm == "getitem":
                raise Unsupported("getitem args=%r" % (node.args,))
            raise Unsupported("call_function %s" % (nm or fn))

        if node.op == "call_method":
            m = node.target
            x = env[node.args[0]] if isinstance(node.args[0], fx.Node) else None
            if m == "to":
                return x
            if m == "pow":
                e = node.args[1]
                if e == 2:
                    return self.op(MUL, [x, x, self.act0()])
                raise Unsupported("pow(%s)" % e)
            if m == "mean":
                dim = node.args[1] if len(node.args) > 1 else -1
                return self._reduce(x, MEAN, dim)
            if m == "sum":
                # 只支持 keepdim=True（shape_infer 的 REDUCE_SUM 默认保留维度）
                dim = node.args[1] if len(node.args) > 1 else -1
                if len(node.args) > 2 and node.args[2] is False:
                    raise Unsupported("sum(keepdim=False)")
                return self._reduce(x, REDUCE_SUM, dim)
            if m == "reshape":
                shp = [int(v) for v in node.args[1:]]
                return self.reshape(x, shp)
            if m in ("t", "T"):
                # .T 对 2D 是转置；对更高维是维度反序
                sh = self._shape(x)
                if len(sh) == 2:
                    perm = [1, 0]
                else:
                    perm = list(range(len(sh)))[::-1]
                nd = len(perm)
                if nd == 2 or nd == 4:
                    return self.op(TRANSPOSE, [x, self.const_i32(perm)], perm=perm)
                return self.transpose_n(x, perm)
            if m == "transpose":
                return self.transpose(x, int(node.args[1]), int(node.args[2]))
            if m == "permute":
                perm = [int(v) for v in node.args[1:]]
                return self.transpose_n(x, perm)
            raise Unsupported("call_method %s" % m)

        if node.op == "call_function" and getattr(node.target, "__name__", "") == "getitem":
            import traceback
            raise Unsupported("getitem args=%r\n%s" % (node.args, "".join(traceback.format_stack()[-6:])))
        raise Unsupported("node.op=%s target=%s" % (node.op, node.target))

    def _as_operand(self, v):
        """operand id 是 int；float 一定是数值常量；int 若不在 shapes 里也是常量。"""
        if type(v) is float:
            return self._maybe_const(v, None)
        if type(v) is int and v not in self.si.shapes:
            return self._maybe_const(float(v), None)
        return v

    def _maybe_const(self, v, src):
        if isinstance(v, (int, float)):
            return self.const(np.array([float(v)], np.float16), nd=(1,))
        return v

    # ---------- 输出 ----------
    def graph_text(self):
        lines = list(self.opds)
        lines.append("IO %d %s %d %s"
                     % (len(self.ins), " ".join(str(i) for i in self.ins),
                        len(self.outs), " ".join(str(i) for i in self.outs)))
        lines.extend(self.ops)
        lines.append("END")
        return "\n".join(lines) + "\n"

    def validate(self):
        from shape_infer import validate_txt
        def rf(p):
            try: return open(p, "rb").read()
            except Exception: return None
        return validate_txt(self.graph_text(), rf)
