"""算子形状推导 + 校验

为什么必须有这层：
  adapter **不校验** operand 声明的形状和算子实际输出是否一致。
  声明错了它不报错，只按你声明的尺寸分配 buffer，算子只写实际那部分，
  剩下的字节是未初始化内存 → 后续算子读进去 → 数值全错但 rc 全 0。
  （2026-11-13 实测：REDUCE_MAX 声明 [H,T,T] 实际输出 [H,T,1]，attention 误差 0.33）

三层防护：
  infer()        —— 单个算子的形状推导
  Graph          —— 构图时自动推导 + 校验
  validate_txt() —— 发图前校验手写的图描述文件（最有用的一层）
"""

import struct

ADD, AVGPOOL, CONCAT, CONV = 0, 1, 2, 3
FC = 9
LOGISTIC, MUL, RELU, RELU1, RELU6 = 14, 18, 19, 20, 21
MAXPOOL, RESHAPE = 17, 22
SOFTMAX, TANH, MEAN = 25, 28, 31
SUB, TRANSPOSE = 36, 37
ABS, CAST = 38, 45
EXP, LOG, DIV = 49, 60, 30
LOG_SOFTMAX = 64
MAXIMUM, MINIMUM, NEG = 65, 66, 67
REDUCE_MAX, REDUCE_SUM = 77, 80
RSQRT, SIN, SQRT = 83, 85, 88
BATCH_MATMUL = 102

NAMES = {0:"ADD",1:"AVGPOOL",2:"CONCAT",3:"CONV_2D",9:"FC",14:"LOGISTIC",17:"MAXPOOL",
         18:"MUL",19:"RELU",22:"RESHAPE",25:"SOFTMAX",28:"TANH",30:"DIV",31:"MEAN",
         36:"SUB",37:"TRANSPOSE",38:"ABS",45:"CAST",49:"EXP",60:"LOG",64:"LOG_SOFTMAX",
         65:"MAXIMUM",66:"MINIMUM",67:"NEG",77:"REDUCE_MAX",80:"REDUCE_SUM",
         83:"RSQRT",85:"SIN",88:"SQRT",102:"BATCH_MATMUL"}

UNARY = {14, 19, 20, 21, 25, 28, 38, 45, 49, 60, 64, 67, 83, 85, 88}
BINARY = {0, 18, 30, 36, 65, 66}
REDUCE = {31, 77, 80}


class ShapeError(Exception):
    pass


def _broadcast(a, b):
    """numpy 风格广播。"""
    a = list(a); b = list(b)
    n = max(len(a), len(b))
    a = [1] * (n - len(a)) + a
    b = [1] * (n - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x == y or x == 1 or y == 1:
            out.append(max(x, y))
        else:
            raise ShapeError("broadcast 不兼容: %s vs %s" % (a, b))
    return tuple(out)


def infer(op, ins, **kw):
    """推导输出形状，返回 list[tuple]。
    kw: axis / perm / shape / adj_x / adj_y / keepdim
    """
    ins = [tuple(i) for i in ins]

    if op == FC:                                  # (x, w, b, act)
        x, w = ins[0], ins[1]
        if x[-1] != w[-1]:
            raise ShapeError("FC: x[-1]=%d != w[-1]=%d" % (x[-1], w[-1]))
        return [tuple(x[:-1]) + (w[0],)]

    if op in BINARY:
        if len(ins) < 2:
            raise ShapeError("二元算子需要至少 2 个输入")
        return [_broadcast(ins[0], ins[1])]

    if op in REDUCE:                              # (x, axis, act)
        axis = kw.get("axis")
        if axis is None:
            raise ShapeError("%s 需要 axis" % NAMES.get(op, op))
        nd = len(ins[0])
        if axis < 0: axis += nd
        if not (0 <= axis < nd):
            raise ShapeError("%s: axis=%d 越界 (nd=%d)" % (NAMES.get(op, op), axis, nd))
        out = list(ins[0])
        if kw.get("keepdim", True): out[axis] = 1
        else: del out[axis]
        return [tuple(out)]

    if op in UNARY:
        return [ins[0]]

    if op == TRANSPOSE:                           # (x, perm)
        perm = kw.get("perm")
        if perm is None:
            raise ShapeError("TRANSPOSE 需要 perm")
        if len(perm) != len(ins[0]):
            raise ShapeError("TRANSPOSE: perm 长度 %d != 张量维度 %d" % (len(perm), len(ins[0])))
        return [tuple(ins[0][p] for p in perm)]

    if op == RESHAPE:                             # (x, shape)
        shp = kw.get("shape")
        if shp is None:
            raise ShapeError("RESHAPE 需要 shape")
        t0 = 1
        for v in ins[0]: t0 *= v
        t1 = 1
        for v in shp: t1 *= v
        if t0 != t1:
            raise ShapeError("RESHAPE: %s 有 %d 个元素, %s 有 %d 个" % (ins[0], t0, tuple(shp), t1))
        return [tuple(shp)]

    if op == CONCAT:                              # (a, b, axis)
        axis = kw.get("axis")
        if axis is None:
            raise ShapeError("CONCAT 需要 axis")
        a, b = ins[0], ins[1]
        if len(a) != len(b):
            raise ShapeError("CONCAT: 维度不同 %s vs %s" % (a, b))
        out = list(a)
        out[axis] += b[axis]
        return [tuple(out)]

    if op == BATCH_MATMUL:                        # (a, b, adj_x, adj_y)
        ax = int(kw.get("adj_x", 0)); ay = int(kw.get("adj_y", 0))
        a, b = list(ins[0]), list(ins[1])
        if len(a) != 3 or len(b) != 3:
            raise ShapeError("BATCH_MATMUL 需要 3D 输入，收到 %s / %s" % (tuple(a), tuple(b)))
        B = a[0]
        M, K = (a[2], a[1]) if ax else (a[1], a[2])
        K2, N = (b[2], b[1]) if ay else (b[1], b[2])
        if K != K2:
            raise ShapeError("BATCH_MATMUL: 内积维 %d vs %d" % (K, K2))
        if b[0] not in (B, 1):
            raise ShapeError("BATCH_MATMUL: batch %d vs %d" % (B, b[0]))
        return [(B, M, N)]

    if op == CONV:                                # (x, w, b, pad..., stride..., act)
        # x: NHWC [N,H,W,C]，w: OHWC [O,KH,KW,C]
        x, w = ins[0], ins[1]
        if len(x) != 4 or len(w) != 4:
            raise ShapeError("CONV_2D 需要 4D：x=%s w=%s" % (tuple(x), tuple(w)))
        if x[3] != w[3]:
            raise ShapeError("CONV_2D: 输入通道 %d != 权重通道 %d" % (x[3], w[3]))
        pad = kw.get("pad") or (1, 1, 1, 1)       # (l, r, t, b)
        stride = kw.get("stride") or (1, 1)       # (w, h)
        H, W = x[1], x[2]
        OH = (H + pad[2] + pad[3] - w[1]) // stride[1] + 1
        OW = (W + pad[0] + pad[1] - w[2]) // stride[0] + 1
        if OH <= 0 or OW <= 0:
            raise ShapeError("CONV_2D: 输出尺寸非法 OH=%d OW=%d" % (OH, OW))
        return [(x[0], OH, OW, w[0])]

    raise ShapeError("未实现的算子 %s (%s)" % (NAMES.get(op, "?"), op))


def check(tag, declared, inferred):
    """声明形状必须等于推导形状。"""
    d, i = tuple(declared), tuple(inferred)
    if d != i:
        raise ShapeError("[%s] 声明 %s != 推导 %s  —— adapter 会静默出错" % (tag, d, i))


class Graph(object):
    """链式构图：加算子时自动推导输出形状。"""

    def __init__(self):
        self.shapes = {}
        self.kinds = {}
        self.ops = []
        self.next_id = 0

    def operand(self, shape, kind="o"):
        oid = self.next_id; self.next_id += 1
        self.shapes[oid] = tuple(shape)
        self.kinds[oid] = kind
        return oid

    def op(self, opcode, ins, **kw):
        out = infer(opcode, [self.shapes[i] for i in ins], **kw)[0]
        oid = self.operand(out)
        self.ops.append((opcode, list(ins), [oid], kw))
        return oid

    def validate(self):
        for opcode, ins, outs, kw in self.ops:
            want = infer(opcode, [self.shapes[i] for i in ins], **kw)[0]
            check("op%d" % opcode, self.shapes[outs[0]], want)
        return len(self.ops)


def validate_txt(text, read_file=None):
    """校验图描述里每个 OPD 声明的形状是否等于算子推导形状。

    text: 图描述文本（多行）
    read_file(path) -> bytes 或 None，用来读 w 常量（axis/perm/shape 参数）
    返回错误字符串列表，空 = 通过。
    """
    opds = {}
    ops = []
    errors = []

    def const_ints(oid):
        e = opds.get(oid)
        if e is None or e["kind"] not in ("a", "w"):
            return None
        if e["kind"] == "a":
            return [e["val"]]
        if read_file is None or not e["path"]:
            return None
        b = read_file(e["path"])
        if not b or len(b) % 4:
            return None
        return list(struct.unpack("<%di" % (len(b) // 4), b))

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if p[0] == "OPD" and len(p) >= 6:
            oid = int(p[1]); ty = int(p[2]); nd = int(p[3])
            dims = tuple(int(x) for x in p[4:4 + nd])
            # 格式: OPD id type nd dims... scale zp [kind ...]
            rest = p[4 + nd + 2:]
            kind, val, path = "o", None, None
            if rest:
                if rest[0] == "a" and len(rest) >= 2: kind, val = "a", int(rest[1])
                elif rest[0] == "w" and len(rest) >= 2: kind, path = "w", rest[1]
                elif rest[0] == "i": kind = "in"
            opds[oid] = dict(type=ty, nd=nd, dims=dims, kind=kind, val=val, path=path)
        elif p[0] == "OP" and len(p) >= 4:
            code = int(p[1]); ni = int(p[2])
            ins = [int(x) for x in p[3:3 + ni]]
            no = int(p[3 + ni])
            outs = [int(x) for x in p[4 + ni:4 + ni + no]]
            ops.append((code, ins, outs))

    for code, ins, outs in ops:
        kw = {}
        try:
            if code in REDUCE:
                v = const_ints(ins[1]); kw["axis"] = v[0] if v else None
            elif code == TRANSPOSE:
                kw["perm"] = const_ints(ins[1])
            elif code == RESHAPE:
                kw["shape"] = const_ints(ins[1])
            elif code == CONCAT:
                v = const_ints(ins[2]); kw["axis"] = v[0] if v else None
            elif code == BATCH_MATMUL:
                vx = const_ints(ins[2]); vy = const_ints(ins[3])
                kw["adj_x"] = vx[0] if vx else 0
                kw["adj_y"] = vy[0] if vy else 0
            elif code == CONV:
                # pad 在 ins[3..6]（l,r,t,b），stride 在 ins[7..8]（w,h）
                kw["pad"] = tuple((const_ints(ins[i]) or [1])[0] for i in range(3, 7))
                kw["stride"] = tuple((const_ints(ins[i]) or [1])[0] for i in range(7, 9))
            want = infer(code, [opds[i]["dims"] for i in ins], **kw)[0]
        except ShapeError as e:
            errors.append("%s: 推导失败: %s" % (NAMES.get(code, code), e))
            continue
        except Exception as e:
            errors.append("%s: %r" % (NAMES.get(code, code), e))
            continue
        for oid in outs:
            got = opds.get(oid, {}).get("dims")
            if got is None:
                errors.append("%s: 输出 %d 没有 OPD 声明" % (NAMES.get(code, code), oid))
            elif tuple(got) != tuple(want):
                errors.append("%s: 输出 %d 声明 %s != 推导 %s  ← adapter 会静默出错"
                              % (NAMES.get(code, code), oid, tuple(got), tuple(want)))
    return errors


# ------------------------- 自检 -------------------------
if __name__ == "__main__":
    H, T, d = 2, 4, 8
    ok = 0; bad = []

    def must_fail(fn):
        try:
            fn()
        except ShapeError:
            return
        raise ShapeError("本该失败却通过了")

    def t(tag, fn):
        global ok
        try:
            fn(); ok += 1; print("  PASS %s" % tag)
        except Exception as e:
            bad.append((tag, e)); print("  FAIL %s: %s" % (tag, e))

    t("FC",             lambda: check("fc", infer(FC, [(T,64),(128,64),(128,),(1,)])[0], (T,128)))
    t("FC 维度不符",      lambda: must_fail(lambda: infer(FC, [(T,64),(128,32),(128,),(1,)])))
    t("ADD 广播",        lambda: check("add", infer(ADD, [(H,T,T),(H,T,1)])[0], (H,T,T)))
    t("SUB 广播",        lambda: check("sub", infer(SUB, [(H,T,T),(H,T,1)])[0], (H,T,T)))
    t("MUL 同形",        lambda: check("mul", infer(MUL, [(T,d),(T,d)])[0], (T,d)))
    t("TRANSPOSE",      lambda: check("tr", infer(TRANSPOSE, [(1,T,H,d)], perm=[0,2,1,3])[0], (1,H,T,d)))
    t("RESHAPE",        lambda: check("rs", infer(RESHAPE, [(1,H,T,d)], shape=[H,T,d])[0], (H,T,d)))
    t("RESHAPE 元素错",   lambda: must_fail(lambda: infer(RESHAPE, [(1,H,T,d)], shape=[H,T,d+1])))
    t("BMM adj_y=1",    lambda: check("bmm", infer(BATCH_MATMUL, [(H,T,d),(H,T,d)], adj_x=0, adj_y=1)[0], (H,T,T)))
    t("BMM adj=0",      lambda: check("bmm", infer(BATCH_MATMUL, [(H,T,d),(H,d,T)], adj_x=0, adj_y=0)[0], (H,T,T)))
    t("BMM 4D 拒绝",     lambda: must_fail(lambda: infer(BATCH_MATMUL, [(1,H,T,d),(1,H,T,d)], adj_x=0, adj_y=1)))
    t("BMM 内积不符",     lambda: must_fail(lambda: infer(BATCH_MATMUL, [(H,T,d),(H,T,d+1)], adj_x=0, adj_y=1)))
    t("REDUCE keepdim", lambda: check("rd", infer(REDUCE_SUM, [(H,T,T),(1,),(1,)], axis=2)[0], (H,T,1)))
    t("REDUCE 轴越界",    lambda: must_fail(lambda: infer(REDUCE_SUM, [(H,T,T)], axis=5)))
    t("CONCAT",         lambda: check("cc", infer(CONCAT, [(T,d),(T,d)], axis=1)[0], (T,2*d)))
    t("EXP",            lambda: check("exp", infer(EXP, [(H,T,T)])[0], (H,T,T)))
    t("CAST",           lambda: check("cast", infer(CAST, [(T,d)])[0], (T,d)))

    print("\n%d 通过, %d 失败" % (ok, len(bad)))
    for tag, e in bad: print("  !! %s: %s" % (tag, e))

    # 链式：attention
    g = Graph()
    q = g.operand((1,T,H,d), "in"); k = g.operand((1,T,H,d), "in"); v = g.operand((1,T,H,d), "in")
    p1 = g.operand((4,), "w")
    qh = g.op(TRANSPOSE, [q, p1], perm=[0,2,1,3])
    kh = g.op(TRANSPOSE, [k, p1], perm=[0,2,1,3])
    vh = g.op(TRANSPOSE, [v, p1], perm=[0,2,1,3])
    sh3 = g.operand((3,), "w")
    q3 = g.op(RESHAPE, [qh, sh3], shape=[H,T,d])
    k3 = g.op(RESHAPE, [kh, sh3], shape=[H,T,d])
    v3 = g.op(RESHAPE, [vh, sh3], shape=[H,T,d])
    ax = g.operand((1,), "a"); ay = g.operand((1,), "a")
    sc = g.op(BATCH_MATMUL, [q3, k3, ax, ay], adj_x=0, adj_y=1)
    m  = g.operand((1,T,T), "w"); act = g.operand((1,), "a")
    msk = g.op(ADD, [sc, m, act])
    axis = g.operand((1,), "a")
    mx = g.op(REDUCE_MAX, [msk, axis, act], axis=2)
    sb = g.op(SUB, [msk, mx, act])
    ex = g.op(EXP, [sb])
    sm = g.op(REDUCE_SUM, [ex, axis, act], axis=2)
    sf = g.op(DIV, [ex, sm, act])
    o3 = g.op(BATCH_MATMUL, [sf, v3, ax, ax], adj_x=0, adj_y=0)
    sh4 = g.operand((4,), "w")
    o4 = g.op(RESHAPE, [o3, sh4], shape=[1,H,T,d])
    p2 = g.operand((4,), "w")
    fin = g.op(TRANSPOSE, [o4, p2], perm=[0,2,1,3])
    print("attention 链: %d 个算子, 输出 %s" % (g.validate(), g.shapes[fin]))
    assert g.shapes[fin] == (1,T,H,d)

    # validate_txt: 手写图描述（perm 是 INT32 常量，必须走 w 文件；a 只能表示标量）
    import tempfile, os as _os
    pf = _os.path.join(tempfile.gettempdir(), "si_perm.bin")
    open(pf, "wb").write(struct.pack("<4i", 0, 2, 1, 3))

    def rdf(pth):
        try: return open(pth, "rb").read()
        except Exception: return None

    good = "OPD 0 8 4 1 %d %d %d 0 0 i\n" % (T,H,d)
    good += "OPD 1 4 1 4 0 0 w " + pf + "\n"
    good += "OPD 2 8 4 1 %d %d %d 0 0 o\n" % (H,T,d)
    good += "IO 1 0 1 2\nOP 37 2 0 1 1 2\nEND\n"
    e1 = validate_txt(good, rdf)
    print("validate_txt 正确图: %d 个错误" % len(e1))
    assert not e1, e1

    # 故意把输出声明改成转置前的形状，必须被抓到
    badtxt = good.replace("OPD 2 8 4 1 %d %d %d" % (H,T,d), "OPD 2 8 4 1 %d %d %d" % (T,H,d))
    e2 = validate_txt(badtxt, rdf)
    print("validate_txt 错误图: %d 个错误" % len(e2))
    assert e2, "应该抓到错误"
    print("  抓到:", e2[0])
    print("\n全部通过")
