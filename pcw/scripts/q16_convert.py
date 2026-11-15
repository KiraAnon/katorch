"""把 fp16 的 g.txt 转成 W8A16 图。独立后处理，不修改 lowering.py。

用法:
    python q16_convert.py <源目录> <输出目录> [scale_table.json]

量化配方（对齐官方 miao-harem NPU 实现）:
  · FC 权重        -> INT8 per-output-channel (type 11 + QCH)
        sw[o] = max|W[o]| / 127 ; q = clip(rint(W/sw), -127, 127)
  · FC bias        -> INT32 (type 4),  q = rint(b / (sx * sw))
  · 其它 FP16 常量 -> INT16 (type 7)
  · 激活           -> INT16 (type 7)
  · LOGISTIC/SOFTMAX 输出 scale 固定 1/32768
  · SHAPE_ONLY(22,37,86) 输出继承输入 scale（否则整图不可映射）
"""
import os, sys, json
import numpy as np

T_ACT, T_W8, T_I32, T_F16 = 7, 11, 4, 8
FC_OPCODE = 9
SHAPE_ONLY = (22, 37, 86)
PROB_OPS = (14, 25)


def parse_opd(ln):
    tok = ln.split()
    oid, ty, nd = int(tok[1]), int(tok[2]), int(tok[3])
    dims = [int(v) for v in tok[4:4 + nd]]
    rest = tok[6 + nd:]
    return oid, ty, nd, dims, (rest[0] if rest else ""), (rest[1:] if len(rest) > 1 else [])


def parse_op(ln):
    tok = ln.split()
    code, ni = int(tok[1]), int(tok[2])
    ins = [int(v) for v in tok[3:3 + ni]]
    no = int(tok[3 + ni])
    outs = [int(v) for v in tok[4 + ni:4 + ni + no]]
    return code, ins, outs


def main():
    src, dst = sys.argv[1], sys.argv[2]
    table_path = sys.argv[3] if len(sys.argv) > 3 else None
    lines = [l.rstrip("\n") for l in open(os.path.join(src, "g.txt"))]
    os.makedirs(dst, exist_ok=True)

    tbl = {}
    if table_path and os.path.exists(table_path):
        tbl = {str(k): float(v) for k, v in json.load(open(table_path)).items()}
    DEF = float(os.environ.get("Q16_ACT_SCALE", "0.02"))   # 未标定激活的安全值(覆盖±655)
    DEFC = float(os.environ.get("Q16_CONST_SCALE", "1e-3"))

    info = {}
    for ln in lines:
        if ln.startswith("OPD "):
            oid, ty, nd, dims, kind, payload = parse_opd(ln)
            info[oid] = dict(ty=ty, nd=nd, dims=dims, kind=kind, payload=payload)

    op_list = [parse_op(l) for l in lines if l.startswith("OP ")]
    fc_ins, fc_bias_of, is_bias = {}, {}, set()
    for code, ins, outs in op_list:
        if code == FC_OPCODE and len(ins) >= 3:
            fc_ins[ins[1]] = (ins[0], ins[2])
            fc_bias_of[ins[2]] = ins[1]
            is_bias.add(ins[2])
    fc_weight_ids = set(fc_ins)

    # 未标定 operand 的兜底：同形状已标定张量的尺度的中位数
    from collections import defaultdict
    _byshape = defaultdict(list)
    for oid, o in info.items():
        if o["ty"] == T_F16 and o["kind"] != "w" and str(oid) in tbl:
            _byshape[tuple(o["dims"])].append(tbl[str(oid)])
    shape_hint = {k: (sorted(v)[len(v) // 2] if v else DEF) for k, v in _byshape.items()}

    scale_of = {}
    for oid, o in info.items():
        if o["ty"] != T_F16:
            continue
        if o["kind"] == "w":
            arr0 = np.fromfile(o["payload"][0], dtype=np.float16).astype(np.float32)
            mx0 = float(np.abs(arr0).max()) if arr0.size else 0.0
            scale_of[oid] = max(mx0 / 32767.0, 1e-7)
        else:
            v = tbl.get(str(oid))
            if v is None:
                v = shape_hint.get(tuple(o["dims"]), DEF)
            scale_of[oid] = v

    # ---------- 按算子语义推导尺度（只修正标定失效的那些）----------
    # FX 节点到 operand 的映射在个别位置会错位，表现是标定值落在下限 1e-8（等于没数据）。
    # 对这些 operand 用图上语义推：MUL 乘积、ADD/SUB 取大、FC-reduce 求和放大 N 倍。
    BROKEN = 5e-8

    def broken(oid):
        return scale_of.get(oid, 0.0) <= BROKEN

    fc_weight_of = {v: w for w, (i, b) in fc_ins.items()}   # 占位，下面重建

    # 识别 FC-reduce：权重形如 (1,N) 且所有元素相等
    reduce_fc = {}
    for oid, o in info.items():
        if o["ty"] == T_F16 and o["kind"] == "w" and len(o["dims"]) == 2 and o["dims"][0] == 1:
            arr = np.fromfile(o["payload"][0], dtype=np.float16).astype(np.float32)
            if arr.size and np.allclose(arr, arr.flat[0]):
                reduce_fc[oid] = (int(o["dims"][1]), float(arr.flat[0]))

    for _ in range(10):
        changed = False
        for code, ins, outs in op_list:
            data = [v for v in ins if info.get(v, {}).get("ty", 0) not in (1, T_I32)]
            if not data or not outs:
                continue
            o = outs[0]
            if not broken(o):
                continue
            s = None
            if code == 18 and len(data) >= 2:              # MUL
                a, b = scale_of.get(data[0]), scale_of.get(data[1])
                if a and b:
                    s = max(a * b * 32767.0, 1e-7)
            elif code in (0, 36):                          # ADD / SUB
                cand = [scale_of.get(v) for v in data]
                cand = [c for c in cand if c]
                if cand:
                    s = max(cand)
            elif code == FC_OPCODE and data:
                w = data[1] if len(data) > 1 else None
                if w in reduce_fc:
                    n, wv = reduce_fc[w]
                    si = scale_of.get(data[0])
                    if si:
                        s = max(si * n * abs(wv), 1e-7)
            if s is not None and s > BROKEN:
                scale_of[o] = min(s, 2.0)
                changed = True
        if not changed:
            break

    # ---------- FP16 norm 岛 ----------
    # DEQUANTIZE(6) 的输出到 QUANTIZE(72) 的输入之间，全部保持 FP16（scale=0）。
    # 这样归一化那些对精度敏感的算子就不用碰 INT16 的坏归约/rsqrt。
    ISLAND_IN, ISLAND_OUT = 6, 72
    island = set()
    frontier = [v for code, ins, outs in op_list if code == ISLAND_IN for v in outs]
    while frontier:
        v = frontier.pop()
        if v in island:
            continue
        island.add(v)
        for code, ins, outs in op_list:
            if code == ISLAND_OUT:
                continue
            if v in ins:
                for o in outs:
                    if o not in island:
                        frontier.append(o)

    # 反向：岛内算子消费的常量也必须是 FP16（否则同一算子出现 INT16/FP16 混用）
    for _ in range(6):
        add = set()
        for code, ins2, outs2 in op_list:
            if code == ISLAND_IN or code == ISLAND_OUT:
                continue
            if any(o in island for o in outs2):
                for v in ins2:
                    if v not in island and info.get(v, {}).get("ty") == T_F16:
                        add.add(v)
        if not add:
            break
        island |= add
    # DEQUANTIZE 的输入、QUANTIZE 的输出都不属于岛
    for code, ins2, outs2 in op_list:
        if code == ISLAND_IN:
            island -= set(ins2)
        if code == ISLAND_OUT:
            island -= set(outs2)

    # 岛内的 FC：权重/bias 必须保持 FP16，不能转 INT8
    island_fc_w, island_fc_b = set(), set()
    for code, ins2, outs2 in op_list:
        if code == FC_OPCODE and len(ins2) >= 3 and ins2[0] in island:
            island_fc_w.add(ins2[1]); island_fc_b.add(ins2[2])
    if island_fc_w:
        fc_weight_ids -= island_fc_w
        is_bias -= island_fc_b

    # ---------- 尺度微调 ----------
    # 实测 MDLA 能正确处理不同尺度之间的重量化（ADD/MUL/SUB 输入尺度差 8 倍，误差 0.003%），
    # 所以每个 operand 直接用自己标定出来的尺度即可，不做全局统一。
    # 唯一必须一致的是 SHAPE_ONLY 算子（reshape/transpose），否则整图不可映射。
    for code, ins, outs in op_list:
        if code in PROB_OPS:
            for v in outs:
                if v in scale_of:
                    scale_of[v] = 1.0 / 32768.0

    hr = float(os.environ.get("Q16_HEADROOM", "1.6"))   # 安全余量，防饱和
    for v in scale_of:
        scale_of[v] *= hr

    lo = float(os.environ.get("Q16_SMIN", "1e-7"))
    hi = float(os.environ.get("Q16_SMAX", "2.0"))
    for v in scale_of:
        scale_of[v] = min(max(scale_of[v], lo), hi)

    for _ in range(8):
        changed = False
        for code, ins, outs in op_list:
            if code not in SHAPE_ONLY or not ins or not outs:
                continue
            s = scale_of.get(ins[0])
            if s is None:
                continue
            for v in outs:
                if v in scale_of and scale_of[v] != s:
                    scale_of[v] = s; changed = True
        if not changed:
            break
    out_lines, qch, stats = [], [], {"w8": 0, "bias": 0, "act": 0, "const16": 0, "keep": 0}
    for ln in lines:
        if not ln.startswith("OPD "):
            out_lines.append(ln)
            continue
        oid, ty, nd, dims, kind, payload = parse_opd(ln)
        if ty != T_F16:
            out_lines.append(ln); stats["keep"] += 1; continue

        if oid in island:
            # FP16 norm 岛内部：原样保留 fp16
            out_lines.append(ln); stats["island"] = stats.get("island", 0) + 1; continue

        if kind == "w":
            base = os.path.basename(payload[0])
            if oid in is_bias:
                w_id = fc_bias_of[oid]
                in_id = fc_ins[w_id][0]
                wbase = os.path.basename(info[w_id]["payload"][0])
                sw = np.fromfile(os.path.join(dst, wbase + ".sc"), dtype=np.float32)
                sx = np.float32(scale_of.get(in_id, DEF))
                bias = np.fromfile(payload[0], dtype=np.float16).astype(np.float32)
                q = np.clip(np.rint(bias / (sx * sw)), -2**31, 2**31 - 1).astype(np.int32)
                bp = os.path.join(dst, base + ".i32")
                q.tofile(bp)
                out_lines.append("OPD %d %d %d %s 0 0 w %s"
                                 % (oid, T_I32, nd, " ".join(map(str, dims)), bp))
                stats["bias"] += 1
            elif len(dims) == 2 and oid in fc_weight_ids:
                arr = np.fromfile(payload[0], dtype=np.float16).astype(np.float32).reshape(dims)
                mx = np.max(np.abs(arr), axis=1)
                sw = np.where(mx > 0, mx / 127.0, 1.0).astype(np.float32)
                q = np.clip(np.rint(arr / sw[:, None]), -127, 127).astype(np.int8)
                wp = os.path.join(dst, base + ".q8")
                sp = os.path.join(dst, base + ".sc")
                q.tofile(wp); sw.tofile(sp)
                out_lines.append("OPD %d %d %d %s 0 0 w %s"
                                 % (oid, T_W8, nd, " ".join(map(str, dims)), wp))
                qch.append("QCH %d 0 %d %s" % (oid, len(sw), sp))
                stats["w8"] += 1
            else:
                arr = np.fromfile(payload[0], dtype=np.float16).astype(np.float32)
                s = np.float32(scale_of.get(oid, DEFC))
                if arr.size and np.abs(arr / s).max() > 32767:
                    s = np.float32(np.abs(arr).max() / 32767.0)
                q = np.clip(np.rint(arr / s), -32767, 32767).astype(np.int16)
                cp = os.path.join(dst, base + ".q16")
                q.tofile(cp)
                out_lines.append("OPD %d %d %d %s %.9g 0 w %s"
                                 % (oid, T_ACT, nd, " ".join(map(str, dims)), s, cp))
                stats["const16"] += 1
            continue

        s = scale_of.get(oid, DEF)
        out_lines.append("OPD %d %d %d %s %.9g 0 %s"
                         % (oid, T_ACT, nd, " ".join(map(str, dims)), s,
                            " ".join([kind] + payload)))
        stats["act"] += 1

    merged, done = [], False
    for ln in out_lines:
        if not done and ln.startswith("OP "):
            merged.extend(qch); done = True
        merged.append(ln)
    if not done:
        merged.extend(qch)

    # 输出元数据：调用方必须知道每个输入/输出的类型和尺度才能正确读写
    io_txt = [l for l in merged if l.startswith("IO ")]
    meta = {"inputs": [], "outputs": []}
    if io_txt:
        tk = io_txt[0].split()
        nin = int(tk[1])
        ids_in = [int(v) for v in tk[2:2 + nin]]
        nout = int(tk[2 + nin])
        ids_out = [int(v) for v in tk[3 + nin:3 + nin + nout]]
        for key, ids in (("inputs", ids_in), ("outputs", ids_out)):
            for o in ids:
                info_o = info.get(o, {})
                ty = T_ACT
                sc = scale_of.get(o, 0.0)
                if o in island:
                    ty, sc = T_F16, 0.0
                elif info_o.get("kind") == "w":
                    ty, sc = T_F16, 0.0
                meta[key].append({"t": int(ty), "s": float(sc)})
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)

    open(os.path.join(dst, "g.txt"), "w").write("\n".join(merged) + "\n")
    print(json.dumps(stats))
    print("-> %s/g.txt (%d 行, %d QCH)" % (dst, len(merged), len(qch)))


if __name__ == "__main__":
    main()
