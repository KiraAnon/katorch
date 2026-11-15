"""校验转换后的 W8A16 图，找出非法 operand。"""
import os, sys

G = sys.argv[1] if len(sys.argv) > 1 else "/data/local/tmp/q16b00/g.txt"
lines = [l.rstrip("\n") for l in open(G)]

T_ACT, T_W8, T_I32, T_F16 = 7, 11, 4, 8

opds = {}
for ln in lines:
    if not ln.startswith("OPD "):
        continue
    tok = ln.split()
    oid, ty, nd = int(tok[1]), int(tok[2]), int(tok[3])
    dims = [int(v) for v in tok[4:4 + nd]]
    scale = float(tok[4 + nd]); zp = int(tok[5 + nd])
    rest = tok[6 + nd:]
    opds[oid] = dict(ty=ty, nd=nd, dims=dims, scale=scale, zp=zp,
                     kind=rest[0] if rest else "", payload=rest[1:])

qch = {}
for ln in lines:
    if ln.startswith("QCH "):
        tok = ln.split()
        qch[int(tok[1])] = dict(cdim=int(tok[2]), n=int(tok[3]), path=tok[4])

print("operand 总数 %d, QCH %d" % (len(opds), len(qch)))

bad = []
for oid, o in sorted(opds.items()):
    ty = o["ty"]
    if ty == T_ACT:
        if not (o["scale"] > 0) or o["scale"] != o["scale"]:
            bad.append((oid, "ACT scale 非法", o["scale"]))
    elif ty == T_W8:
        q = qch.get(oid)
        if not q:
            bad.append((oid, "W8 缺 QCH", ""))
        else:
            if q["cdim"] >= len(o["dims"]):
                bad.append((oid, "QCH channelDim 越界", q["cdim"]))
            elif q["n"] != o["dims"][q["cdim"]]:
                bad.append((oid, "QCH 数量不匹配 dims[%d]=%d vs %d"
                            % (q["cdim"], o["dims"][q["cdim"]], q["n"]), ""))
            elif not os.path.exists(q["path"]):
                bad.append((oid, "QCH scale 文件不存在", q["path"]))
    elif ty == T_F16:
        bad.append((oid, "仍是 FP16（会与 INT16 混用）", o["kind"]))

print("\n=== 问题 operand: %d 个 ===" % len(bad))
for b in bad[:25]:
    print("  oid=%-5d %-40s %s" % b)

# 检查 QCH 里有没有全零/非有限 scale
print("\n=== QCH scale 数组抽样检查 ===")
import struct
nbad = 0
for oid, q in list(qch.items())[:200]:
    if not os.path.exists(q["path"]):
        continue
    raw = open(q["path"], "rb").read()
    vals = struct.unpack("<%df" % (len(raw) // 4), raw)
    zero = sum(1 for v in vals if v == 0)
    nonfinite = sum(1 for v in vals if not (v == v and abs(v) < 1e30))
    if zero or nonfinite:
        nbad += 1
        if nbad <= 5:
            print("  oid=%d  零值 scale %d 个, 非有限 %d 个, 首个=%s"
                  % (oid, zero, nonfinite, vals[:4]))
print("  有问题的 QCH: %d / %d" % (nbad, min(len(qch), 200)))
