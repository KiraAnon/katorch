"""诊断岛边界：列出所有 DEQUANTIZE(6)/QUANTIZE(72) 及其输入输出 operand 的类型与尺度。"""
import json

G = "/data/local/tmp/q16i/g.txt"
CAL = "/data/local/tmp/calib_i.json"

opd = {}
ops = []
for ln in open(G):
    if ln.startswith("OPD "):
        t = ln.split()
        nd = int(t[3])
        opd[int(t[1])] = dict(ty=int(t[2]), nd=nd,
                              dims=[int(v) for v in t[4:4 + nd]],
                              scale=float(t[4 + nd]),
                              kind=t[6 + nd] if len(t) > 6 + nd else "")
    elif ln.startswith("OP "):
        t = ln.split()
        code, ni = int(t[1]), int(t[2])
        ins = [int(v) for v in t[3:3 + ni]]
        no = int(t[3 + ni])
        outs = [int(v) for v in t[4 + ni:4 + ni + no]]
        ops.append((code, ins, outs, ln.strip()))

tbl = json.load(open(CAL))
print("=== DEQUANTIZE(6) 边界 ===")
for code, ins, outs, raw in ops:
    if code != 6:
        continue
    for v in ins:
        o = opd[v]
        print("  输入 %-5d type=%-2d scale=%-12.6g shape=%s  标定=%s"
              % (v, o["ty"], o["scale"], o["dims"][:3], tbl.get(str(v), "-")))
    for v in outs:
        o = opd[v]
        print("  输出 %-5d type=%-2d scale=%-12.6g shape=%s" % (v, o["ty"], o["scale"], o["dims"][:3]))
    print("  --")

print("\n=== QUANTIZE(72) 边界 ===")
for code, ins, outs, raw in ops:
    if code != 72:
        continue
    for v in ins:
        o = opd[v]
        print("  输入 %-5d type=%-2d scale=%-12.6g shape=%s" % (v, o["ty"], o["scale"], o["dims"][:3]))
    for v in outs:
        o = opd[v]
        print("  输出 %-5d type=%-2d scale=%-12.6g shape=%s  标定=%s"
              % (v, o["ty"], o["scale"], o["dims"][:3], tbl.get(str(v), "-")))
    print("  --")

# 统计
import collections
c = collections.Counter(o["ty"] for o in opd.values())
print("\noperand 类型分布:", dict(sorted(c.items())))
