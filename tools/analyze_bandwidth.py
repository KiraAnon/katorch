"""统计算子的数据搬运量，找出真正的带宽瓶颈在哪。"""
import collections, sys

NAMES = {0: "ADD", 9: "FC", 18: "MUL", 22: "RESHAPE", 31: "MEAN", 37: "TRANSPOSE",
         83: "RSQRT", 102: "BATCH_MATMUL", 14: "LOGISTIC", 28: "TANH",
         65: "MAXIMUM", 66: "MINIMUM", 67: "NEG", 30: "DIV", 49: "EXP", 36: "SUB",
         6: "DEQUANTIZE", 72: "QUANTIZE", 77: "REDUCE_MAX", 80: "REDUCE_SUM"}

G = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/data/com.termux/files/home/npt/e2eper512/b00/g.txt"

opd = {}
dims = {}
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD":
        nd = int(p[3])
        d = [int(x) for x in p[4:4 + nd]]
        dims[int(p[1])] = d
        n = 1
        for x in d:
            n *= x
        opd[int(p[1])] = n

# 统计每个算子的读+写字节数
per_op = collections.defaultdict(lambda: [0, 0])   # code -> [次数, 字节]
total_bytes = 0
big_ops = []

for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP":
        continue
    code = int(p[1])
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    nb = 0
    for o in ins:
        nb += opd.get(o, 0) * 2
    for o in outs:
        nb += opd.get(o, 0) * 2
    per_op[code][0] += 1
    per_op[code][1] += nb
    total_bytes += nb
    big_ops.append((nb, code, [dims.get(o) for o in ins][:2], [dims.get(o) for o in outs][:1]))

print("图:", G)
print("总算子:", sum(v[0] for v in per_op.values()))
print("总搬运: %.1f MB" % (total_bytes / 1e6))
print()
print("%-14s %-6s %-12s %s" % ("算子", "次数", "搬运MB", "占比"))
for code, (n, nb) in sorted(per_op.items(), key=lambda x: -x[1][1]):
    print("%-14s %-6d %-12.1f %.1f%%" % (NAMES.get(code, code), n, nb / 1e6, nb / total_bytes * 100))

print()
print("=== 单次搬运最大的 10 个算子 ===")
big_ops.sort(reverse=True)
for nb, code, ins, outs in big_ops[:10]:
    print("  %-14s in=%s out=%s  %.2f MB" % (NAMES.get(code, code), ins, outs, nb / 1e6))

print()
print("=== 按实测 773ms/块 反推带宽 ===")
print("  %.1f MB / 0.773s = %.2f GB/s" % (total_bytes / 1e6, total_bytes / 1e9 / 0.773))
