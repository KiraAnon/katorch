"""检查转换后图里 output operand 的尺度，以及标定表的覆盖情况。"""
import json, os

G = "/data/local/tmp/q16b00/g.txt"
CAL = "/data/local/tmp/calib_scale.json"

opd = {}
last_op = None
for ln in open(G):
    if ln.startswith("OPD "):
        tok = ln.split()
        nd = int(tok[3])
        opd[int(tok[1])] = (int(tok[2]), float(tok[4 + nd]), " ".join(tok[6 + nd:]))
    elif ln.startswith("OP "):
        last_op = ln.strip()

tok = last_op.split()
ni = int(tok[2]); no = int(tok[3 + ni])
out_ids = [int(v) for v in tok[4 + ni:4 + ni + no]]
print("最后一个算子: %s" % last_op)
for o in out_ids:
    ty, sc, tail = opd.get(o, (None, None, ""))
    print("  输出 operand %d: type=%s scale=%.6g tail=%s" % (o, ty, sc, tail))

tbl = json.load(open(CAL))
print("\n标定表 %d 项" % len(tbl))
for o in out_ids:
    print("  标定表里有 %d 吗: %s" % (o, tbl.get(str(o), "没有")))

# 统计覆盖
acts = [oid for oid, v in opd.items() if v[0] == 7 and v[2].startswith(("i", "o"))]
covered = sum(1 for oid in acts if str(oid) in tbl)
print("\n激活 operand %d 个，标定覆盖 %d 个" % (len(acts), covered))

# 抽样对比：标定值 vs 实际用的
print("\n抽样（前 8 个激活）:")
for oid in sorted(acts)[:8]:
    ty, sc, tail = opd[oid]
    print("   id=%-4d 图里 scale=%-12.6g 标定=%-12s kind=%s"
          % (oid, sc, tbl.get(str(oid), "-"), tail[:1]))
