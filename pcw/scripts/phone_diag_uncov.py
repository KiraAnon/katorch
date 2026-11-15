"""列出未被标定覆盖的激活 operand，以及它们由哪个算子产生/消费。"""
import json

G = "/data/local/tmp/q16b00/g.txt"
CAL = "/data/local/tmp/calib_scale.json"
tbl = json.load(open(CAL))

opd, ops = {}, []
for ln in open(G):
    if ln.startswith("OPD "):
        tok = ln.split()
        nd = int(tok[3])
        opd[int(tok[1])] = (int(tok[2]), " ".join(tok[6 + nd:]), tuple(int(v) for v in tok[4:4+nd]))
    elif ln.startswith("OP "):
        tok = ln.split()
        code, ni = int(tok[1]), int(tok[2])
        ins = [int(v) for v in tok[3:3+ni]]
        no = int(tok[3+ni])
        outs = [int(v) for v in tok[4+ni:4+ni+no]]
        ops.append((code, ins, outs))

prod = {}
for code, ins, outs in ops:
    for o in outs:
        prod[o] = code

acts = [o for o, v in opd.items() if v[0] == 7]
unc = [o for o in acts if str(o) not in tbl]
print("激活 %d 个，未标定 %d 个" % (len(acts), len(unc)))

from collections import Counter
c = Counter(prod.get(o, -1) for o in unc)
print("\n未标定的按产生算子分类 (opcode: 个数):")
for code, n in c.most_common(12):
    print("   %-6d %d" % (code, n))

print("\n抽样 12 个未标定的:")
for o in sorted(unc)[:12]:
    ty, tail, shp = opd[o]
    print("   id=%-5d 产生者=%-5s shape=%-22s kind=%s"
          % (o, prod.get(o, "?"), shp, tail[:1]))

# 同形状的已标定项能否借用
print("\n同形状的可借用尺度（取最大，保守）:")
from collections import defaultdict
byshape = defaultdict(list)
for o, v in opd.items():
    if v[0] == 7:
        byshape[v[2]].append(o)
borrow = 0
for o in unc:
    shp = opd[o][2]
    cands = [float(tbl[str(x)]) for x in byshape[shp] if str(x) in tbl]
    if cands:
        borrow += 1
print("   可借用的: %d / %d" % (borrow, len(unc)))
