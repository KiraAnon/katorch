"""看清图里 FC / BATCH_MATMUL 的分布，理解层结构，找复用机会。"""
import os

D = "/data/data/com.termux/files/home/npt/e2eper512/b00"
G = os.path.join(D, "g.txt")

opd = {}
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD":
        nd = int(p[3])
        opd[int(p[1])] = dict(ty=int(p[2]),
                              dims=[int(x) for x in p[4:4 + nd]],
                              tail=" ".join(p[4 + nd:]))

print("=== 所有 FC（28 个）===")
n = 0
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP" or int(p[1]) != 9:
        continue
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    a = opd.get(ins[0], {})
    w = opd.get(ins[1], {})
    o = opd.get(outs[0], {})
    print("  %2d: in=%-18s w=%-18s out=%-18s" %
          (n, a.get("dims"), w.get("dims"), o.get("dims")))
    n += 1

print()
print("=== 所有 BATCH_MATMUL（28 个）===")
n = 0
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP" or int(p[1]) != 102:
        continue
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    a = opd.get(ins[0], {})
    b = opd.get(ins[1], {})
    o = opd.get(outs[0], {})
    print("  %2d: a=%-20s b=%-20s out=%-20s" %
          (n, a.get("dims"), b.get("dims"), o.get("dims")))
    n += 1
