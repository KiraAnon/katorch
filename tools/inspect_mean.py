import os
import numpy as np

D = "/data/data/com.termux/files/home/npt/e2eper512/b00"
G = os.path.join(D, "g.txt")

opd = {}
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD":
        nd = int(p[3])
        opd[int(p[1])] = dict(ty=int(p[2]), dims=[int(x) for x in p[4:4 + nd]],
                              tail=" ".join(p[4 + nd:]))

print("=== 真实图里的 MEAN (31) ===")
n = 0
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP" or int(p[1]) != 31:
        continue
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    print("  OP 行: ni=%d ins=%s -> outs=%s" % (ni, ins, outs))
    for i, oid in enumerate(ins):
        d = opd.get(oid, {})
        print("     in[%d] id=%-5d ty=%-2s dims=%s tail=%s" %
              (i, oid, d.get("ty"), d.get("dims"), d.get("tail", "")[:30]))
    for oid in outs:
        d = opd.get(oid, {})
        print("     out   id=%-5d ty=%-2s dims=%s" % (oid, d.get("ty"), d.get("dims")))
    n += 1
    if n >= 2:
        break

print()
print("=== 顺带看 SQRT(88) / EXP(49) / RSQRT(83) 的输入个数 ===")
for code, nm in ((88, "SQRT"), (49, "EXP"), (83, "RSQRT")):
    for ln in open(G, encoding="utf-8", errors="replace"):
        p = ln.split()
        if p and p[0] == "OP" and int(p[1]) == code:
            print("  %-6s %s" % (nm, ln.strip()))
            break
