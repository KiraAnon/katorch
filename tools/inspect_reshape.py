import numpy as np, os, re

D = "/data/data/com.termux/files/home/npt/e2eper512/b00"
G = os.path.join(D, "g.txt")

# 找所有 RESHAPE
opd = {}
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD":
        nd = int(p[3])
        opd[int(p[1])] = dict(ty=int(p[2]),
                              dims=[int(x) for x in p[4:4 + nd]],
                              tail=" ".join(p[4 + nd:]))

print("=== 每个 RESHAPE 的输入/输出/形状文件 ===")
n = 0
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP" or int(p[1]) != 22:
        continue
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    a = opd.get(ins[0], {})
    s = opd.get(ins[1], {})
    o = opd.get(outs[0], {})
    sf = s.get("tail", "")
    m = re.search(r"(\S+\.bin)", sf)
    content = None
    if m:
        fp = m.group(1)
        if os.path.isfile(fp):
            content = np.fromfile(fp, dtype=np.int32).tolist()
    print("  in %-22s -> out %-22s" % (a.get("dims"), o.get("dims")))
    print("     shape文件 %s = %s" % (os.path.basename(m.group(1)) if m else "?", content))
    n += 1
    if n >= 6:
        break
