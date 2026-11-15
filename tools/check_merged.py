"""检查拼接后的图完整性：operand 定义是否齐全、引用是否越界。"""
import os, sys, collections

G = "/data/local/tmp/merge_test/g.txt"
SRC = "/data/data/com.termux/files/home/npt/e2eper512"

opd = {}
dims = {}
dup = []
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD":
        oid = int(p[1])
        nd = int(p[3])
        if oid in opd:
            dup.append(oid)
        opd[oid] = 1
        dims[oid] = [int(x) for x in p[4:4 + nd]]

print("operand 定义数:", len(opd))
print("重复定义:", dup[:10] if dup else "无")
print("ID 范围: %d .. %d" % (min(opd), max(opd)))

# 检查所有 OP 引用的 operand 是否都有定义
missing = collections.Counter()
nops = 0
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP":
        continue
    nops += 1
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    for o in ins + outs:
        if o not in opd:
            missing[o] += 1

print("算子数:", nops)
print("未定义的引用:", dict(list(missing.items())[:10]) if missing else "无")

print()
print("=== IO 行 ===")
for ln in open(G, encoding="utf-8", errors="replace"):
    if ln.startswith("IO "):
        print(" ", ln.strip())
        p = ln.split()
        ni = int(p[1])
        for o in [int(x) for x in p[2:2 + ni]]:
            print("    in  %d -> %s  %s" % (o, dims.get(o), "OK" if o in opd else "缺失"))
        no = int(p[2 + ni])
        for o in [int(x) for x in p[3 + ni:3 + ni + no]]:
            print("    out %d -> %s  %s" % (o, dims.get(o), "OK" if o in opd else "缺失"))

# 和原图对比 operand 数
for b in ("b00", "b01"):
    ga = os.path.join(SRC, b, "g.txt")
    n = sum(1 for ln in open(ga, encoding="utf-8", errors="replace") if ln.startswith("OPD "))
    print("%s operand 数: %d" % (b, n))
print("拼接后应为: 1021(b00跳过?) + ... 实际 %d" % len(opd))
