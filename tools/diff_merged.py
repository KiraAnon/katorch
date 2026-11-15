"""对比拼接图和原图的 OPD，找出拼接引入的差异。"""
import os

SRC = "/data/data/com.termux/files/home/npt/e2eper512"
M = "/data/local/tmp/merge_test/g.txt"


def opds(gp):
    d = {}
    order = []
    for ln in open(gp, encoding="utf-8", errors="replace"):
        p = ln.split()
        if p and p[0] == "OPD":
            d[int(p[1])] = ln.strip()
            order.append(int(p[1]))
    return d, order


def ops(gp):
    out = []
    for ln in open(gp, encoding="utf-8", errors="replace"):
        p = ln.split()
        if p and p[0] == "OP":
            out.append(ln.strip())
    return out


d0, o0 = opds(SRC + "/b00/g.txt")
d1, o1 = opds(SRC + "/b01/g.txt")
dm, om = opds(M)

print("b00 OPD: %d, b01 OPD: %d, merged: %d" % (len(d0), len(d1), len(dm)))

# b00 部分应该完全一致
bad = 0
for k in d0:
    if d0[k] != dm.get(k):
        bad += 1
        if bad <= 3:
            print("  b00 差异 %d:\n    原: %s\n    新: %s" % (k, d0[k], dm.get(k)))
print("b00 部分差异: %d" % bad)

# b01 部分：id + 1029
bad2 = 0
for k in d1:
    if k < 8:
        continue
    nk = k + 1029
    if d1[k] != dm.get(nk):
        bad2 += 1
        if bad2 <= 3:
            print("  b01 差异 %d->%d:\n    原: %s\n    新: %s" % (k, nk, d1[k], dm.get(nk)))
print("b01 部分差异: %d" % bad2)

# 统计常量（有 w 的）
def nconst(gp):
    n = 0
    for ln in open(gp, encoding="utf-8", errors="replace"):
        p = ln.split()
        if p and p[0] == "OPD" and " w " in ln:
            n += 1
    return n

print()
print("常量数(b00): %d, (b01): %d, (merged): %d" % (nconst(SRC + "/b00/g.txt"),
                                                    nconst(SRC + "/b01/g.txt"), nconst(M)))

# 独立常量文件总大小
import re
tot = 0
seen = set()
for ln in open(M, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD" and " w " in ln:
        m = re.search(r"(\S+\.(bin|f32|dat))", ln)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            if os.path.isfile(m.group(1)):
                tot += os.path.getsize(m.group(1))
print("常量文件总大小: %.1f MB（%d 个不同文件）" % (tot / 1e6, len(seen)))
