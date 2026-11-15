"""分析 141 个 MUL 的形状和用途，看哪些能融合掉。"""
import collections, os

D = "/data/data/com.termux/files/home/npt/e2eper512/b00"
G = os.path.join(D, "g.txt")

opd = {}
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD":
        nd = int(p[3])
        opd[int(p[1])] = dict(dims=[int(x) for x in p[4:4 + nd]],
                              tail=" ".join(p[4 + nd:]))

kinds = collections.Counter()
examples = {}

for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP" or int(p[1]) != 18:
        continue
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    da = opd.get(ins[0], {}).get("dims", [])
    db = opd.get(ins[1], {}).get("dims", [])
    # 分类：标量乘 / 广播乘 / 同形状乘
    if db == [1]:
        k = "标量乘"
    elif len(db) == 0:
        k = "?（无维度）"
    elif len(db) < len(da) or (db and da and db[-1] == da[-1] and len(db) != len(da)):
        k = "广播乘 %s" % (db,)
    elif tuple(da) == tuple(db):
        k = "同形状 %s" % (da,)
    else:
        k = "其他 in=%s w=%s" % (da, db)
    kinds[k] += 1
    if k not in examples:
        examples[k] = (da, db, opd.get(ins[1], {}).get("tail", "")[:40])

print("=== 141 个 MUL 的分类 ===")
for k, n in kinds.most_common():
    print("  %-40s %3d" % (k, n))
    if k in examples:
        da, db, tl = examples[k]
        print("        例: in=%s 乘数=%s %s" % (da, db, ("[" + tl + "]") if tl else ""))
