"""统计两件事：
1. cross-attn K/V 复用能省多少（ctx 8 步不变）
2. 算子融合上限：有多少算子的输出只被 1 个下游使用
"""
import collections, os

D = "/data/data/com.termux/files/home/npt/e2eper512/b00"
G = os.path.join(D, "g.txt")
NAMES = {0: "ADD", 9: "FC", 18: "MUL", 22: "RESHAPE", 31: "MEAN", 37: "TRANSPOSE",
         83: "RSQRT", 102: "BATCH_MATMUL", 14: "LOGISTIC", 28: "TANH",
         65: "MAXIMUM", 66: "MINIMUM", 67: "NEG", 30: "DIV", 49: "EXP", 36: "SUB",
         77: "REDUCE_MAX", 80: "REDUCE_SUM"}

opd = {}
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if p and p[0] == "OPD":
        nd = int(p[3])
        d = [int(x) for x in p[4:4 + nd]]
        n = 1
        for x in d:
            n *= x
        opd[int(p[1])] = dict(dims=d, n=n, tail=" ".join(p[4 + nd:]))

ops = []
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP":
        continue
    code = int(p[1])
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    ops.append((code, ins, outs))

# ---------- 1. cross-attn K/V 复用 ----------
print("=== 1. cross-attn K/V 复用（ctx 8 步不变）===")
kv_ops = []
for i, (code, ins, outs) in enumerate(ops):
    if code != 102 or len(ins) < 2:
        continue
    a = opd.get(ins[0], {})
    b = opd.get(ins[1], {})
    da, db = a.get("dims", []), b.get("dims", [])
    # a 的第 2 维是 ctx 长度 512，b 是权重
    if len(da) == 3 and da[1] == 512:
        kv_ops.append((i, da, db))
print("  找到 %d 个 K/V 投影算子:" % len(kv_ops))
tot_kv = 0
for i, da, db in kv_ops[:4]:
    nb = (opd[ops[i][1][0]]["n"] + opd[ops[i][1][1]]["n"] + opd[ops[i][2][0]]["n"]) * 2
    tot_kv += nb
    print("    op#%d  a=%s b=%s  读+写 %.1f MB" % (i, da, db, nb / 1e6))
if len(kv_ops) > 4:
    per = tot_kv / min(4, len(kv_ops))
    tot_kv = per * len(kv_ops)
    print("    ... 共 %d 个，合计 %.1f MB" % (len(kv_ops), tot_kv / 1e6))

total_bytes = 0
for code, ins, outs in ops:
    if code == 22:      # RESHAPE 免费
        continue
    for o in ins:
        total_bytes += opd.get(o, {}).get("n", 0) * 2
    for o in outs:
        total_bytes += opd.get(o, {}).get("n", 0) * 2

print()
print("  单块总搬运 %.1f MB" % (total_bytes / 1e6))
print("  K/V 占比 %.2f%%" % (tot_kv / total_bytes * 100))
print("  8 步能省 %.1f MB (%.2f%%)" % (tot_kv * 7 / 1e6, tot_kv * 7 / total_bytes / 8 * 100))

# ---------- 2. 可融合性 ----------
print()
print("=== 2. 算子融合上限 ===")
users = collections.defaultdict(list)      # operand -> [消费它的算子索引]
for i, (code, ins, outs) in enumerate(ops):
    for o in ins:
        users[o].append(i)

# 一个算子的输出如果只被一个下游用，且中间没有旁路，就可以融合
fusable = 0
fusable_bytes = 0
for i, (code, ins, outs) in enumerate(ops):
    if len(outs) != 1:
        continue
    o = outs[0]
    u = users.get(o, [])
    if len(u) == 1 and u[0] != i:
        fusable += 1
        fusable_bytes += opd.get(o, {}).get("n", 0) * 2     # 省一次写+读

print("  输出只被 1 个下游消费的算子: %d / %d (%.0f%%)" %
      (fusable, len(ops), fusable / len(ops) * 100))
print("  这些中间结果共 %.1f MB" % (fusable_bytes / 1e6))
print("  若全部融合，单块能省 %.1f MB (%.1f%%)" %
      (fusable_bytes * 2 / 1e6, fusable_bytes * 2 / total_bytes * 100))
print()
print("  折算成时间（8.7 GB/s）: 单步 %.2f 秒，8 步 %.1f 秒" %
      (fusable_bytes * 2 / 8.7e9 * 14, fusable_bytes * 2 / 8.7e9 * 14 * 8))
