"""按「每个算子处理的字节数」重排，找出真正的大头。"""
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
        opd[int(p[1])] = n

# 按「单算子处理字节数」分档统计
buckets = collections.Counter()
bucket_time = collections.Counter()
total_ops = 0
big = []

for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p or p[0] != "OP":
        continue
    code = int(p[1])
    ni = int(p[2])
    ins = [int(x) for x in p[3:3 + ni]]
    no = int(p[3 + ni])
    outs = [int(x) for x in p[4 + ni:4 + ni + no]]
    nb = sum(opd.get(o, 0) for o in ins) * 2 + sum(opd.get(o, 0) for o in outs) * 2
    total_ops += 1
    if nb >= 2_000_000:
        b = "≥2MB"
    elif nb >= 500_000:
        b = "0.5-2MB"
    elif nb >= 50_000:
        b = "50KB-0.5MB"
    else:
        b = "<50KB"
    buckets[b] += 1
    # RESHAPE 免费，不计入时间
    if code != 22:
        bucket_time[b] += nb
    if nb >= 2_000_000:
        big.append((nb, code))

print("=== 按单算子数据量分档 ===")
for b in ("≥2MB", "0.5-2MB", "50KB-0.5MB", "<50KB"):
    print("  %-14s %3d 个" % (b, buckets[b]))
print()

print("=== 大算子（≥2MB）明细 ===")
c = collections.Counter(code for nb, code in big)
for code, n in c.most_common():
    print("  %-14s %2d 个" % (NAMES.get(code, code), n))

print()
print("=== 估算（RESHAPE 免费，其余按 4.2MB→1.4ms 线性）===")
GBps = 4.2e6 * 3 / 1.39e-3 / 1e9     # 实测带宽
print("  实测带宽 %.1f GB/s（读2写1）" % GBps)
est = sum(v for v in bucket_time.values()) / (GBps * 1e9) * 1000
print("  估算总时间 %.0f ms（实测 773 ms）" % est)
print()
print("  大算子搬运 %.1f MB (%.0f%%)" %
      (bucket_time["≥2MB"] / 1e6, bucket_time["≥2MB"] / sum(bucket_time.values()) * 100))
