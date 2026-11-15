"""分析单块图的算子构成，看为什么 773ms 远超 FC 的理论耗时。"""
import os, sys, collections

G = sys.argv[1] if len(sys.argv) > 1 else "/data/local/tmp/blobs512m/b00/g.txt"
if not os.path.isfile(G):
    # 试试找 g.txt
    for cand in ("/data/local/tmp/q16i/g.txt", "/data/local/tmp/blobs512m/b00/g.txt"):
        if os.path.isfile(cand):
            G = cand
            break
print("图文件:", G)
if not os.path.isfile(G):
    print("找不到 g.txt")
    raise SystemExit(1)

NAMES = {0: "ADD", 1: "AVGPOOL", 2: "CONCAT", 3: "CONV", 6: "DEQUANTIZE", 9: "FC",
         14: "LOGISTIC", 17: "MAXPOOL", 18: "MUL", 19: "RELU", 22: "RESHAPE",
         25: "SOFTMAX", 28: "TANH", 30: "DIV", 31: "MEAN", 36: "SUB",
         37: "TRANSPOSE", 38: "ABS", 45: "CAST", 49: "EXP", 60: "LOG",
         65: "MAXIMUM", 66: "MINIMUM", 67: "NEG", 72: "QUANTIZE",
         77: "REDUCE_MAX", 80: "REDUCE_SUM", 83: "RSQRT", 85: "SIN",
         88: "SQRT", 102: "BATCH_MATMUL"}

cnt = collections.Counter()
ops = []
opd = {}
for ln in open(G, encoding="utf-8", errors="replace"):
    p = ln.split()
    if not p:
        continue
    if p[0] == "OP":
        code = int(p[1])
        ni = int(p[2])
        ins = [int(x) for x in p[3:3 + ni]]
        no = int(p[3 + ni])
        outs = [int(x) for x in p[4 + ni:4 + ni + no]]
        cnt[code] += 1
        ops.append((code, ins, outs))
    elif p[0] == "OPD":
        nd = int(p[3])
        opd[int(p[1])] = dict(ty=int(p[2]), dims=[int(x) for x in p[4:4 + nd]],
                              scale=float(p[4 + nd]))

total = sum(cnt.values())
print("总算子数:", total)
print()
print("=== 算子分布 ===")
for code, n in cnt.most_common():
    print("  %-14s %3d" % (NAMES.get(code, str(code)), n))

print()
print("=== FC 的规模 ===")
fc_flops = 0
for code, ins, outs in ops:
    if code != 9:
        continue
    w = opd.get(ins[1], {})
    d = w.get("dims", [])
    i0 = opd.get(ins[0], {})
    di = i0.get("dims", [])
    if len(d) == 2:
        # 估算：输入 token 数 × 输出维
        nin = 1
        for x in di[:-1]:
            nin *= x
        K, N = d[0], d[1]
        fl = 2 * nin * K * N
        fc_flops += fl
        print("  in=%s w=%s  ->  %.2f GFLOP" % (di, d, fl / 1e9))
print()
print("FC 总 FLOPs: %.1f GFLOP" % (fc_flops / 1e9))
print("按 8.6ms/8.6GFLOP=1 TFLOPS 估，FC 理论耗时: %.1f ms" % (fc_flops / 1e9))

print()
print("=== 非 FC 算子最多的是什么 ===")
nonfc = [(c, n) for c, n in cnt.items() if c != 9]
for code, n in sorted(nonfc, key=lambda x: -x[1])[:8]:
    print("  %-14s %3d" % (NAMES.get(code, str(code)), n))
