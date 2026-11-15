"""查 MEAN 最大 35% 误差的来源：是相对误差放大，还是真的算错了。"""
import os, subprocess
import numpy as np

HOST = "/data/data/com.termux/files/home/npt/host_new"
T = "/data/local/tmp/mean_diag"
os.makedirs(T, exist_ok=True)

ROWS, COLS = 512, 256
rng = np.random.default_rng(42)
x = rng.standard_normal((ROWS, COLS)).astype(np.float32)

g = """OPD 0 8 2 %d %d 0 0 i
OPD 1 4 1 1 0 0 a -1
OPD 2 4 1 1 0 0 a 0
OPD 3 8 2 %d 1 0 0 o
OP 31 3 0 1 2 1 3
IO 1 0 1 3
END
""" % (ROWS, COLS, ROWS)

gp = T + "/mean.txt"
open(gp, "w").write(g)

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdout.readline()

def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

print("build  :", s('{"cmd":"build","path":"%s"}' % gp))
print("compile:", s('{"cmd":"compile"}'))

ip = T + "/x.bin"
x.astype(np.float16).tofile(ip)
op = T + "/y.bin"
print("run    :", s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (ip, op)))
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=20)
except Exception: p.kill()

got = np.fromfile(op, dtype=np.float16).astype(np.float32).reshape(-1)
want = x.mean(axis=-1, keepdims=True).reshape(-1)

n = min(len(got), len(want))
got, want = got[:n], want[:n]

abs_err = np.abs(got - want)
rel_err = abs_err / (np.abs(want) + 1e-8)

print()
print("=== 整体 ===")
print("  最大绝对误差: %.6f" % float(abs_err.max()))
print("  中位绝对误差: %.6f" % float(np.median(abs_err)))
print("  最大相对误差: %.2f%%" % (float(rel_err.max()) * 100))
print()

# 相对误差最大的 3 行，看它们的真值
idx = np.argsort(-rel_err)[:3]
print("=== 相对误差最大的 3 行 ===")
for i in idx:
    print("  行 %3d: 真值 %+10.6f  得到 %+10.6f  绝对差 %.6f  相对 %.2f%%"
          % (i, want[i], got[i], abs_err[i], rel_err[i] * 100))

print()
# 和 fp16 累积和的理论误差比
x16 = x.astype(np.float16)
sum16 = x16.astype(np.float32).sum(axis=-1)
sum32 = x.astype(np.float32).sum(axis=-1)
print("=== 对照：纯 numpy 的 fp16 累加误差 ===")
print("  用 fp16 输入做 float32 累加 vs float32 输入: 最大 %.6f"
      % float(np.abs(sum16 - sum32).max()))
print("  注意：NPU 是 fp16 精度累加，误差会更大属正常")
print()
print("=== 结论 ===")
if abs_err.max() < 0.01:
    print("  绝对误差很小（<0.01），35% 相对误差是「真值接近 0」导致的放大，不是算错")
else:
    print("  绝对误差偏大，需要进一步查")
