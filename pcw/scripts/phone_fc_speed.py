"""决定性实验：同一个大 FC，fp16 权重 vs INT8 权重（W8A16），谁快？

如果 INT8 没有明显加速，整条量化路线就没有价值（多出来的 DQ/Q 算子只是纯开销）。
"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/fcspeed"
os.makedirs(T, exist_ok=True)

M, K, N = 1024, 2048, 2048          # x[M,K] @ W[N,K]^T
rng = np.random.default_rng(3)
x = (rng.standard_normal((M, K)) * 0.01).astype(np.float32)
W = (rng.standard_normal((N, K)) * 0.01).astype(np.float32)
b = np.zeros(N, np.float32)

x.astype(np.float16).tofile(T + "/x.bin")
W.astype(np.float16).tofile(T + "/w.bin")
b.astype(np.float16).tofile(T + "/b.bin")

SX = float(np.abs(x).max() / 32767.0)
mx = np.abs(W).max(axis=1)
sw = np.where(mx > 0, mx / 127.0, 1.0).astype(np.float32)
qw = np.clip(np.rint(W / sw[:, None]), -127, 127).astype(np.int8)
qw.tofile(T + "/w8.bin")
sw.tofile(T + "/w8.sc")
qxb = np.clip(np.rint(b / (SX * sw)), -2**31, 2**31 - 1).astype(np.int32)
qxb.tofile(T + "/b32.bin")
qx = np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16)
qx.tofile(T + "/x16.bin")

ref = x @ W.T
SO = float(np.abs(ref).max() / 32767.0)

GF16 = ["OPD 0 8 2 %d %d 0 0 i" % (M, K),
        "OPD 1 8 2 %d %d 0 0 w %s/w.bin" % (N, K, T),
        "OPD 2 8 1 %d 0 0 w %s/b.bin" % (N, T),
        "OPD 3 4 1 1 0 0 a 0",
        "OPD 4 8 2 %d %d 0 0 o" % (M, N),
        "IO 1 0 1 4", "OP 9 4 0 1 2 3 1 4", "END"]
GQ8 = ["OPD 0 7 2 %d %d %.9g 0 i" % (M, K, SX),
       "OPD 1 11 2 %d %d 0 0 w %s/w8.bin" % (N, K, T),
       "QCH 1 0 %d %s/w8.sc" % (N, T),
       "OPD 2 4 1 %d 0 0 w %s/b32.bin" % (N, T),
       "OPD 3 1 1 1 0 0 a 0",
       "OPD 4 7 2 %d %d %.9g 0 o" % (M, N, SO),
       "IO 1 0 1 4", "OP 9 4 0 1 2 3 1 4", "END"]
open(T + "/gf.txt", "w").write("\n".join(GF16) + "\n")
open(T + "/gq.txt", "w").write("\n".join(GQ8) + "\n")


def bench(gpath, ing, tag, dt, reps=8):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gpath)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:40]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:40]
    out = "%s/o_%s.bin" % (T, tag)
    spec = "1 0 %s 1 %s" % (ing, out)
    s('{"cmd":"run","spec":"%s"}' % spec)     # 预热
    ts = []
    for _ in range(reps):
        t0 = time.time()
        r = s('{"cmd":"run","spec":"%s"}' % spec)
        ts.append((time.time() - t0) * 1000)
        if '"ok":true' not in r:
            p.kill(); return None, "run"
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    ts.sort()
    med = ts[len(ts) // 2]
    got = np.fromfile(out, dtype=dt).astype(np.float32)
    return (med, got), None


print("FC %dx%d @ %dx%d" % (M, K, N, N))
r1, e1 = bench(T + "/gf.txt", T + "/x.bin", "f16", np.float16)
r2, e2 = bench(T + "/gq.txt", T + "/x16.bin", "q8", np.int16)
if r1 is None:
    print("fp16 失败:", e1)
if r2 is None:
    print("W8A16 失败:", e2)
if r1 and r2:
    t16, o16 = r1
    t8, o8 = r2
    o8 = o8 * SO
    n = min(len(o16), len(o8))
    rel = np.abs(o16[:n] - o8[:n]).max() / (np.abs(o16[:n]).max() + 1e-12)
    print("fp16   : %.1f ms" % t16)
    print("W8A16  : %.1f ms" % t8)
    print("加速比 : %.2fx" % (t16 / t8 if t8 else 0))
    print("数值误差: %.3f%%" % (rel * 100))
