"""验证：用 FC（全 1 权重）代替 REDUCE_SUM / MEAN 是否数值正确（fp16 与 W8A16 两种）。"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/fcreduce"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(77)
N = 64
x = (rng.standard_normal((1, N)) * 0.05).astype(np.float32)
want_sum = x.sum(axis=1, keepdims=True)
want_mean = x.mean(axis=1, keepdims=True)

x.astype(np.float16).tofile(T + "/x.bin")
SX = float(np.abs(x).max() / 32767.0)


def run(g, tag, tax_in, dt, out_scale=1.0):
    open(T + "/g_%s.txt" % tag, "w").write("\n".join(g) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % (T + "/g_%s.txt" % tag))
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:40]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:40]
    out = T + "/o_%s.bin" % tag
    r = s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (T + "/" + tax_in, out))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:40]
    return np.fromfile(out, dtype=dt).astype(np.float32).flatten() * out_scale, None


# ---------- 1) fp16：FC 做求和 ----------
W_ones = np.ones((1, N), np.float32)
W_ones.astype(np.float16).tofile(T + "/w_ones.bin")
np.zeros(1, np.float32).astype(np.float16).tofile(T + "/b_zero.bin")
g = ["OPD 0 8 2 1 %d 0 0 i" % N,
     "OPD 1 8 2 1 %d 0 0 w %s/w_ones.bin" % (N, T),
     "OPD 2 8 1 1 0 0 w %s/b_zero.bin" % T,
     "OPD 3 1 1 1 0 0 a 0",
     "OPD 4 8 2 1 1 0 0 o",
     "IO 1 0 1 4", "OP 9 4 0 1 2 3 1 4", "END"]
r, e = run(g, "f16sum", "x.bin", np.float16)
if r is None:
    print("fp16 FC-sum 失败:", e)
else:
    print("fp16 FC 求和: 设备 %+.6f  期望 %+.6f  误差 %.4f%%"
          % (r[0], want_sum.flatten()[0],
             abs(r[0] - want_sum.flatten()[0]) / abs(want_sum.flatten()[0]) * 100))

# ---------- 2) W8A16：FC 做求和 ----------
q = np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16)
q.tofile(T + "/x16.i16")
SW = np.array([1.0 / 127.0], np.float32)          # 全 1 权重 -> max=1, scale=1/127
qw = np.full((1, N), 127, np.int8)
qw.tofile(T + "/w8.bin"); SW.tofile(T + "/w8.sc")
qb = np.zeros(1, np.int32)                        # bias = 0
qb.tofile(T + "/b32.bin")
want_s = float(want_sum.flatten()[0])
SY = float(abs(want_s) / 32767.0)
gq = ["OPD 0 7 2 1 %d %.9g 0 i" % (N, SX),
      "OPD 1 11 2 1 %d 0 0 w %s/w8.bin" % (N, T),
      "QCH 1 0 1 %s/w8.sc" % T,
      "OPD 2 4 1 1 0 0 w %s/b32.bin" % T,
      "OPD 3 1 1 1 0 0 a 0",
      "OPD 4 7 2 1 1 %.9g 0 o" % SY,
      "IO 1 0 1 4", "OP 9 4 0 1 2 3 1 4", "END"]
r2, e2 = run(gq, "w8sum", "x16.i16", np.int16, out_scale=SY)
if r2 is None:
    print("W8A16 FC-sum 失败:", e2)
else:
    print("W8A16 FC 求和: 设备 %+.6f  期望 %+.6f  误差 %.4f%%"
          % (r2[0], want_s, abs(r2[0] - want_s) / abs(want_s) * 100))
