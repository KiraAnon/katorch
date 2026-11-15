"""测 MEAN / REDUCE_SUM / RSQRT 在 INT16 下是否算对。"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/redq"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(31)
x = (rng.standard_normal((1, 256)) * 0.05).astype(np.float32)
x.astype(np.float16).tofile(T + "/x.bin")
S = float(np.abs(x).max() / 32767.0)


def run(g, tag, dt):
    open(T + "/g_%s.txt" % tag, "w").write("\n".join(g) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % (T + "/g_%s.txt" % tag))
    if '"ok":true' not in r:
        p.kill(); return None, "build"
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile"
    out = T + "/o_%s.bin" % tag
    r = s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (T + "/x.bin", out))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run"
    return np.fromfile(out, dtype=dt).astype(np.float32).flatten(), None


q = np.clip(np.rint(x / S), -32767, 32767).astype(np.int16)
q.tofile(T + "/qx.i16")

cases = [
    ("MEAN(31)", 31, np.mean(x, axis=1)),
    ("REDUCE_MAX(77)", 77, np.max(x, axis=1)),
    ("REDUCE_SUM(80)", 80, np.sum(x, axis=1)),
]
for name, code, want in cases:
    SO = float(np.abs(want).max() / 32767.0)
    g = ["OPD 0 7 2 1 256 %.9g 0 i" % S,
         "OPD 1 4 1 1 0 0 a -1",
         "OPD 2 4 1 1 0 0 a 0",
         "OPD 3 7 2 1 1 %.9g 0 o" % SO,
         "IO 1 0 1 3", "OP %d 3 0 1 2 1 3" % code, "END"]
    r, e = run(g, "c%d" % code, np.int16)
    if r is None:
        print("%-16s %s" % (name, e)); continue
    got = r * SO
    w = float(want.flatten()[0]); gv = float(got[0])
    print("%-16s 设备 %+.6f  期望 %+.6f  误差 %.3f%%" % (name, gv, w, abs(gv - w) / max(abs(w), 1e-12) * 100))

# RSQRT：y = 1/sqrt(x2)，x2 为正值
x2 = (np.abs(rng.standard_normal((1, 256))) * 0.05 + 0.001).astype(np.float32)
x2.astype(np.float16).tofile(T + "/x2.bin")
S2 = float(np.abs(x2).max() / 32767.0)
q2 = np.clip(np.rint(x2 / S2), -32767, 32767).astype(np.int16)
q2.tofile(T + "/qx2.i16")
want_r = 1.0 / np.sqrt(x2)
SR = float(np.abs(want_r).max() / 32767.0)
g = ["OPD 0 7 2 1 256 %.9g 0 i" % S2,
     "OPD 1 7 2 1 256 %.9g 0 o" % SR,
     "IO 1 0 1 1", "OP 83 1 0 1 1", "END"]
open(T + "/g_rs.txt", "w").write("\n".join(g) + "\n")
p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdout.readline()
def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()
r = s('{"cmd":"build","path":"%s"}' % (T + "/g_rs.txt"))
ok = '"ok":true' in r
if ok:
    r = s('{"cmd":"compile"}')
    ok = '"ok":true' in r
if ok:
    r = s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (T + "/qx2.i16", T + "/o_rs.bin"))
    ok = '"ok":true' in r
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()
if ok:
    got = np.fromfile(T + "/o_rs.bin", dtype=np.int16).astype(np.float32) * SR
    ref = want_r.flatten()
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    print("%-16s 误差 %.4f%%  %s" % ("RSQRT(83)", rel * 100, "OK" if rel < 0.02 else "错"))
else:
    print("%-16s 失败 %s" % ("RSQRT(83)", r[:50]))
