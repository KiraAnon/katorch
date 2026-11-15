"""对照：同样的归约算子在 FP16 下是否正确。"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/redq"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(31)
x = (rng.standard_normal((1, 256)) * 0.05).astype(np.float32)
x.astype(np.float16).tofile(T + "/x.bin")


def run(g, tag):
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
    r = s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (T + "/x.bin", out))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:40]
    return np.fromfile(out, dtype=np.float16).astype(np.float32).flatten(), None


for name, code, want in (("MEAN", 31, np.mean(x, axis=1)),
                         ("REDUCE_MAX", 77, np.max(x, axis=1)),
                         ("REDUCE_SUM", 80, np.sum(x, axis=1))):
    g = ["OPD 0 8 2 1 256 0 0 i",
         "OPD 1 4 1 1 0 0 a -1",
         "OPD 2 4 1 1 0 0 a 0",
         "OPD 3 8 2 1 1 0 0 o",
         "IO 1 0 1 3", "OP %d 3 0 1 2 1 3" % code, "END"]
    r, e = run(g, "f%d" % code)
    if r is None:
        print("%-12s %s" % (name, e)); continue
    w = float(want.flatten()[0])
    print("%-12s fp16 设备 %+.6f  期望 %+.6f  %s"
          % (name, float(r[0]), w, "OK" if abs(r[0] - w) < abs(w) * 0.02 + 1e-6 else "错"))
