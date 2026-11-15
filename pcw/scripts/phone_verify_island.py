"""验证：带岛的 fp16 图 vs 原 fp16 图，输出应该几乎完全一致。"""
import os, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
OUT = D + "/gen5"
T = "/data/local/tmp/vib"
os.makedirs(T, exist_ok=True)

INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
paths = [os.path.join(OUT, f) for f in INS]


def run(gpath, outp):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gpath)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:50]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:50]
    spec = "%d %s 1 %s" % (len(paths), " ".join("%d %s" % (i, f) for i, f in enumerate(paths)), outp)
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:50]
    return np.fromfile(outp, dtype=np.float16).astype(np.float32), None


a, e1 = run(D + "/e2eper512/b00/g.txt", T + "/ref.bin")
b, e2 = run(D + "/e2eper512i/b00/g.txt", T + "/isl.bin")
if a is None:
    print("原图失败:", e1)
elif b is None:
    print("岛图失败:", e2)
else:
    n = min(len(a), len(b))
    d = np.abs(a[:n] - b[:n])
    rel = np.sqrt((d ** 2).mean()) / (np.sqrt((a[:n] ** 2).mean()) + 1e-12)
    print("原 fp16 : max=%.4f std=%.4f" % (np.abs(a).max(), a.std()))
    print("岛 fp16 : max=%.4f std=%.4f" % (np.abs(b).max(), b.std()))
    print("两者 NRMSE %.4f%%   最大差 %.5f" % (rel * 100, d.max()))
