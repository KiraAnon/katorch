"""测混合精度：INT16 <-> FP16 用 CAST(45) 连接，能否编译并算对。

若能，就可以做「FP16 norm 岛」——把归一化那段留在 FP16，绕开 INT16 归约/rsqrt 的精度问题。
"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/castmix"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(101)
x = (rng.standard_normal((1, 64)) * 0.05).astype(np.float32)
want = x * 2.0
x.astype(np.float16).tofile(T + "/x.bin")
SX = float(np.abs(x).max() / 32767.0)
q = np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16)
q.tofile(T + "/x.i16")
TWO = np.array([2.0], np.float16)
TWO.tofile(T + "/two.bin")


def run(g, tag, ins, dt, out_scale=1.0, n_out=1):
    open(T + "/g_%s.txt" % tag, "w").write("\n".join(g) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % (T + "/g_%s.txt" % tag))
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:45]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:45]
    out = T + "/o_%s.bin" % tag
    spec = "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), out)
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:45]
    return np.fromfile(out, dtype=dt).astype(np.float32).flatten() * out_scale, None


# --- A: INT16 输入 -> CAST 到 FP16 -> MUL -> CAST 回 INT16 ---
SO = float(np.abs(want).max() / 32767.0)
g = ["OPD 0 7 2 1 64 %.9g 0 i" % SX,
     "OPD 1 8 2 1 64 %.9g 0 o" % SX,          # CAST 输出：FP16，同尺度语义
     "OPD 2 8 1 1 0 0 w %s/two.bin" % T,      # FP16 常量
     "OPD 3 8 2 1 64 0 0 o",                  # FP16 中间
     "OPD 4 7 2 1 64 %.9g 0 o" % SO,          # CAST 回 INT16
     "IO 1 0 1 4",
     "OP 45 1 0 1 1",                          # CAST INT16 -> FP16
     "OP 18 3 1 2 9 1 3",                      # 需要 act，先借个位置
     "OP 45 1 3 1 4",                          # CAST FP16 -> INT16
     "END"]
# 上面 act 用的 9 号不存在，改成显式定义
g = ["OPD 0 7 2 1 64 %.9g 0 i" % SX,
     "OPD 1 8 2 1 64 %.9g 0 o" % SX,
     "OPD 2 8 1 1 0 0 w %s/two.bin" % T,
     "OPD 3 4 1 1 0 0 a 0",
     "OPD 4 8 2 1 64 0 0 o",
     "OPD 5 7 2 1 64 %.9g 0 o" % SO,
     "IO 1 0 1 5",
     "OP 45 1 0 1 1",
     "OP 18 3 1 2 3 1 4",
     "OP 45 1 4 1 5",
     "END"]
r, e = run(g, "A", [T + "/x.i16"], np.int16, out_scale=SO)
if r is None:
    print("A  INT16->FP16->MUL->INT16 : %s" % e)
else:
    ref = want.flatten()
    rel = np.abs(r - ref).max() / (np.abs(ref).max() + 1e-12)
    print("A  INT16->FP16->MUL->INT16 : 误差 %.4f%%  %s" % (rel * 100, "OK" if rel < 0.02 else "错"))
