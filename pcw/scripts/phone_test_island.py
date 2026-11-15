"""验证完整的混合精度链路：INT16 -> DEQUANTIZE(6) -> FP16 运算 -> QUANTIZE(72) -> INT16"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/island"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(31)
x = (rng.standard_normal((1, 64)) * 0.05).astype(np.float32)
want = x * 2.0
SX = float(np.abs(x).max() / 32767.0)
SO = float(np.abs(want).max() / 32767.0)
q = np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16)
q.tofile(T + "/x.i16")
np.array([2.0], np.float16).tofile(T + "/two.bin")


def run(g, tag, ing, dt, scale=1.0):
    fp = "%s/g_%s.txt" % (T, tag)
    open(fp, "w").write("\n".join(g) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % fp)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:40]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:40]
    out = "%s/o_%s.bin" % (T, tag)
    r = s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (T + "/" + ing, out))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:40]
    return np.fromfile(out, dtype=dt).astype(np.float32) * scale, None


# 完整链路：INT16 --6--> FP16 --MUL--> FP16 --72--> INT16
g = ["OPD 0 7 2 1 64 %.9g 0 i" % SX,          # 0 INT16 输入
     "OPD 1 8 2 1 64 0 0 o",                  # 1 FP16 岛内
     "OPD 2 8 1 1 0 0 w %s/two.bin" % T,      # 2 FP16 常量 2.0
     "OPD 3 4 1 1 0 0 a 0",                   # 3 act
     "OPD 4 8 2 1 64 0 0 o",                  # 4 FP16 结果
     "OPD 5 7 2 1 64 %.9g 0 o" % SO,          # 5 INT16 输出
     "IO 1 0 1 5",
     "OP 6 1 0 1 1",                          # DEQUANTIZE
     "OP 18 3 1 2 3 1 4",                     # MUL（FP16 里）
     "OP 72 1 4 1 5",                         # QUANTIZE
     "END"]
r, e = run(g, "island", "x.i16", np.int16, scale=SO)
if r is None:
    print("完整链路: %s" % e)
else:
    ref = want.flatten()
    got = r
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
    print("INT16→DQ→MUL→Q→INT16  误差 %.4f%%  %s" % (rel * 100, "OK" if rel < 0.02 else "错"))

# 只到 FP16 输出（确认 DQ 后的值是对的）
g2 = ["OPD 0 7 2 1 64 %.9g 0 i" % SX,
      "OPD 1 8 2 1 64 0 0 o",
      "IO 1 0 1 1", "OP 6 1 0 1 1", "END"]
r2, e2 = run(g2, "dq", "x.i16", np.float16)
if r2 is not None:
    d = np.abs(r2 - x.flatten()).max()
    print("DEQUANTIZE 单独: 最大差 %.6f (量程 %.4f)" % (d, np.abs(x).max()))
