"""试 DEQUANTIZE / QUANTIZE 系列算子能否做 INT16 <-> FP16 转换。"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/castmix"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(101)
x = (rng.standard_normal((1, 64)) * 0.05).astype(np.float32)
SX = float(np.abs(x).max() / 32767.0)
np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16).tofile(T + "/x.i16")
x.astype(np.float16).tofile(T + "/x.bin")


def try_g(g, tag):
    open(T + "/g_%s.txt" % tag, "w").write("\n".join(g) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % (T + "/g_%s.txt" % tag))
    if '"ok":true' not in r:
        p.kill(); return "build " + r[:40]
    r = s('{"cmd":"compile"}')
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    if '"ok":true' not in r:
        return "compile " + r[:40]
    return "编译 OK"


# 候选 opcode：6=DEQUANTIZE(旧)，60/61/62=QUANTIZE/DEQUANTIZE(新)
for code in (6, 60, 61, 62, 63):
    # INT16 输入 -> 该算子 -> FP16 输出
    A = ["OPD 0 7 2 1 64 %.9g 0 i" % SX,
         "OPD 1 8 2 1 64 0 0 o",
         "IO 1 0 1 1", "OP %d 1 0 1 1" % code, "END"]
    r1 = try_g(A, "dq%d" % code)
    # FP16 输入 -> 该算子 -> INT16 输出
    B = ["OPD 0 8 2 1 64 0 0 i",
         "OPD 1 7 2 1 64 %.9g 0 o" % SX,
         "IO 1 0 1 1", "OP %d 1 0 1 1" % code, "END"]
    r2 = try_g(B, "q%d" % code)
    print("opcode %-3d  INT16→FP16: %-22s  FP16→INT16: %s" % (code, r1, r2))
