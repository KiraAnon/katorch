"""CAST 的各种最小形式，逐个试探哪些能编译。"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/castmix"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(101)
x = (rng.standard_normal((1, 64)) * 0.05).astype(np.float32)
x.astype(np.float16).tofile(T + "/x.bin")
SX = float(np.abs(x).max() / 32767.0)
np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16).tofile(T + "/x.i16")


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
        p.kill(); return "build " + r[:45]
    r = s('{"cmd":"compile"}')
    r2 = s('{"cmd":"sup"}')
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    if '"ok":true' not in r:
        return "compile " + r[:45]
    return "编译 OK  支持性: " + r2[:70]


cases = {}

# 1) 纯 fp16 图里放一个 CAST（fp16 输入 -> fp16 输出，同类型）
cases["fp16图+CAST同类型"] = [
    "OPD 0 8 2 1 64 0 0 i",
    "OPD 1 8 2 1 64 0 0 o",
    "IO 1 0 1 1", "OP 45 1 0 1 1", "END"]

# 2) INT16 输入 -> CAST 到 FP16，输出就是模型输出
cases["INT16→FP16(输出)"] = [
    "OPD 0 7 2 1 64 %.9g 0 i" % SX,
    "OPD 1 8 2 1 64 0 0 o",
    "IO 1 0 1 1", "OP 45 1 0 1 1", "END"]

# 3) FP16 输入 -> CAST 到 INT16，输出就是模型输出
cases["FP16→INT16(输出)"] = [
    "OPD 0 8 2 1 64 0 0 i",
    "OPD 1 7 2 1 64 %.9g 0 o" % SX,
    "IO 1 0 1 1", "OP 45 1 0 1 1", "END"]

# 4) INT16 -> CAST -> INT16（同类型，验证 CAST 在 INT16 里能用）
cases["INT16→INT16 CAST"] = [
    "OPD 0 7 2 1 64 %.9g 0 i" % SX,
    "OPD 1 7 2 1 64 %.9g 0 o" % SX,
    "IO 1 0 1 1", "OP 45 1 0 1 1", "END"]

for name, g in cases.items():
    print("%-22s %s" % (name, try_g(g, name.replace("→", "_").replace("+", "_"))))
