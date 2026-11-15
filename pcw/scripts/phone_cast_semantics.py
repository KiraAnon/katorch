"""验证候选 opcode 是否真的做了 INT16 <-> FP16 转换（看数值）。"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/sem"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(7)
x = (rng.standard_normal(1) * 0.05).astype(np.float32)          # 单元素，量级 0.05
SX = 3.0e-06
q = np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16)
q.tofile(T + "/x.i16")
x.astype(np.float16).tofile(T + "/x.bin")
print("输入真值 %.6f   INT16 码 %d   尺度 %.3g" % (x[0], q[0], SX))


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
        p.kill(); return None, "build"
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile"
    out = "%s/o_%s.bin" % (T, tag)
    r = s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (T + "/" + ing, out))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=30)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run"
    return np.fromfile(out, dtype=dt).astype(np.float32) * scale, None


print("\n=== INT16 -> FP16（期望得到 %.6f 本身）===" % x[0])
for code in (6, 99, 72):
    g = ["OPD 0 7 2 1 1 %.9g 0 i" % SX, "OPD 1 8 2 1 1 0 0 o",
         "IO 1 0 1 1", "OP %d 1 0 1 1" % code, "END"]
    r, e = run(g, "d%d" % code, "x.i16", np.float16)
    print("  opcode %-3d -> %s" % (code, ("%.6f" % r[0]) if r is not None else e))

print("\n=== FP16 -> INT16（期望得到整数码 %d，按尺度 %.3g 还原）===" % (q[0], SX))
for code in (72, 99, 38, 67):
    g = ["OPD 0 8 2 1 1 0 0 i", "OPD 1 7 2 1 1 %.9g 0 o" % SX,
         "IO 1 0 1 1", "OP %d 1 0 1 1" % code, "END"]
    r, e = run(g, "q%d" % code, "x.bin", np.int16, scale=SX)
    print("  opcode %-3d -> %s" % (code, ("%.6f" % r[0]) if r is not None else e))
