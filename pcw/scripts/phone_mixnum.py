"""验证混合尺度下 ADD/MUL/SUB 的数值正确性（是否真的做了 re-quantize）。"""
import numpy as np, subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/mixnum"
os.makedirs(T, exist_ok=True)

rng = np.random.default_rng(11)
x = (rng.standard_normal((1, 64)) * 0.05).astype(np.float32)
y = (rng.standard_normal((1, 64)) * 0.30).astype(np.float32)   # 和 x 量级差 6 倍


def qz(a, s):
    return np.clip(np.rint(a / s), -32767, 32767).astype(np.int16)


def build(op, s1, s2, so, extra_const=None):
    """extra_const: (path, scale) —— 给 MUL/ADD 当第二个输入"""
    g = ["OPD 0 7 2 1 64 %.9g 0 i" % s1]
    if extra_const:
        g.append("OPD 1 7 1 64 %.9g 0 w %s" % (extra_const[1], extra_const[0]))
    else:
        g.append("OPD 1 7 2 1 64 %.9g 0 i" % s2)
    g += ["OPD 2 1 1 1 0 0 a 0",
          "OPD 3 7 2 1 64 %.9g 0 o" % so,
          "IO 1 0 1 3",
          "OP %d 3 0 1 2 1 3" % op,
          "END"]
    p = "%s/g%d.txt" % (T, op)
    open(p, "w").write("\n".join(g) + "\n")
    return p


def run(p, ing, outs, dt):
    pr = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
    pr.stdout.readline()
    def s(c):
        pr.stdin.write(c + "\n"); pr.stdin.flush()
        return pr.stdout.readline().strip()
    r = s('{"cmd":"build","path":"%s"}' % p)
    if '"ok":true' not in r: pr.kill(); return None, "build " + r[:40]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r: pr.kill(); return None, "compile " + r[:40]
    r = s('{"cmd":"run","spec":"1 0 %s 1 0 %s"}' % (ing, outs))
    pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
    try: pr.wait(timeout=60)
    except: pr.kill()
    if '"ok":true' not in r: return None, "run " + r[:40]
    return np.fromfile(outs, dtype=dt).astype(np.float32), None


S1, S2 = 0.05 / 32767.0, 0.35 / 32767.0        # 各按自己的范围取尺度
cases = [("ADD", 0, x + y), ("SUB", 36, x - y), ("MUL", 18, x * y)]

print("%-5s %-10s %-10s %s" % ("算子", "最大误差", "相对误差", "结果"))
for name, code, want in cases:
    so = float(np.abs(want).max() / 32767.0)
    p = build(code, S1, S2, so)
    x.astype(np.float16).tofile(T + "/x.bin")
    y.astype(np.float16).tofile(T + "/y.bin")
    g, err = run(p, T + "/x.bin", T + "/o.bin", np.int16)
    if g is None:
        print("%-5s %s" % (name, err)); continue
    got = g * so
    rel = np.abs(got - want).max() / (np.abs(want).max() + 1e-12)
    print("%-5s %-10.6f %-10.4f%% %s" % (name, np.abs(got - want).max(), rel * 100,
                                         "OK" if rel < 0.02 else "错"))

# 常量混尺度
print("\n=== 常量混尺度（act@S1 与 const@S3）===")
S3 = 0.30 / 32767.0
for name, code, cval, fn in [("ADD", 0, 0.2, lambda a, c: a + c),
                             ("SUB", 36, 0.2, lambda a, c: a - c),
                             ("MUL", 18, 1.5, lambda a, c: a * c)]:
    const = np.full(64, cval, np.float32)
    want = fn(x, const)
    so = float(np.abs(want).max() / 32767.0)
    const.astype(np.float16).tofile(T + "/c.bin")
    p = build(code, S1, S2, so, extra_const=(T + "/c.bin", S3))
    g, err = run(p, T + "/x.bin", T + "/oc.bin", np.int16)
    if g is None:
        print("%-5s %s" % (name, err)); continue
    got = g * so
    rel = np.abs(got - want).max() / (np.abs(want).max() + 1e-12)
    print("%-5s %-10.6f %-10.4f%% %s" % (name, np.abs(got - want).max(), rel * 100,
                                         "OK" if rel < 0.02 else "错"))
