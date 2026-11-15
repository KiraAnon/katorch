"""实测 MEAN / SQRT / RSQRT / EXP 在 fp16 下的精度。"""
import json, os, subprocess
import numpy as np

HOST = "/data/data/com.termux/files/home/npt/host_new"
T = "/data/local/tmp/op_prec"
os.makedirs(T, exist_ok=True)

N = 4096          # 一维，好算


def run_graph(name, graph, inp, out_shape):
    gp = "%s/%s.txt" % (T, name)
    open(gp, "w").write(graph)
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:50]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:50]

    ip = "%s/%s_in.bin" % (T, name)
    inp.astype(np.float16).tofile(ip)
    op = "%s/%s_out.bin" % (T, name)
    r = s('{"cmd":"run","spec":"1 0 %s 1 %s"}' % (ip, op))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=20)
    except Exception: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:50]
    return np.fromfile(op, dtype=np.float16).astype(np.float32), None


def report(name, got, want):
    if got is None:
        print("  %-8s 失败" % name)
        return
    g = got.reshape(-1)[:len(want.reshape(-1))]
    w = want.reshape(-1)
    n = min(len(g), len(w))
    g, w = g[:n], w[:n]
    rel = np.abs(g - w) / (np.abs(w) + 1e-8)
    rel = rel[np.isfinite(rel)]
    mx = float(np.percentile(rel, 100)) * 100 if len(rel) else 0
    med = float(np.median(rel)) * 100 if len(rel) else 0
    p99 = float(np.percentile(rel, 99)) * 100 if len(rel) else 0
    print("  %-8s 中位 %.4f%%  P99 %.4f%%  最大 %.4f%%" % (name, med, p99, mx))


# ---------- 1. MEAN ----------
x = np.random.randn(N).astype(np.float32)
g = """OPD 0 8 1 %d 0 0 i
OPD 1 1 0   0 0 a 0
OPD 2 8 1 1 0 0 o
OP 31 2 0 1 1 2
IO 1 0 1 2
END
""" % N
got, err = run_graph("mean", g, x, (1,))
print("MEAN (axis=0, 全数组均值)")
if err:
    print("  失败:", err)
else:
    report("fp16", got, np.array([x.mean()], np.float32))

# ---------- 2. SQRT ----------
x = np.abs(np.random.randn(N)).astype(np.float32) + 0.01
g = """OPD 0 8 1 %d 0 0 i
OPD 1 8 1 %d 0 0 o
OP 88 1 0 1 1
IO 1 0 1 1
END
""" % (N, N)
got, err = run_graph("sqrt", g, x, (N,))
print("SQRT")
if err:
    print("  失败:", err)
else:
    report("fp16", got, np.sqrt(x))

# ---------- 3. RSQRT ----------
x = np.abs(np.random.randn(N)).astype(np.float32) + 0.1
g = """OPD 0 8 1 %d 0 0 i
OPD 1 8 1 %d 0 0 o
OP 83 1 0 1 1
IO 1 0 1 1
END
""" % (N, N)
got, err = run_graph("rsqrt", g, x, (N,))
print("RSQRT")
if err:
    print("  失败:", err)
else:
    report("fp16", got, 1.0 / np.sqrt(x))

# ---------- 4. EXP ----------
x = (np.random.randn(N) * 0.5).astype(np.float32)
g = """OPD 0 8 1 %d 0 0 i
OPD 1 8 1 %d 0 0 o
OP 49 1 0 1 1
IO 1 0 1 1
END
""" % (N, N)
got, err = run_graph("exp", g, x, (N,))
print("EXP")
if err:
    print("  失败:", err)
else:
    report("fp16", got, np.exp(x))
