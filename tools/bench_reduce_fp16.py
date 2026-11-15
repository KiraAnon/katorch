"""重测 MEAN（正确写法：3 输入）。顺带测 REDUCE_SUM/MAX 在 fp16 下。"""
import os, subprocess
import numpy as np

HOST = "/data/data/com.termux/files/home/npt/host_new"
T = "/data/local/tmp/mean_test"
os.makedirs(T, exist_ok=True)

ROWS, COLS = 512, 256
x = np.random.randn(ROWS, COLS).astype(np.float32)


def run(name, graph, inp, n_out):
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


def rep(name, got, want):
    g = got.reshape(-1); w = want.reshape(-1)
    n = min(len(g), len(w)); g, w = g[:n], w[:n]
    rel = np.abs(g - w) / (np.abs(w) + 1e-8)
    rel = rel[np.isfinite(rel)]
    print("  %-10s 中位 %.4f%%  最大 %.4f%%" %
          (name, float(np.median(rel)) * 100, float(np.max(rel)) * 100))


print("输入 [%d, %d]" % (ROWS, COLS))
print()

# MEAN 沿最后一维：3 输入
g = """OPD 0 8 2 %d %d 0 0 i
OPD 1 4 1 1 0 0 a -1
OPD 2 4 1 1 0 0 a 0
OPD 3 8 2 %d 1 0 0 o
OP 31 3 0 1 2 1 3
IO 1 0 1 3
END
""" % (ROWS, COLS, ROWS)
got, err = run("mean", g, x, ROWS)
print("MEAN(axis=-1)")
if err:
    print("  失败:", err)
else:
    rep("fp16", got, x.mean(axis=-1, keepdims=True))

# REDUCE_SUM 沿最后一维
g = """OPD 0 8 2 %d %d 0 0 i
OPD 1 4 1 1 0 0 a -1
OPD 2 4 1 1 0 0 a 0
OPD 3 8 2 %d 1 0 0 o
OP 80 3 0 1 2 1 3
IO 1 0 1 3
END
""" % (ROWS, COLS, ROWS)
got, err = run("rsum", g, x, ROWS)
print("REDUCE_SUM(axis=-1)")
if err:
    print("  失败:", err)
else:
    rep("fp16", got, x.sum(axis=-1, keepdims=True))

# REDUCE_MAX 沿最后一维
g = """OPD 0 8 2 %d %d 0 0 i
OPD 1 4 1 1 0 0 a -1
OPD 2 4 1 1 0 0 a 0
OPD 3 8 2 %d 1 0 0 o
OP 77 3 0 1 2 1 3
IO 1 0 1 3
END
""" % (ROWS, COLS, ROWS)
got, err = run("rmax", g, x, ROWS)
print("REDUCE_MAX(axis=-1)")
if err:
    print("  失败:", err)
else:
    rep("fp16", got, x.max(axis=-1, keepdims=True))
