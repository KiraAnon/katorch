"""测 BATCH_MATMUL 的真实成本。注意力是 16 head × 1024×1024。"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
T = "/data/local/tmp/bmcost"
os.makedirs(T, exist_ok=True)

# 注意力：Q [1,1024,128] @ K^T [1,128,1024] -> [1,1024,1024]
# 但真实图里是 [16,1024,1024] 的形状（16 head 合并）
# 先测一个小一点但结构相同的：a=[1,1024,128], b=[1,128,1024]
M, KD, N = 1024, 128, 1024

rng = np.random.default_rng(0)
(rng.standard_normal((1, M, KD)) * 0.1).astype(np.float16).tofile(T + "/a.bin")
(rng.standard_normal((1, KD, N)) * 0.1).astype(np.float16).tofile(T + "/b.bin")


def graph(n):
    L = [
        "OPD 0 8 3 %d %d %d 0 0 i" % (1, M, KD),
        "OPD 1 8 3 %d %d %d 0 0 w %s/b.bin" % (1, KD, N, T),
        "OPD 2 4 1 1 0 0 a 0",
        "OPD 3 4 1 1 0 0 a 0",
    ]
    prev = 0
    oid = 4
    for i in range(n):
        L.append("OPD %d 8 3 %d %d %d 0 0 o" % (oid, 1, M, N))
        # BATCH_MATMUL(a, b, adj_x, adj_y) 4 输入
        L.append("OP 102 4 %d 1 2 3 1 %d" % (prev, oid))
        prev = oid
        oid += 1
    L.append("IO 1 0 1 %d" % prev)
    L.append("END")
    return "\n".join(L) + "\n"


def bench(gp, tag, reps=5):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:70]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:70]
    out = "%s/o_%s.bin" % (T, tag)
    spec = "1 0 %s/a.bin 1 %s" % (T, out)
    s('{"cmd":"run","spec":"%s"}' % spec)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        r = s('{"cmd":"run","spec":"%s"}' % spec)
        ts.append((time.time() - t0) * 1000)
        if '"ok":true' not in r:
            p.kill(); return None, "run"
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=30)
    except Exception: p.kill()
    ts.sort()
    return ts[len(ts) // 2], None


flops = 2 * M * KD * N / 1e9
print("BATCH_MATMUL: [1,%d,%d] @ [1,%d,%d] -> [1,%d,%d]" % (M, KD, KD, N, M, N))
print("单次 %.2f GFLOP，输出 %.1f MB" % (flops, M * N * 2 / 1e6))
print()
print("%-8s %-12s %-12s %s" % ("个数", "耗时ms", "每算子ms", "TFLOPS"))
for n in (1, 4):
    gp = "%s/g%d.txt" % (T, n)
    open(gp, "w").write(graph(n))
    t, err = bench(gp, "b%d" % n)
    if t is None:
        print("%-8d 失败 %s" % (n, err))
    else:
        print("%-8d %-12.1f %-12.2f %s" %
              (n, t, t / n, ("%.3f" % (n * flops / (t / 1000))) if n else "-"))
