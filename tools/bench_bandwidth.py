"""测 MDLA 的带宽上限：用超大张量的简单操作，看能到多少 GB/s。"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
T = "/data/local/tmp/bwtest"
os.makedirs(T, exist_ok=True)


def bench(S, K, n_mul=1, reps=5):
    tag = "%dx%d_%d" % (S, K, n_mul)
    a = "%s/a_%s.bin" % (T, tag)
    b = "%s/b_%s.bin" % (T, tag)
    nb = S * K * 2
    if not os.path.exists(a) or os.path.getsize(a) != nb:
        rng = np.random.default_rng(0)
        (rng.standard_normal((S, K)) * 0.01).astype(np.float16).tofile(a)
        (rng.standard_normal((S, K)) * 0.01).astype(np.float16).tofile(b)

    L = ["OPD 0 8 2 %d %d 0 0 i" % (S, K),
         "OPD 1 8 2 %d %d 0 0 w %s" % (S, K, b),
         "OPD 2 4 1 1 0 0 a 0"]
    prev, oid = 0, 3
    for _ in range(n_mul):
        L.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
        L.append("OP 18 3 %d 1 2 1 %d" % (prev, oid))
        prev = oid
        oid += 1
    L.append("IO 1 0 1 %d" % prev)
    L.append("END")
    gp = "%s/g_%s.txt" % (T, tag)
    open(gp, "w").write("\n".join(L) + "\n")

    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill(); return None
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None
    out = "%s/o_%s.bin" % (T, tag)
    spec = "1 0 %s 1 %s" % (a, out)
    s('{"cmd":"run","spec":"%s"}' % spec)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        r = s('{"cmd":"run","spec":"%s"}' % spec)
        ts.append((time.time() - t0) * 1000)
        if '"ok":true' not in r:
            p.kill(); return None
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except Exception: p.kill()
    ts.sort()
    ms = ts[len(ts) // 2]
    # 每个 MUL 读 2 张写 1 张
    total = n_mul * nb * 3
    return ms, total / (ms / 1000) / 1e9, nb


print("%-18s %-8s %-10s %-10s %s" % ("张量", "MB/张", "耗时ms", "带宽GB/s", "备注"))
cases = [
    (1024, 2048, 1), (1024, 2048, 8), (1024, 2048, 32),
    (2048, 4096, 1), (4096, 4096, 1),
]
for S, K, n in cases:
    r = bench(S, K, n)
    if r is None:
        print("%-18s 失败（可能超 IOVA）" % ("%dx%d" % (S, K)))
        continue
    ms, gbs, nb = r
    print("%-18s %-8.1f %-10.1f %-10.1f %s" %
          ("%dx%d ×%d" % (S, K, n), nb / 1e6, ms, gbs,
           "多了是启动开销" if n > 1 else ""))
