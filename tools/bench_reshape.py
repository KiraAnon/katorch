"""RESHAPE 链的真实成本（修正 shape 声明）。"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
T = "/data/local/tmp/reshapecost"
os.makedirs(T, exist_ok=True)

S, K = 1024, 2048
NB = S * K * 2
(np.random.default_rng(0).standard_normal((S, K)) * 0.01).astype(np.float16).tofile(T + "/a.bin")

np.array([S, K], np.int32).tofile(T + "/s2.bin")        # 2 个元素
np.array([1, S, K], np.int32).tofile(T + "/s3.bin")     # 3 个元素


def graph(n):
    """n 个 RESHAPE，交替 [S,K] <-> [1,S,K]。"""
    L = ["OPD 0 8 2 %d %d 0 0 i" % (S, K)]
    prev, oid = 0, 1
    for i in range(n):
        to3 = (i % 2 == 0)
        if to3:
            L.append("OPD %d 8 3 %d %d %d 0 0 o" % (oid, 1, S, K))
            L.append("OPD %d 4 1 3 0 0 w %s/s3.bin" % (oid + 1, T))
        else:
            L.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
            L.append("OPD %d 4 1 2 0 0 w %s/s2.bin" % (oid + 1, T))
        L.append("OP 22 2 %d %d 1 %d" % (prev, oid + 1, oid))
        prev = oid
        oid += 2
    # 最后保证输出是 [S,K]
    if n % 2 == 1:
        L.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
        L.append("OPD %d 4 1 2 0 0 w %s/s2.bin" % (oid + 1, T))
        L.append("OP 22 2 %d %d 1 %d" % (prev, oid + 1, oid))
        prev = oid
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


print("RESHAPE 链：每次 [%d,%d] <-> [1,%d,%d]，%.1f MB" % (S, K, S, K, NB / 1e6))
print()
print("%-8s %-12s %s" % ("个数", "耗时ms", "每算子ms"))
for n in (1, 4, 16):
    gp = "%s/gr%d.txt" % (T, n)
    open(gp, "w").write(graph(n))
    t, err = bench(gp, "r%d" % n)
    if t is None:
        print("%-8d 失败 %s" % (n, err))
    else:
        print("%-8d %-12.1f %.2f" % (n, t, t / n))
