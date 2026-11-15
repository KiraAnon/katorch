"""测 MDLA 单算子固定开销：连续做 N 个 ADD，看耗时随 N 怎么变。"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/opcost"
os.makedirs(T, exist_ok=True)

S, K = 1024, 2048
rng = np.random.default_rng(0)
(rng.standard_normal((S, K)) * 0.01).astype(np.float16).tofile(T + "/a.bin")
(rng.standard_normal((S, K)) * 0.01).astype(np.float16).tofile(T + "/b.bin")


def make_graph(n_add):
    """OP 格式: OP <opcode> <n_in> <in_ids...> <n_out> <out_ids...>"""
    lines = [
        "OPD 0 8 2 %d %d 0 0 i" % (S, K),
        "OPD 1 8 2 %d %d 0 0 w %s/b.bin" % (S, K, T),
    ]
    prev = 0
    oid = 2
    for i in range(n_add):
        lines.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
        # ADD(prev, 1) -> oid
        lines.append("OP 0 2 %d 1 1 %d" % (prev, oid))
        prev = oid
        oid += 1
    lines.append("IO 1 0 1 %d" % prev)
    lines.append("END")
    return "\n".join(lines) + "\n"


def bench(n_add, reps=5):
    gp = "%s/g%d.txt" % (T, n_add)
    open(gp, "w").write(make_graph(n_add))
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill()
        return None, "build " + r[:80]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill()
        return None, "compile " + r[:80]
    out = "%s/o%d.bin" % (T, n_add)
    spec = "1 0 %s/a.bin 1 %s" % (T, out)
    s('{"cmd":"run","spec":"%s"}' % spec)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        r = s('{"cmd":"run","spec":"%s"}' % spec)
        ts.append((time.time() - t0) * 1000)
        if '"ok":true' not in r:
            p.kill()
            return None, "run " + r[:60]
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try:
        p.wait(timeout=30)
    except Exception:
        p.kill()
    ts.sort()
    return ts[len(ts) // 2], None


print("每个 ADD 处理 [%d,%d] fp16 = %.1f MB" % (S, K, S * K * 2 / 1e6))
print()
print("%-8s %-12s %-12s" % ("ADD数", "耗时ms", "每算子ms"))
prev = None
for n in (1, 2, 4, 8, 16, 32):
    t, err = bench(n)
    if t is None:
        print("%-8d 失败: %s" % (n, err))
        continue
    per = t / n
    delta = ""
    if prev is not None:
        dt = t - prev[1]
        dn = n - prev[0]
        delta = "  增量 %.2f ms/算子" % (dt / dn) if dn else ""
    print("%-8d %-12.1f %-12.2f%s" % (n, t, per, delta))
    prev = (n, t)
