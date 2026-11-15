"""测不同规模的图，看耗时随规模的变化规律，判断瓶颈是算子数还是数据量。"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/sizetest"
os.makedirs(T, exist_ok=True)

# 造图：ADD 链，参数化 S（token 数）和链长
def make(S, K, n):
    a = "%s/a_%d_%d.bin" % (T, S, K)
    if not os.path.exists(a):
        rng = np.random.default_rng(0)
        (rng.standard_normal((S, K)) * 0.01).astype(np.float16).tofile(a)
    lines = [
        "OPD 0 8 2 %d %d 0 0 i" % (S, K),
        "OPD 1 8 1 1 0 0 a 1",          # 标量常量 1.0（fp16 数值直接给不了，用 type1 INT32=1）
    ]
    prev = 0
    oid = 2
    for i in range(n):
        lines.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
        lines.append("OP 0 2 %d 1 1 %d" % (prev, oid))
        prev = oid
        oid += 1
    lines.append("IO 1 0 1 %d" % prev)
    lines.append("END")
    return "\n".join(lines) + "\n", a


def run(gp, inp, tag, reps=5):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:60]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile rc=" + r[:60]
    out = "%s/o_%s.bin" % (T, tag)
    spec = "1 0 %s 1 %s" % (inp, out)
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


print("%-8s %-8s %-8s %-12s %s" % ("S", "K", "ADD数", "耗时ms", "每算子ms"))
for S, K in ((128, 2048), (512, 2048), (1024, 2048)):
    for n in (1, 4, 16):
        gp = "%s/g_%d_%d_%d.txt" % (T, S, K, n)
        txt, inp = make(S, K, n)
        open(gp, "w").write(txt)
        t, err = run(gp, inp, "%d_%d_%d" % (S, K, n))
        if t is None:
            print("%-8d %-8d %-8d 失败 %s" % (S, K, n, err))
        else:
            print("%-8d %-8d %-8d %-12.1f %.3f" % (S, K, n, t, t / n))
