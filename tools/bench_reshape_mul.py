"""造图对比：RESHAPE 链 vs MUL 链 vs 空转，看每个算子真实成本。"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/reshapecost"
os.makedirs(T, exist_ok=True)

S, K = 1024, 2048
NB = S * K * 2                       # 4MB

# 输入 + 常量
a = (np.random.default_rng(0).standard_normal((S, K)) * 0.01).astype(np.float16)
a.tofile(T + "/a.bin")
a.tofile(T + "/b.bin")

# 交替 shape：[1024,2048] ↔ [1,1024,2048]
shp2d = np.array([S, K], np.int32)
shp3d = np.array([1, S, K], np.int32)
shp2d.tofile(T + "/s2.bin")
shp3d.tofile(T + "/s3.bin")


def graph_reshape(n):
    """n 个 RESHAPE，交替 2D/3D。"""
    L = ["OPD 0 8 2 %d %d 0 0 i" % (S, K)]
    prev = 0
    oid = 1
    for i in range(n):
        d = "[1,%d,%d]" % (S, K) if i % 2 == 0 else "[%d,%d]" % (S, K)
        dims = "3 %d %d %d" % (1, S, K) if i % 2 == 0 else "2 %d %d" % (S, K)
        sf = "%s/s3.bin" % T if i % 2 == 0 else "%s/s2.bin" % T
        L.append("OPD %d 8 %s 0 0 o" % (oid, dims))
        L.append("OPD %d 4 1 2 0 0 w %s" % (oid + 1, sf))
        L.append("OP 22 2 %d %d 1 %d" % (prev, oid + 1, oid))
        prev = oid
        oid += 2
    # 输出形状必须和输入一致（[S,K]）
    if n % 2 == 1:
        L.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
        L.append("OP 22 2 %d %d 1 %d" % (prev, oid + 1, oid))
        L.append("OPD %d 4 1 2 0 0 w %s/s2.bin" % (oid + 1, T))
        prev = oid
    L.append("IO 1 0 1 %d" % prev)
    L.append("END")
    return "\n".join(L) + "\n"


def graph_mul(n):
    """n 个 MUL（同 shape 相乘）。"""
    L = ["OPD 0 8 2 %d %d 0 0 i" % (S, K),
         "OPD 1 8 2 %d %d 0 0 w %s/b.bin" % (S, K, T),
         "OPD 2 4 1 1 0 0 a 0"]
    prev = 0
    oid = 3
    for i in range(n):
        L.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
        L.append("OP 18 3 %d 1 2 1 %d" % (prev, oid))
        prev = oid
        oid += 1
    L.append("IO 1 0 1 %d" % prev)
    L.append("END")
    return "\n".join(L) + "\n"


def bench(gpath, inp, tag, reps=5):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gpath)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:70]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:70]
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


print("数据: [%d,%d] fp16 = %.1f MB/张" % (S, K, NB / 1e6))
print()
print("%-10s %-8s %-12s %s" % ("类型", "个数", "耗时ms", "每算子ms"))
for name, gen in (("RESHAPE", graph_reshape), ("MUL", graph_mul)):
    for n in (1, 4, 16):
        gp = "%s/g_%s_%d.txt" % (T, name, n)
        open(gp, "w").write(gen(n))
        t, err = bench(gp, T + "/a.bin", "%s_%d" % (name, n))
        if t is None:
            print("%-10s %-8d 失败 %s" % (name, n, err))
        else:
            print("%-10s %-8d %-12.1f %.2f" % (name, n, t, t / n))
