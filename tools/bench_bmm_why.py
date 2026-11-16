"""BATCH_MATMUL 为什么比 FC 慢 1.84 倍？把可能的原因逐个分离。

已知（实测）
    BMM([1,M,K], [1,K,N], adj=0,0)  55.1 ms
    FC ([M,K],  [N,K])              30.0 ms      1.84x

候选解释
    H1  权重布局：BMM 读 [K,N]，FC 读 [N,K]。行主序下 FC 的每一行是连续的，
        BMM 要按列跳。(K=2048 时一行 4KB，跨行步长也是 4KB —— 值得实测)
    H2  算子实现：FC 是 MDLA 的专用算子，可能有更好的 MAC 调度或分块。
    H3  维度包装：BMM 输入是 3D [1,M,K]，多一层 reshape。

怎么分离
    A  BMM, 权重 [1,K,N], adj_y=0     —— 当前的慢版本
    B  BMM, 权重 [1,N,K], adj_y=1     —— 只换权重布局 + 转置标志
    C  FC , 权重 [N,K]                —— 快版本
    D  BMM, 权重 [1,K,N] 但 K 很小    —— 看慢是否随 K 放大

    如果 B ≈ C：慢是布局造成的（H1），那么 lowering 不必多存一份转置权重，
                改成 adj_y=1 即可，省 43% 常量。
    如果 B ≈ A：慢是算子本身（H2），转置副本就是必要代价。

用法（设备上）：
    python3 -P bench_bmm_why.py
"""
import os
import subprocess
import sys
import time

import numpy as np

M = int(os.environ.get("KA_M", "1024"))
K = int(os.environ.get("KA_K", "2048"))
N = int(os.environ.get("KA_N", "8192"))
HOST = os.environ.get("KA_HOST", "/data/data/com.termux/files/home/npt/host_new")
T = os.environ.get("KA_WORK", "/data/local/tmp/ka_why")
REPS = 5
MIN_FREE_MB = 900

os.makedirs(T, exist_ok=True)


def mem_avail():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024.0
    except OSError:
        pass
    return -1


class Host:
    def __init__(self):
        self.p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdout.readline()

    def cmd(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        try:
            self.cmd('{"cmd":"quit"}'); self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def ensure_weights():
    """三种权重，尺寸已知，缺哪个补哪个。"""
    w_kn = os.path.join(T, "w_kn.bin")     # [K, N] 行主序
    w_nk = os.path.join(T, "w_nk.bin")     # [N, K] 行主序
    if not os.path.exists(w_nk):
        rng = np.random.default_rng(3)
        step = 1024
        with open(w_nk, "wb") as f:
            for r0 in range(0, N, step):
                r1 = min(r0 + step, N)
                (rng.standard_normal((r1 - r0, K)) * 0.02).astype(np.float16).tofile(f)
        arr = np.fromfile(w_nk, dtype=np.float16).reshape(N, K)
        arr.T.astype(np.float16).tofile(w_kn)
    x = os.path.join(T, "x.bin")
    if not os.path.exists(x):
        (np.random.default_rng(5).standard_normal((M, K)) * 0.5).astype(np.float16).tofile(x)
    b = os.path.join(T, "bias.bin")
    if not os.path.exists(b):
        np.zeros(N, np.float16).tofile(b)
    act = os.path.join(T, "act.bin")
    np.array([0], np.int32).tofile(act)
    return x, w_kn, w_nk, b, act


def write_graph(name, lines):
    p = os.path.join(T, "g_%s.txt" % name)
    open(p, "w").write("\n".join(lines + ["END"]))
    return p


def bench(gpath, tag, x):
    out = os.path.join(T, "o_%s.bin" % tag)
    h = Host()
    try:
        r = h.cmd('{"cmd":"build","path":"%s"}' % gpath)
        if '"ok":true' not in r:
            print("  build 失败:", r[:100]); return None, None
        r = h.cmd('{"cmd":"compile"}')
        if '"ok":true' not in r:
            print("  compile 失败:", r[:100]); return None, None
        spec = '1 0 %s 1 %s' % (x, out)
        r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
        if '"ok":true' not in r:
            print("  run 失败:", r[:100]); return None, None
        best = None
        for _ in range(REPS):
            t0 = time.time()
            r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
            dt = (time.time() - t0) * 1000
            if '"ok":true' not in r:
                break
            best = dt if best is None else min(best, dt)
        return best, np.fromfile(out, dtype=np.float16)
    finally:
        h.close()


def main():
    av = mem_avail()
    print("M=%d K=%d N=%d  可用 %.0f MB" % (M, K, N, av))
    if 0 < av < MIN_FREE_MB:
        print("内存不足"); return 4
    x, w_kn, w_nk, b, act = ensure_weights()

    cases = []

    # A: BMM + [K,N] + adj_y=0
    cases.append(("A_BMM_KN_adj0", write_graph("a", [
        "OPD 0 8 3 1 %d %d 0 0 i" % (M, K),
        "OPD 1 8 3 1 %d %d 0 0 w %s" % (K, N, w_kn),
        "OPD 2 4 1 1 0 0 a 0",
        "OPD 3 4 1 1 0 0 a 0",
        "OPD 4 8 3 1 %d %d 0 0 o" % (M, N),
        "OP 102 4 0 1 2 3 1 4",
        "IO 1 0 1 4",
    ])))

    # B: BMM + [N,K] + adj_y=1（转置标志，而非转置数据）
    cases.append(("B_BMM_NK_adj1", write_graph("b", [
        "OPD 0 8 3 1 %d %d 0 0 i" % (M, K),
        "OPD 1 8 3 1 %d %d 0 0 w %s" % (N, K, w_nk),
        "OPD 2 4 1 1 0 0 a 0",
        "OPD 3 4 1 1 0 0 a 1",
        "OPD 4 8 3 1 %d %d 0 0 o" % (M, N),
        "OP 102 4 0 1 2 3 1 4",
        "IO 1 0 1 4",
    ])))

    # C: FC
    cases.append(("C_FC", write_graph("c", [
        "OPD 0 8 2 %d %d 0 0 i" % (M, K),
        "OPD 1 8 2 %d %d 0 0 w %s" % (N, K, w_nk),
        "OPD 2 8 1 %d 0 0 w %s" % (N, b),
        "OPD 3 4 1 1 0 0 w %s" % act,
        "OPD 4 8 2 %d %d 0 0 o" % (M, N),
        "OP 9 4 0 1 2 3 1 4",
        "IO 1 0 1 4",
    ])))

    results = {}
    ref = None
    for tag, gp in cases:
        print("\n[%s]" % tag, flush=True)
        t, o = bench(gp, tag, x)
        if t:
            print("  %.1f ms" % t)
            results[tag] = t
        if o is not None:
            if ref is None:
                ref = o
            else:
                fa, fb = ref.astype(np.float32), o.astype(np.float32)
                scale = max(float(np.abs(fa).max()), 1e-9)
                print("  与 A 相比 max|Δ| 相对 = %.4f%%" % (np.abs(fa-fb).max()/scale*100))

    print("\n" + "=" * 58)
    base = results.get("A_BMM_KN_adj0")
    fc = results.get("C_FC")
    b = results.get("B_BMM_NK_adj1")
    if base and fc and b:
        print("A  BMM [K,N] adj=0 : %6.1f ms" % base)
        print("B  BMM [N,K] adj=1 : %6.1f ms   (%.2fx vs A)" % (b, base / b))
        print("C  FC  [N,K]       : %6.1f ms   (%.2fx vs A)" % (fc, base / fc))
        print()
        if b < base * 0.85:
            print("→ 慢主要是【权重布局】。改 adj_y=1 就能接近 FC，且不用存转置副本。")
        elif abs(b - base) < base * 0.1:
            print("→ adj_y 没帮助。慢在【算子实现本身】，转置副本是必要代价。")
        else:
            print("→ 部分改善，两种因素都有。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
