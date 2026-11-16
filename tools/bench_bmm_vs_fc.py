"""BATCH_MATMUL vs FC：同一个矩阵乘，哪个算子更便宜？

背景
  lowering.py 对 2D@2D 的 matmul 走 BATCH_MATMUL，对 3D@2D 走 FC。
  DiT 的 MLP 是 x[1024,2048] @ W[8192,2048].T，输入是 2D —— 所以走的是 BMM。
  op_cost.py 估算显示：28 个 BATCH_MATMUL 占了 61% 的搬运量，其中最大的 4 个
  就是 MLP 的 up 投影（输出 [1024,8192]）。

  而 FC 是 MDLA 专用的全连接算子，理论上有更好的硬件路径。
  如果 FC 更便宜，改 lowering 的一个分支就能动到 60% 的时间。

测什么
  同一份数学运算：
      [1024, 2048] @ [2048, 8192] -> [1024, 8192]
  两种表达：
      BMM:  BATCH_MATMUL([1,1024,2048], [1,2048,8192])
      FC:   FC([1024,2048], [8192,2048])
  两者只差算子类型，输入输出逐元素应当相同。

怎么判
  1. 数值：两个输出 relative error < 1% 才算等价
  2. 时间：各跑 5 次取最小
  3. 如果 FC 快 —— 改 lowering 的 2D@2D 分支

用法（设备上，Termux）：
    python3 -P bench_bmm_vs_fc.py
环境变量：
    KA_M 序列长度（默认 1024）
    KA_K 输入维（默认 2048）
    KA_N 输出维（默认 8192）
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
T = os.environ.get("KA_WORK", "/data/local/tmp/ka_bmf")
REPS = 5
MIN_FREE_MB = 900

os.makedirs(T, exist_ok=True)


def mem_available_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1


class Host:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.banner = self.p.stdout.readline().strip()

    def cmd(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        try:
            self.cmd('{"cmd":"quit"}')
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def make_graphs():
    """造两张图。权重只生成一次，BMM 用转置版。"""
    rng = np.random.default_rng(7)
    w_path = os.path.join(T, "w_fc.bin")          # [N, K] 给 FC
    wt_path = os.path.join(T, "w_bmm.bin")        # [K, N] 给 BMM
    b_path = os.path.join(T, "bias.bin")          # [N]

    if not os.path.exists(w_path):
        # 分块生成，避免一次性吃满内存（这机器 OOM 过）
        step = 1024
        with open(w_path, "wb") as fw, open(wt_path, "wb") as ft:
            for r0 in range(0, N, step):
                r1 = min(r0 + step, N)
                blk = (rng.standard_normal((r1 - r0, K)) * 0.02).astype(np.float16)
                blk.tofile(fw)
                # 转置块写进 BMM 权重：wt[k][r0:r1]
                # 需要按列写，先缓冲整列太慢，改为直接把整块转置后按行写
                blk.T.astype(np.float16).tofile(ft)
            # 上面的写法不适用于 wt：需要 [K, N] 行主序
        # 重新按正确顺序生成 wt（一次性，K*N*2 = 33MB，安全）
        arr = np.fromfile(w_path, dtype=np.float16).reshape(N, K)
        arr.T.astype(np.float16).tofile(wt_path)
        np.zeros(N, np.float16).tofile(b_path)

    x_path = os.path.join(T, "x.bin")
    if not os.path.exists(x_path):
        x = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
        x.tofile(x_path)

    act = os.path.join(T, "act.bin")
    np.array([0], np.int32).tofile(act)

    g_bmm = os.path.join(T, "g_bmm.txt")
    open(g_bmm, "w").write("\n".join([
        "OPD 0 8 3 1 %d %d 0 0 i" % (M, K),
        "OPD 1 8 3 1 %d %d 0 0 w %s" % (K, N, wt_path),
        "OPD 2 4 1 1 0 0 a 0",
        "OPD 3 4 1 1 0 0 a 0",
        "OPD 4 8 3 1 %d %d 0 0 o" % (M, N),
        "OP 102 4 0 1 2 3 1 4",
        "IO 1 0 1 4",
        "END",
    ]))

    g_fc = os.path.join(T, "g_fc.txt")
    open(g_fc, "w").write("\n".join([
        "OPD 0 8 2 %d %d 0 0 i" % (M, K),
        "OPD 1 8 2 %d %d 0 0 w %s" % (N, K, w_path),
        "OPD 2 8 1 %d 0 0 w %s" % (N, b_path),
        "OPD 3 4 1 1 0 0 w %s" % act,
        "OPD 4 8 2 %d %d 0 0 o" % (M, N),
        "OP 9 4 0 1 2 3 1 4",
        "IO 1 0 1 4",
        "END",
    ]))
    return g_bmm, g_fc, x_path


def bench(gpath, tag, x_path):
    print("\n[%s] %s" % (tag, gpath), flush=True)
    out = os.path.join(T, "out_%s.bin" % tag)
    h = Host(HOST)
    try:
        r = h.cmd('{"cmd":"build","path":"%s"}' % gpath)
        if '"ok":true' not in r:
            print("  build 失败:", r[:110])
            return None, None
        t0 = time.time()
        r = h.cmd('{"cmd":"compile"}')
        ct = time.time() - t0
        if '"ok":true' not in r:
            print("  compile 失败:", r[:110])
            return None, None
        spec = '1 0 %s 1 %s' % (x_path, out)
        r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
        if '"ok":true' not in r:
            print("  run 失败:", r[:110])
            return None, None
        best = None
        for _ in range(REPS):
            t0 = time.time()
            r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
            dt = (time.time() - t0) * 1000
            if '"ok":true' not in r:
                break
            best = dt if best is None else min(best, dt)
        print("  编译 %.1f s   run %.1f ms" % (ct, best))
        return best, np.fromfile(out, dtype=np.float16)
    finally:
        h.close()


def main():
    avail = mem_available_mb()
    print("M=%d K=%d N=%d   可用内存 %.0f MB" % (M, K, N, avail))
    if 0 < avail < MIN_FREE_MB:
        print("内存不足 %d MB，退出" % MIN_FREE_MB)
        return 4

    g_bmm, g_fc, x_path = make_graphs()
    tb, ob = bench(g_bmm, "bmm", x_path)
    tf, of = bench(g_fc, "fc", x_path)

    print("\n" + "=" * 56)
    if ob is not None and of is not None and ob.size == of.size:
        a = ob.astype(np.float32).ravel()
        b = of.astype(np.float32).ravel()
        scale = max(float(np.abs(a).max()), 1e-9)
        rel = float(np.abs(a - b).max()) / scale
        print("数值：max|Δ| 相对 %.4f%%  %s" % (rel * 100, "等价" if rel < 0.01 else "不等价！"))
    if tb and tf:
        print("BMM : %.1f ms" % tb)
        print("FC  : %.1f ms" % tf)
        print("加速比 %.2fx  (>1 表示 FC 更快)" % (tb / tf))
        # 单步里这类算子有 4 个（MLP up/down × 2 层块）
        print("按单块 4 个算，能省 %.1f ms/块" % ((tb - tf) * 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
