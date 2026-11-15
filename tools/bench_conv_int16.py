"""CONV_2D：fp16 权重 vs INT16 权重，速度差多少？

背景（实测链条）：
  · FC 权重 INT8 (W8A16)  -> 能编能跑，但只快 1.14x，多出来的 DQ/Q 把收益吃光
  · CONV 权重 INT8        -> 三种写法（per-channel+QCH / 无QCH / 非对称 per-tensor）
                             全部 compile rc=-516，这条通路根本不收 INT8
  · CONV 权重 INT16       -> 能编（本脚本要测的就是它快不快）

如果 INT16 在 CONV 上有实质收益，那「给纯卷积栈（VAE/vocoder）做 16 位量化」
就是这条线上唯一还站得住的量化方向；如果也是 1.0x，量化整件事到此为止。

用法：python bench_conv_int16.py [--big]
"""
import os
import subprocess
import sys
import time

import numpy as np

H = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
HOST = os.environ.get("KA_HOST", H + "/npt/host_new")
T = os.environ.get("KA_WORK", "/data/local/tmp/ka_bench/work")

BIG = "--big" in sys.argv
H_, W_, C_, O_ = (256, 256, 384, 384) if BIG else (128, 128, 256, 256)
KS, PAD, REPS = 3, 1, 5
MIN_FREE_MB = 1200

os.makedirs(T, exist_ok=True)


def mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1


class Host:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdout.readline()

    def cmd(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        try:
            self.cmd('{"cmd":"quit"}'); self.p.wait(timeout=15)
        except Exception:
            self.p.kill()


def graph(w_type, w_path, bias, w_scale=None, out_type=8):
    L, oid = [], 0
    x = oid; L.append("OPD %d 8 4 1 %d %d %d 0 0 i" % (x, H_, W_, C_)); oid += 1
    w = oid
    if w_scale is None:
        L.append("OPD %d %d 4 %d %d %d %d 0 0 w %s" % (w, w_type, O_, KS, KS, C_, w_path))
    else:
        L.append("OPD %d %d 4 %d %d %d %d %s 0 w %s"
                 % (w, w_type, O_, KS, KS, C_, w_scale, w_path))
    oid += 1
    b = oid; L.append("OPD %d 8 1 %d 0 0 w %s" % (b, O_, bias)); oid += 1
    scal = []
    for v in (PAD, PAD, PAD, PAD, 1, 1, 0):
        L.append("OPD %d 1 0 0 0 a %d" % (oid, v)); scal.append(oid); oid += 1
    o = oid; L.append("OPD %d %d 4 1 %d %d %d 0 0 o" % (o, out_type, H_, W_, O_)); oid += 1
    ins = [x, w, b] + scal
    L.append("OP 3 %d %s 1 %d" % (len(ins), " ".join(str(i) for i in ins), o))
    L.append("IO 1 %d 1 %d" % (x, o))
    L.append("END")
    return "\n".join(L) + "\n"


def time_run(gpath, in_path, out_path):
    h = Host(HOST)
    try:
        r = h.cmd('{"cmd":"build","path":"%s"}' % gpath)
        if '"ok":true' not in r:
            return None, "build " + r[:80]
        r = h.cmd('{"cmd":"compile"}')
        if '"ok":true' not in r:
            return None, "compile " + r[:80]
        spec = "1 0 %s 1 %s" % (in_path, out_path)
        h.cmd('{"cmd":"run","spec":"%s"}' % spec)
        best = None
        for _ in range(REPS):
            t0 = time.time()
            r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
            dt = (time.time() - t0) * 1000
            if '"ok":true' not in r:
                return None, "run " + r[:80]
            best = dt if best is None else min(best, dt)
        return best, None
    finally:
        h.close()


def main():
    avail = mem_available_mb()
    print("配置 %dx%dx%d -> %d   可用内存 %.0f MB" % (H_, W_, C_, O_, avail))
    if avail > 0 and avail < MIN_FREE_MB:
        print("拒绝运行：内存不足 %d MB" % MIN_FREE_MB)
        return 4

    rng = np.random.default_rng(11)
    in_path = os.path.join(T, "i16x.bin")
    if not os.path.exists(in_path):
        print("  生成输入（分块）...")
        with open(in_path, "wb") as f:
            for i in range(0, H_, 16):
                h = min(16, H_ - i)
                a = rng.standard_normal((1, h, W_, C_), dtype=np.float32)
                f.write((a * np.float32(0.1)).astype(np.float16).tobytes())

    bias = os.path.join(T, "i16b.bin")
    np.zeros((O_,), np.float16).tofile(bias)

    W = rng.standard_normal((O_, KS, KS, C_), dtype=np.float32) * np.float32(0.02)
    W.astype(np.float16).tofile(os.path.join(T, "i16w16.bin"))
    s16 = float(np.abs(W).max() / 32767.0)
    np.clip(np.rint(W / s16), -32767, 32767).astype(np.int16).tofile(os.path.join(T, "i16wi.bin"))
    flop = 2.0 * H_ * W_ * O_ * C_ * KS * KS

    print("\n[A] fp16 权重")
    g16 = os.path.join(T, "i16_g16.txt")
    open(g16, "w").write(graph(8, os.path.join(T, "i16w16.bin"), bias))
    t16, err = time_run(g16, in_path, T + "/i16_o16.bin")
    if err:
        print("  失败:", err); return 2
    print("  %.2f ms   %.1f GFLOP/s" % (t16, flop / t16 / 1e6))

    print("\n[B] INT16 权重 (type 7, per-tensor scale %.2e)" % s16)
    gi = os.path.join(T, "i16_gi.txt")
    open(gi, "w").write(graph(7, os.path.join(T, "i16wi.bin"), bias, w_scale="%.10f" % s16))
    ti, err = time_run(gi, in_path, T + "/i16_oi.bin")
    if err:
        print("  失败:", err); return 3
    print("  %.2f ms   %.1f GFLOP/s" % (ti, flop / ti / 1e6))

    print("\n加速比：%.2fx" % (t16 / ti))
    if t16 / ti < 1.15:
        print("判决：INT16 在 CONV 上也没有实质收益")
    else:
        print("判决：INT16 在 CONV 上有真实收益")
    return 0


if __name__ == "__main__":
    sys.exit(main())
