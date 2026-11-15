"""CONV_2D：fp16 权重 vs INT8 per-channel 权重，谁快？

这是决定「要不要给 KAtorch 加 INT8」的关键测量。

背景：早先只测过 FC（8.6 ms -> 7.6 ms，1.14x），据此判定量化净亏。
但 CONV 在 MDLA 上是**另一条硬件通路**（fp16 下能到 700-1200 GFLOP/s，
而 FC 只有约 1 TFLOP/s），两条通路的 INT8 加速比没有理由相同。

★ 内存纪律（第一次跑把手机 OOM 到 kernel panic，教训写在这）
  手机常驻着一堆吃内存的进程，可用内存可能只有 3 GB 左右。所以：
    · 输入张量一律**分块生成**，峰值控制在几十 MB，不要一次标准正态 50M 元素；
    · 默认尺寸取小（128x128x256），够测出加速比，图也不大；
    · 开跑前先读 /proc/meminfo，可用内存不够就直接拒绝，不冒险。
  要用大尺寸显式加 --big，并自己先腾内存（停掉 ComfyUI 等）。

用法：
    python bench_conv_int8.py                # 默认小尺寸
    python bench_conv_int8.py --big          # 256x256x384（需要充足内存）
    KA_WORK=/data/local/tmp/convbench python bench_conv_int8.py

判决：
  ~1.00x        -> INT8 在 CONV 上也没加速，量化到此为止
  明显 >1.2x    -> 值得为纯卷积栈实现 per-channel 量化
  编译失败      -> 这条通路不接受 INT8 权重
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

H = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
HOST = os.environ.get("KA_HOST", H + "/npt/host_new")
T = os.environ.get("KA_WORK", "/data/local/tmp/convbench")

BIG = "--big" in sys.argv
if BIG:
    H_, W_, C_, O_ = 256, 256, 384, 384          # 输入 50 MB，FP16 基线约 146 ms
else:
    H_, W_, C_, O_ = 128, 128, 256, 256          # 输入 8.4 MB，基线约 20 ms

KS, PAD, REPS = 3, 1, 5
CHUNK = 16                                        # 每次生成多少行，控制峰值内存
MIN_FREE_MB = 1200                                # 可用内存低于这个就不跑

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


def conv_graph(weight_type, weight_path, bias_path):
    """CONV: x, w, b, pad(l,r,t,b), stride(h,w), act -> 10 inputs."""
    L, oid = [], 0
    x = oid; L.append("OPD %d 8 4 1 %d %d %d 0 0 i" % (x, H_, W_, C_)); oid += 1
    w = oid; L.append("OPD %d %d 4 %d %d %d %d 0 0 w %s"
                      % (w, weight_type, O_, KS, KS, C_, weight_path)); oid += 1
    b = oid; L.append("OPD %d 8 1 %d 0 0 w %s" % (b, O_, bias_path)); oid += 1
    scal = []
    for v in (PAD, PAD, PAD, PAD, 1, 1, 0):
        L.append("OPD %d 1 0 0 0 a %d" % (oid, v)); scal.append(oid); oid += 1
    o = oid; L.append("OPD %d 8 4 1 %d %d %d 0 0 o" % (o, H_, W_, O_)); oid += 1
    L.append("OP 3 %d %s 1 %d" % (len([x, w, b] + scal),
                                  " ".join(str(i) for i in [x, w, b] + scal), o))
    L.append("IO 1 %d 1 %d" % (x, o))
    L.append("END")
    return "\n".join(L) + "\n", w


def run_once(host, gpath, in_path, out_path):
    r = host.cmd('{"cmd":"build","path":"%s"}' % gpath)
    if '"ok":true' not in r:
        return None, "build " + r[:90]
    r = host.cmd('{"cmd":"compile"}')
    if '"ok":true' not in r:
        return None, "compile " + r[:90]
    spec = "1 0 %s 1 %s" % (in_path, out_path)
    host.cmd('{"cmd":"run","spec":"%s"}' % spec)          # warm-up
    best = None
    for _ in range(REPS):
        t0 = time.time()
        r = host.cmd('{"cmd":"run","spec":"%s"}' % spec)
        dt = (time.time() - t0) * 1000
        if '"ok":true' not in r:
            return None, "run " + r[:90]
        best = dt if best is None else min(best, dt)
    return best, None


def gen_input(path, rng):
    """分块写入，避免一次性申请几十上百 MB。"""
    if os.path.exists(path) and os.path.getsize(path) == H_ * W_ * C_ * 2:
        return
    print("  生成输入 %.1f MB（分块）..." % (H_ * W_ * C_ * 2 / 1e6), flush=True)
    with open(path, "wb") as f:
        for i in range(0, H_, CHUNK):
            h = min(CHUNK, H_ - i)
            a = rng.standard_normal((1, h, W_, C_), dtype=np.float32)
            f.write((a * np.float32(0.1)).astype(np.float16).tobytes())
            del a


def gen_weights(rng):
    w16 = os.path.join(T, "w16.bin")
    if os.path.exists(w16):
        return w16
    print("  生成权重 %.2f MB ..." % (O_ * KS * KS * C_ * 2 / 1e6), flush=True)
    W = rng.standard_normal((O_, KS, KS, C_), dtype=np.float32) * np.float32(0.02)
    Wf = W.astype(np.float16)
    Wf.tofile(w16)
    np.zeros((O_,), np.float16).tofile(os.path.join(T, "b.bin"))
    # per-output-channel 量化
    sw = np.abs(W).reshape(O_, -1).max(axis=1) / 127.0
    sw[sw == 0] = 1e-6
    q = np.clip(np.rint(W / sw[:, None, None, None]), -127, 127).astype(np.int8)
    q.tofile(os.path.join(T, "w8.bin"))
    sw.astype(np.float32).tofile(os.path.join(T, "w8.scales"))
    return w16


def main():
    avail = mem_available_mb()
    print("配置 %dx%dx%d -> %d   (%.0f MB 输入)" % (H_, W_, C_, O_, H_ * W_ * C_ * 2 / 1e6))
    print("可用内存：%.0f MB" % avail)
    if avail > 0 and avail < MIN_FREE_MB:
        print("\n拒绝运行：可用内存 %.0f MB < %d MB。" % (avail, MIN_FREE_MB))
        print("先腾内存（停掉 ComfyUI / 其他大进程）再来，或者用更小的 KA_WORK。")
        return 4

    rng = np.random.default_rng(20261115)
    in_path = os.path.join(T, "x.bin")
    gen_input(in_path, rng)
    w16 = gen_weights(rng)
    flop = 2.0 * H_ * W_ * O_ * C_ * KS * KS

    host = Host(HOST)
    print("banner:", host.banner)
    try:
        print("\n[A] fp16 权重")
        g16, wid16 = conv_graph(8, w16, os.path.join(T, "b.bin"))
        p16 = os.path.join(T, "g16.txt")
        open(p16, "w").write(g16)
        t16, err = run_once(host, p16, in_path, T + "/o16.bin")
        if err:
            print("  失败:", err); return 2
        print("  %.2f ms   %.1f GFLOP/s" % (t16, flop / t16 / 1e6))

        print("\n[B] INT8 per-channel 权重 (type 11 + QCH)")
        g8, wid8 = conv_graph(11, os.path.join(T, "w8.bin"), os.path.join(T, "b.bin"))
        g8 = g8.replace("END", "QCH %d 0 %d %s\nEND" % (wid8, O_, os.path.join(T, "w8.scales")))
        p8 = os.path.join(T, "g8.txt")
        open(p8, "w").write(g8)
        t8, err = run_once(host, p8, in_path, T + "/o8.bin")
        if err:
            print("  失败:", err)
            print("\n判决：CONV_2D 不接受 INT8 权重 —— 量化对卷积这条路不通")
            return 3
        print("  %.2f ms   %.1f GFLOP/s" % (t8, flop / t8 / 1e6))

        ratio = t16 / t8
        print("\n加速比：%.2fx" % ratio)
        if ratio < 1.15:
            print("判决：INT8 在 CONV 上也没有实质加速 —— 量化到此为止")
        else:
            print("判决：CONV 的 INT8 有真实收益 —— 值得为纯卷积栈实现 per-channel 量化")
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
