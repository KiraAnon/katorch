"""CONV_2D：fp16 权重 vs INT8 per-channel 权重，谁快？

这是决定「要不要给 KAtorch 加 INT8」的关键测量。

背景：早先只测过 FC（8.6 ms -> 7.6 ms，1.14x），据此判定量化净亏。
但 CONV 在 MDLA 上是**另一条硬件通路**（fp16 下能到 700-1200 GFLOP/s，
而 FC 只有约 1 TFLOP/s），两条通路的 INT8 加速比没有理由相同。
VAE 解码器是纯卷积栈，正好落在这条通路上。

测什么
  A. CONV_2D 权重 fp16            —— 基线
  B. CONV_2D 权重 INT8 per-channel —— 量化后是否更快，以及能否编译

形态：NHWC 输入 [1,H,W,C] / OHWC 权重 [O,3,3,C]，padding 1，stride 1。
默认 256x256x384->384，和 docs/operators.md 里那次 1189.8 GFLOP/s 的测量同规格，
数字可直接对比。

用法（在设备上）：
    python bench_conv_int8.py
    KA_WORK=/data/local/tmp/convbench python bench_conv_int8.py

输出判决：
  1.00x 左右        -> INT8 在 CONV 上也没加速，量化到此为止
  明显 >1.2x        -> 值得为卷积栈实现 per-channel 量化
  编译失败(-516)    -> 这条通路不接受 INT8 权重
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

H_, W_, C_, O_ = 256, 256, 384, 384
KS = 3
PAD = 1
REPS = 5

os.makedirs(T, exist_ok=True)


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


def conv_graph(weight_id_type, weight_path, bias_path, out_path):
    """CONV: x, w, b, pad(l,r,t,b), stride(h,w), act -> 10 inputs."""
    L = []
    oid = 0
    x = oid; L.append("OPD %d 8 4 1 %d %d %d 0 0 i" % (x, H_, W_, C_)); oid += 1
    w = oid; L.append("OPD %d %d 4 %d %d %d %d 0 0 w %s"
                      % (w, weight_id_type, O_, KS, KS, C_, weight_path)); oid += 1
    b = oid; L.append("OPD %d 8 1 %d 0 0 w %s" % (b, O_, bias_path)); oid += 1
    scal = []
    for v in (PAD, PAD, PAD, PAD, 1, 1, 0):     # l r t b  sh sw  act
        L.append("OPD %d 1 0 0 0 a %d" % (oid, v)); scal.append(oid); oid += 1
    o = oid; L.append("OPD %d 8 4 1 %d %d %d 0 0 o" % (o, H_, W_, O_)); oid += 1
    ins = [x, w, b] + scal
    L.append("OP %d %d %s 1 %d" % (3, len(ins), " ".join(str(i) for i in ins), o))
    L.append("IO 1 %d 1 %d" % (x, o))
    L.append("END")
    return "\n".join(L) + "\n", w, o


def run_once(host, gpath, in_path, out_path):
    r = host.cmd('{"cmd":"build","path":"%s"}' % gpath)
    if '"ok":true' not in r:
        return None, "build " + r[:80]
    r = host.cmd('{"cmd":"compile"}')
    if '"ok":true' not in r:
        return None, "compile " + r[:80]
    spec = "1 0 %s 1 %s" % (in_path, out_path)
    host.cmd('{"cmd":"run","spec":"%s"}' % spec)      # warm-up
    best = None
    for _ in range(REPS):
        t0 = time.time()
        r = host.cmd('{"cmd":"run","spec":"%s"}' % spec)
        dt = (time.time() - t0) * 1000
        if '"ok":true' not in r:
            return None, "run " + r[:80]
        best = dt if best is None else min(best, dt)
    return best, None


def main():
    rng = np.random.default_rng(20261115)

    in_path = os.path.join(T, "x.bin")
    w16_path = os.path.join(T, "w16.bin")
    w8_path = os.path.join(T, "w8.bin")
    sc_path = os.path.join(T, "w8.scales")
    b_path = os.path.join(T, "b.bin")
    out16 = os.path.join(T, "o16.bin")
    out8 = os.path.join(T, "o8.bin")

    if not os.path.exists(in_path) or os.path.getsize(in_path) != H_ * W_ * C_ * 2:
        print("生成输入 (%.0f MB) ..." % (H_ * W_ * C_ * 2 / 1e6), flush=True)
        x = (rng.standard_normal((1, H_, W_, C_)) * 0.1).astype(np.float16)
        x.tofile(in_path)

    if not os.path.exists(w16_path):
        print("生成权重 (%.1f MB) ..." % (O_ * KS * KS * C_ * 2 / 1e6), flush=True)
        W = (rng.standard_normal((O_, KS, KS, C_)) * 0.02).astype(np.float16)
        W.tofile(w16_path)
        b = np.zeros((O_,), np.float16)
        b.tofile(b_path)
        # per-output-channel 量化
        Wa = W.astype(np.float32)
        sw = np.abs(Wa).reshape(O_, -1).max(axis=1) / 127.0
        sw[sw == 0] = 1e-6
        q = np.clip(np.rint(Wa / sw[:, None, None, None]), -127, 127).astype(np.int8)
        q.tofile(w8_path)
        sw.astype(np.float32).tofile(sc_path)

    flop = 2.0 * H_ * W_ * O_ * C_ * KS * KS

    print("\n基线：fp16 权重")
    host = Host(HOST)
    try:
        g16, wid16, oid16 = conv_graph(8, w16_path, b_path, out16)
        p16 = os.path.join(T, "g16.txt")
        open(p16, "w").write(g16)
        t16, err = run_once(host, p16, in_path, out16)
        if err:
            print("  fp16 也失败：", err)
            return 2
        print("  %.1f ms   %.1f GFLOP/s" % (t16, flop / t16 / 1e6))

        print("\n对照：INT8 per-channel 权重 (type 11 + QCH)")
        g8, wid8, oid8 = conv_graph(11, w8_path, b_path, out8)
        # QCH 跟着权重 operand 走：<operand id> <channelDim> <count> <scales(f32)>
        g8 = g8.replace("END", "QCH %d 0 %d %s\nEND" % (wid8, O_, sc_path))
        p8 = os.path.join(T, "g8.txt")
        open(p8, "w").write(g8)
        t8, err = run_once(host, p8, in_path, out8)
        if err:
            print("  失败：", err)
            print("\n判决：CONV_2D 在这颗 MDLA 上不接受 INT8 权重（量化这条路对卷积不通）")
            return 3

        print("  %.1f ms   %.1f GFLOP/s" % (t8, flop / t8 / 1e6))
        print("\n加速比：%.2fx" % (t16 / t8))
        if t16 / t8 < 1.15:
            print("判决：INT8 在 CONV 上也没有实质加速 —— 量化到此为止，不值得实现")
        else:
            print("判决：CONV 的 INT8 有真实收益 —— 值得为纯卷积栈实现 per-channel 量化")

        # 精度：把 INT8 输出和 fp16 输出比一下（只作参考，两者 scale 不同）
        if os.path.exists(out16) and os.path.exists(out8):
            a = np.fromfile(out16, dtype=np.float16)
            bnb = os.path.getsize(out8)
            print("\n输出：fp16 %d 元素；int8 图输出 %d 字节" % (a.size, bnb))
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
