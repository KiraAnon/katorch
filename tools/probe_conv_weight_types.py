"""CONV_2D 到底接受哪些权重形态？

bench_conv_int8.py 发现 INT8 per-channel 权重会被 CONV 拒（compile rc=-516）。
在据此判定「量化对卷积不通」之前，先把几种可能的写法都试一遍，
避免把「我写错了」误判成「硬件不支持」。

试这几种权重 operand 形态（输入输出保持 fp16）：
  8   TENSOR_FLOAT16                    —— 基线，已知可用
  11  QUANT8_SYMM_PER_CHANNEL + QCH     —— per-channel INT8
  11  QUANT8_SYMM_PER_CHANNEL 无 QCH    —— 缺量化参数，看是否至少编过
  5   QUANT8_ASYMM                      —— per-tensor 非对称 INT8（OPD 里给 scale/zp）
  7   QUANT16_SYMM                      —— INT16，作为对照

每个都独立进程跑，互不污染。

用法：python probe_conv_weight_types.py
"""
import os
import subprocess
import sys

import numpy as np

H = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
HOST = os.environ.get("KA_HOST", H + "/npt/host_new")
T = os.environ.get("KA_WORK", "/data/local/tmp/ka_bench/work")

H_, W_, C_, O_ = 128, 128, 256, 256
KS, PAD = 3, 1

os.makedirs(T, exist_ok=True)


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


def build_graph(w_type, w_path, bias, w_scale=None, w_zp=0, qch=None):
    L, oid = [], 0
    x = oid; L.append("OPD %d 8 4 1 %d %d %d 0 0 i" % (x, H_, W_, C_)); oid += 1
    w = oid
    if w_scale is not None:
        L.append("OPD %d %d 4 %d %d %d %d %s %d w %s"
                 % (w, w_type, O_, KS, KS, C_, w_scale, w_zp, w_path))
    else:
        L.append("OPD %d %d 4 %d %d %d %d 0 0 w %s"
                 % (w, w_type, O_, KS, KS, C_, w_path))
    oid += 1
    b = oid; L.append("OPD %d 8 1 %d 0 0 w %s" % (b, O_, bias)); oid += 1
    scal = []
    for v in (PAD, PAD, PAD, PAD, 1, 1, 0):
        L.append("OPD %d 1 0 0 0 a %d" % (oid, v)); scal.append(oid); oid += 1
    o = oid; L.append("OPD %d 8 4 1 %d %d %d 0 0 o" % (o, H_, W_, O_)); oid += 1
    ins = [x, w, b] + scal
    L.append("OP 3 %d %s 1 %d" % (len(ins), " ".join(str(i) for i in ins), o))
    L.append("IO 1 %d 1 %d" % (x, o))
    if qch:
        L.append("QCH %d 0 %d %s" % (w, O_, qch))
    L.append("END")
    return "\n".join(L) + "\n"


def main():
    rng = np.random.default_rng(7)
    in_path = os.path.join(T, "px.bin")
    if not os.path.exists(in_path):
        with open(in_path, "wb") as f:
            for i in range(0, H_, 16):
                h = min(16, H_ - i)
                a = rng.standard_normal((1, h, W_, C_), dtype=np.float32)
                f.write((a * np.float32(0.1)).astype(np.float16).tobytes())
    bias = os.path.join(T, "pb.bin")
    np.zeros((O_,), np.float16).tofile(bias)

    W = rng.standard_normal((O_, KS, KS, C_), dtype=np.float32) * np.float32(0.02)
    W.astype(np.float16).tofile(os.path.join(T, "pw16.bin"))
    per_t = float(np.abs(W).max() / 127.0)
    sw = np.abs(W).reshape(O_, -1).max(axis=1) / 127.0
    sw[sw == 0] = 1e-6
    np.clip(np.rint(W / sw[:, None, None, None]), -127, 127).astype(np.int8).tofile(
        os.path.join(T, "pw8.bin"))
    sw.astype(np.float32).tofile(os.path.join(T, "pw8.scales"))
    np.clip(np.rint(W / per_t), -127, 127).astype(np.int8).tofile(os.path.join(T, "pw8t.bin"))
    # INT16 权重（type 7 要求整数，用 per-tensor scale 放大到 16 位）
    s16 = float(np.abs(W).max() / 32767.0)
    np.clip(np.rint(W / s16), -32767, 32767).astype(np.int16).tofile(os.path.join(T, "pw16i.bin"))

    cases = [
        ("fp16 (base)",        8,  "pw16.bin",  None,           0, None),
        ("INT8 per-channel+QCH", 11, "pw8.bin",  None,           0, "pw8.scales"),
        ("INT8 per-channel noQCH", 11, "pw8.bin", None,          0, None),
        ("INT8 asymm per-tensor", 5, "pw8t.bin", "%.8f" % per_t, 0, None),
        ("INT16 symm per-tensor", 7, "pw16i.bin", "%.10f" % s16, 0, None),
    ]

    print("CONV_2D 权重形态探测（%dx%dx%d -> %d）\n" % (H_, W_, C_, O_))
    results = []
    for name, wt, wf, wsc, wzp, qch in cases:
        gp = os.path.join(T, "probe_%d.txt" % wt + ("_q" if qch else ""))
        txt = build_graph(wt, os.path.join(T, wf), bias,
                          w_scale=wsc, w_zp=wzp,
                          qch=(os.path.join(T, qch) if qch else None))
        open(gp, "w").write(txt)
        h = Host(HOST)
        try:
            r1 = h.cmd('{"cmd":"build","path":"%s"}' % gp)
            b_ok = '"ok":true' in r1
            if not b_ok:
                results.append((name, "BUILD", r1[:60])); 
                print("  %-24s BUILD  %s" % (name, r1[:70])); continue
            r2 = h.cmd('{"cmd":"compile"}')
            c_ok = '"ok":true' in r2
            results.append((name, "OK" if c_ok else "COMPILE", r2[:60]))
            print("  %-24s %s  %s" % (name, "OK  " if c_ok else "FAIL", r2[:70]))
        finally:
            h.close()

    print("\n=== 汇总 ===")
    for name, st, detail in results:
        print("  %-24s %s" % (name, st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
