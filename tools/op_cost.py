"""分析 g.txt：估算每个算子的搬运量和计算量，排出「谁最慢」。

为什么要这个
  MDLA 的图是整体编译的（284 个算子一起编译成一个 .dla），运行时不暴露 per-op 计时，
  MTK 也没有对手机开放的算子级 profiler。所以要判断「哪个算子拖慢速度」，
  只能从图本身算：每个算子的输入/输出/权重形状都在 OPD 里写着，
  搬运字节数是可以精确算出来的，再用实测带宽换算成时间。

模型（基于真机实测）
  · 带宽      8.7 GB/s（[1024,2048] 串 32 个 MUL 实测）
  · FC        约 1 TFLOP/s
  · CONV_2D   700~1200 GFLOP/s
  · 逐元素    实测受带宽限制，算力用不上

  单算子时间 ≈ max(搬运字节 / 带宽, FLOPs / 算力)

  这个模型对「带宽受限」的算子准，对计算受限的偏保守 —— 所以输出里标了 bottleneck，
  方便判断该信哪个。

用法
    python tools/op_cost.py <g.txt> [--const-root 前缀替换规则] [--top 20]
    python tools/op_cost.py F:\\KAtorch\\artifacts\\b00\\g.txt --top 25

局限（必须知道）
  · 不知道片上缓存复用：真实搬运量可能低于估算（权重可能只读一次）
  · 不知道算子调度重叠：实际是串行，但硬件可能有流水
  · 所以绝对值会偏大，看相对排序和占比更有意义
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

# ---- 实测参数 ----
BANDWIDTH_GBS = 8.7
FLOP_FC = 1.0e12
FLOP_CONV = 1.0e12
FLOP_DEFAULT = 0.5e12        # 逐元素/归约基本用不上算力，给个保守值

TYPE_BYTES = {0: 4, 1: 4, 3: 4, 4: 4, 5: 1, 7: 2, 8: 2, 11: 1, 13: 1, 14: 1}

OP_NAMES = {
    0: "ADD", 2: "CONCAT", 3: "CONV_2D", 6: "DEQUANTIZE", 9: "FC",
    14: "LOGISTIC", 18: "MUL", 19: "RELU", 22: "RESHAPE", 25: "SOFTMAX",
    28: "TANH", 30: "DIV", 31: "MEAN", 36: "SUB", 37: "TRANSPOSE",
    38: "ABS", 45: "CAST", 49: "EXP", 60: "LOG", 64: "LOG_SOFTMAX",
    65: "MAXIMUM", 66: "MINIMUM", 67: "NEG", 72: "QUANTIZE",
    77: "REDUCE_MAX", 80: "REDUCE_SUM", 83: "RSQRT", 85: "SIN",
    88: "SQRT", 102: "BATCH_MATMUL",
}

# 编译期被消除的算子：不产生真实搬运
FREE_OPS = {22, 37}          # RESHAPE / TRANSPOSE 由编译期处理（实测删掉不影响时间）


class Operand(object):
    __slots__ = ("oid", "type", "dims", "elems", "bytes", "kind", "path")

    def __init__(self, oid, type_, dims, kind, path=None):
        self.oid = oid
        self.type = type_
        self.dims = dims
        n = 1
        for d in dims:
            n *= d
        self.elems = n
        self.bytes = n * TYPE_BYTES.get(type_, 4)
        self.kind = kind          # 'i' / 'o' / 'a' / 'w'
        self.path = path


def parse(path):
    operands = {}
    io_in, io_out = [], []
    ops = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("OPD "):
            t = ln.split()
            oid = int(t[1]); ty = int(t[2]); nd = int(t[3])
            dims = [int(x) for x in t[4:4 + nd]]
            rest = t[6 + nd:]
            kind = "o"
            p = None
            if rest:
                if rest[0] == "i":
                    kind = "i"
                elif rest[0] == "a":
                    kind = "a"
                elif rest[0] == "w":
                    kind = "w"
                    p = rest[1] if len(rest) > 1 else None
            operands[oid] = Operand(oid, ty, dims, kind, p)
        elif ln.startswith("IO "):
            t = ln.split()
            ni = int(t[1])
            io_in = [int(x) for x in t[2:2 + ni]]
            no = int(t[2 + ni])
            io_out = [int(x) for x in t[3 + ni:3 + ni + no]]
        elif ln.startswith("OP "):
            t = ln.split()
            code = int(t[1]); ni = int(t[2])
            ins = [int(x) for x in t[3:3 + ni]]
            no = int(t[3 + ni])
            outs = [int(x) for x in t[4 + ni:4 + ni + no]]
            ops.append((code, ins, outs))
    return operands, io_in, io_out, ops


def flops_of(code, ins, outs, operands):
    """粗算 FLOPs。拿不准的返回 None（表示计算不是瓶颈）。"""
    try:
        o = operands[outs[0]]
        x = operands[ins[0]]
    except (IndexError, KeyError):
        return None
    if code == 9:                       # FC: (M,K) x (N,K) -> 2*M*N*K
        w = operands[ins[1]]
        if len(w.dims) >= 2:
            M = 1
            for d in x.dims[:-1]:
                M *= d
            N, K = w.dims[0], w.dims[-1]
            return 2.0 * M * N * K
    if code == 102:                     # BATCH_MATMUL
        w = operands[ins[1]]
        if len(x.dims) >= 3 and len(w.dims) >= 3:
            B, M, K = x.dims[-3], x.dims[-2], x.dims[-1]
            N = w.dims[-1]
            return 2.0 * B * M * K * N
    if code == 3:                       # CONV_2D
        return o.elems * 9.0            # 粗略按 3x3 估
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--const-root", default=None,
                    help="常量文件缺失时用什么前缀替换，例如 'F:/KAtorch/PC编译/wsl-sysroot/'")
    args = ap.parse_args()

    operands, io_in, io_out, ops = parse(args.path)
    print("图: %s" % args.path)
    print("算子 %d 个，operand %d 个，输入 %d 个，输出 %d 个"
          % (len(ops), len(operands), len(io_in), len(io_out)))

    # 主路径输出（决定搬运量的大头）
    main_out = operands[io_out[0]] if io_out and io_out[0] in operands else None
    if main_out:
        print("主输出: shape=%s, %.1f MB" % (main_out.dims, main_out.bytes / 1e6))

    # 统计每个算子的成本
    rows = []
    total_ns = 0.0
    by_code = defaultdict(lambda: {"n": 0, "ns": 0.0, "bytes": 0})
    for code, ins, outs in ops:
        in_bytes = sum(operands[i].bytes for i in ins
                       if i in operands and operands[i].kind != "a")
        out_bytes = sum(operands[o].bytes for o in outs if o in operands)
        # 权重（w 常量）单独记
        w_bytes = sum(operands[i].bytes for i in ins
                      if i in operands and operands[i].kind == "w")
        total_bytes = in_bytes + out_bytes
        fl = flops_of(code, ins, outs, operands)
        t_transport = total_bytes / (BANDWIDTH_GBS * 1e9) * 1e9      # ns
        t_compute = (fl / FLOP_DEFAULT * 1e9) if fl else 0.0
        if code == 9:
            t_compute = fl / FLOP_FC * 1e9 if fl else 0.0
        elif code == 3:
            t_compute = fl / FLOP_CONV * 1e9 if fl else 0.0
        t = max(t_transport, t_compute)

        free = code in FREE_OPS
        if free:
            t = 0.0
        rows.append({
            "code": code, "name": OP_NAMES.get(code, str(code)),
            "ns": t, "bytes": total_bytes, "w_bytes": w_bytes,
            "flops": fl, "free": free,
            "in": operands[ins[0]].dims if ins and ins[0] in operands else None,
            "out": operands[outs[0]].dims if outs and outs[0] in operands else None,
            "bottleneck": "compute" if (t_compute > t_transport) else "transport",
        })
        total_ns += t
        b = by_code[code]
        b["n"] += 1; b["ns"] += t; b["bytes"] += total_bytes

    print("\n估算总时间: %.1f ms  (带宽 %.1f GB/s)" % (total_ns / 1e6, BANDWIDTH_GBS))
    real = None
    print("  注：实测单块约 505 ms（优化后），如果估算明显偏离，说明有片上复用或重叠。")

    print("\n=== 按 opcode 汇总（估算耗时降序）===")
    print("%-14s %6s %10s %10s %9s" % ("算子", "个数", "估算ms", "占比", "搬运MB"))
    for code, b in sorted(by_code.items(), key=lambda x: -x[1]["ns"]):
        if b["ns"] < 1:
            continue
        print("%-14s %6d %10.1f %9.1f%% %9.0f"
              % (OP_NAMES.get(code, str(code)), b["n"], b["ns"] / 1e6,
                 b["ns"] / total_ns * 100, b["bytes"] / 1e6))

    print("\n=== 最贵的 %d 个算子 ===" % args.top)
    print("%-3s %-14s %9s %9s %-24s %-14s" % ("#", "算子", "估算ms", "搬运MB", "输出形状", "瓶颈"))
    for i, r in enumerate(sorted(rows, key=lambda x: -x["ns"])[:args.top]):
        print("%-3d %-14s %9.2f %9.1f %-24s %-14s"
              % (i + 1, r["name"], r["ns"] / 1e6, r["bytes"] / 1e6,
                 str(r["out"])[:24], r["bottleneck"]))

    # 被编译期消除的
    free_n = sum(1 for r in rows if r["free"])
    free_b = sum(r["bytes"] for r in rows if r["free"])
    print("\n编译期消除的算子（RESHAPE/TRANSPOSE）: %d 个，%d MB 名义搬运（实际免费）"
          % (free_n, free_b / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
