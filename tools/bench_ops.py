"""算子性能矩阵：把所有能用的算子按典型尺寸测一遍，出一张可以查的表。

为什么要有这个
  MDLA 上同一个数学运算常有多种写法，价格差得离谱。已经实测到的：
      BATCH_MATMUL [K,N] 布局  53.0 ms   vs  FC [N,K]  28.5 ms    1.86x
      NEG + MAXIMUM 求绝对值和 ABS 谁快？没人测过。
  逐个试错太慢，而且容易漏。这个脚本把「算子 × 尺寸」系统扫一遍，
  输出 JSON + Markdown 两张表，编译器可以查，人也可以看。

测什么
  每类算子挑几个真实模型里会出现的尺寸（来自 DiT/VAE 的实际图），
  构造最小可编译的图，测稳态执行时间。

输出
  op_table.json   机器可读（编译器用来选算子）
  op_table.md     人看的表格

用法（设备上）
    python3 -P bench_ops.py                     # 跑全部
    python3 -P bench_ops.py --only matmul,unary # 只跑部分
    python3 -P bench_ops.py --out /tmp/tbl      # 指定输出前缀

注意
  · 每个算子只测「单算子图」，不含上下文。多算子融合后的表现可能不同，
    但这张表回答的是「同一个运算换写法值不值」，那正是单算子对比。
  · 时间取多次最小值，减少调度抖动。
  · 编译失败会被记为 unsupported，这本身也是有用的信息。
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HOST = os.environ.get("KA_HOST", "/data/data/com.termux/files/home/npt/host_new")
T = os.environ.get("KA_WORK", "/data/local/tmp/ka_ops")
REPS = 8
MIN_FREE_MB = 800

TYPE_F16 = 8
TYPE_I32 = 4

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


def f16(path, shape, scale=0.5, seed=0):
    if not os.path.exists(path):
        rng = np.random.default_rng(seed)
        a = (rng.standard_normal(shape) * scale).astype(np.float16)
        a.tofile(path)
    return path


def i32(path, vals):
    if not os.path.exists(path):
        np.array(vals, np.int32).tofile(path)
    return path


# ---------------- 用例定义 ----------------
# 每个 case: (组, 名字, 图文本, 输入文件列表, 输出形状)

def build_cases():
    C = []
    T1 = os.path.join(T, "c1.bin")

    # ---- 一元逐元素：同样大小的输入，比较不同算子的成本 ----
    for name, op in [("RELU", 19), ("LOGISTIC", 14), ("TANH", 28), ("EXP", 49),
                     ("NEG", 67), ("ABS", 38), ("RSQRT", 83), ("SQRT", 88),
                     ("SIN", 85), ("LOG", 60)]:
        x = f16(os.path.join(T, "u_%d.bin" % op), (1024, 2048))
        g = "\n".join([
            "OPD 0 8 2 1024 2048 0 0 i",
            "OPD 1 8 2 1024 2048 0 0 o",
            "OP %d 1 0 1 1" % op,
            "IO 1 0 1 1", "END"])
        C.append(("unary", name, g, [x], (1024, 2048)))

    # ---- 二元逐元素 ----
    xa = f16(os.path.join(T, "b_a.bin"), (1024, 2048))
    xb = f16(os.path.join(T, "b_b.bin"), (1024, 2048), seed=1)
    act = i32(os.path.join(T, "act0.bin"), [0])
    # ADD/MUL/SUB/DIV 吃 3 个输入（x, y, act）；MAXIMUM/MINIMUM 只吃 2 个。
    # 真实图里的证据：OP 0 3 ... 是 ADD 带 act，OP 65 2 ... 是 MAXIMUM 不带。
    for name, op, n_in in [("ADD", 0, 3), ("MUL", 18, 3), ("SUB", 36, 3), ("DIV", 30, 3),
                           ("MAXIMUM", 65, 2), ("MINIMUM", 66, 2)]:
        if n_in == 3:
            g = "\n".join([
                "OPD 0 8 2 1024 2048 0 0 i",
                "OPD 1 8 2 1024 2048 0 0 i",
                "OPD 2 4 1 1 0 0 w %s" % act,
                "OPD 3 8 2 1024 2048 0 0 o",
                "OP %d 3 0 1 2 1 3" % op,
                "IO 2 0 1 1 3", "END"])
        else:
            # id 必须连续（0,1,2）—— host 的 M_addOp 按出现顺序分配，
            # 跳号会让 M_setVal 对不上，build 直接失败。这个坑踩过两次。
            g = "\n".join([
                "OPD 0 8 2 1024 2048 0 0 i",
                "OPD 1 8 2 1024 2048 0 0 i",
                "OPD 2 8 2 1024 2048 0 0 o",
                "OP %d 2 0 1 1 2" % op,
                "IO 2 0 1 1 2", "END"])
        C.append(("binary", name, g, [xa, xb], (1024, 2048)))

    # ---- 求绝对值：NEG+MAX 两算子 vs 单 ABS ----
    g = "\n".join([
        "OPD 0 8 2 1024 2048 0 0 i",
        "OPD 1 8 2 1024 2048 0 0 o",
        "OPD 2 8 2 1024 2048 0 0 o",
        "OP 67 1 0 1 1",
        "OP 65 2 0 1 1 2",
        "IO 1 0 1 2", "END"])
    C.append(("pattern", "ABS_via_NEG_MAX", g, [xa], (1024, 2048)))

    # ---- 归约（末轴）----
    ax = i32(os.path.join(T, "axis1.bin"), [1])
    for name, op in [("MEAN", 31), ("REDUCE_SUM", 80), ("REDUCE_MAX", 77)]:
        g = "\n".join([
            "OPD 0 8 2 1024 2048 0 0 i",
            "OPD 1 4 1 1 0 0 w %s" % ax,
            "OPD 2 4 1 1 0 0 w %s" % act,
            "OPD 3 8 2 1024 1 0 0 o",
            "OP %d 3 0 1 2 1 3" % op,
            "IO 1 0 1 3", "END"])
        C.append(("reduce", name + "_axis1", g, [xa], (1024, 1)))

    # ---- 转置 ----
    perm = i32(os.path.join(T, "perm10.bin"), [1, 0])
    g = "\n".join([
        "OPD 0 8 2 2048 1024 0 0 i",
        "OPD 1 4 1 2 0 0 w %s" % perm,
        "OPD 2 8 2 1024 2048 0 0 o",
        "OP 37 2 0 1 1 2",
        "IO 1 0 1 2", "END"])
    x2048 = f16(os.path.join(T, "tr_in.bin"), (2048, 1024))
    C.append(("shape", "TRANSPOSE_2D", g, [x2048], (1024, 2048)))

    shp = i32(os.path.join(T, "shp.bin"), [2048, 1024])
    g = "\n".join([
        "OPD 0 8 2 1024 2048 0 0 i",
        "OPD 1 4 1 2 0 0 w %s" % shp,
        "OPD 2 8 2 2048 1024 0 0 o",
        "OP 22 2 0 1 1 2",
        "IO 1 0 1 2", "END"])
    C.append(("shape", "RESHAPE", g, [xa], (2048, 1024)))

    # ---- 矩阵乘：同一个运算的三种写法 ----
    K, N, M = 2048, 8192, 1024
    xm = f16(os.path.join(T, "mm_x.bin"), (M, K))
    w_nk = f16(os.path.join(T, "mm_nk.bin"), (N, K), scale=0.02, seed=2)
    w_kn = f16(os.path.join(T, "mm_kn.bin"), (K, N), scale=0.02, seed=3)
    bias = f16(os.path.join(T, "mm_b.bin"), (N,), scale=0.0)

    g = "\n".join([
        "OPD 0 8 2 %d %d 0 0 i" % (M, K),
        "OPD 1 8 2 %d %d 0 0 w %s" % (N, K, w_nk),
        "OPD 2 8 1 %d 0 0 w %s" % (N, bias),
        "OPD 3 4 1 1 0 0 w %s" % act,
        "OPD 4 8 2 %d %d 0 0 o" % (M, N),
        "OP 9 4 0 1 2 3 1 4",
        "IO 1 0 1 4", "END"])
    C.append(("matmul", "FC_%dx%dx%d" % (M, K, N), g, [xm], (M, N)))

    g = "\n".join([
        "OPD 0 8 3 1 %d %d 0 0 i" % (M, K),
        "OPD 1 8 3 1 %d %d 0 0 w %s" % (K, N, w_kn),
        "OPD 2 4 1 1 0 0 w %s" % act,
        "OPD 3 4 1 1 0 0 w %s" % act,
        "OPD 4 8 3 1 %d %d 0 0 o" % (M, N),
        "OP 102 4 0 1 2 3 1 4",
        "IO 1 0 1 4", "END"])
    C.append(("matmul", "BMM_KN_adj00", g, [xm], (M, N)))

    one = os.path.join(T, "one.bin")
    if not os.path.exists(one):
        np.array([1], np.int32).tofile(one)
    g = "\n".join([
        "OPD 0 8 3 1 %d %d 0 0 i" % (M, K),
        "OPD 1 8 3 1 %d %d 0 0 w %s" % (N, K, w_nk),
        "OPD 2 4 1 1 0 0 w %s" % act,
        "OPD 3 4 1 1 0 0 w %s" % one,
        "OPD 4 8 3 1 %d %d 0 0 o" % (M, N),
        "OP 102 4 0 1 2 3 1 4",
        "IO 1 0 1 4", "END"])
    C.append(("matmul", "BMM_NK_adj01", g, [xm], (M, N)))

    # ---- softmax 两种写法（fp16 下原生是坏的，这里只量成本，正确性另行判断）----
    g = "\n".join([
        "OPD 0 8 2 1024 2048 0 0 i",
        "OPD 1 4 1 1 0 0 w %s" % ax,
        "OPD 2 4 1 1 0 0 w %s" % act,
        "OPD 3 8 2 1024 1 0 0 o",
        "OPD 4 8 2 1024 2048 0 0 o",
        "OPD 5 8 2 1024 2048 0 0 o",
        "OPD 6 8 2 1024 1 0 0 o",
        "OPD 7 8 2 1024 2048 0 0 o",
        "OP 77 3 0 1 2 1 3",
        "OP 36 3 0 3 2 1 4",
        "OP 49 1 4 1 5",
        "OP 80 3 5 1 2 1 6",
        "OP 30 3 5 6 2 1 7",
        "IO 1 0 1 7", "END"])
    C.append(("pattern", "SOFTMAX_unroll5", g, [xa], (1024, 2048)))

    return C


def run_case(case):
    grp, name, gtext, inputs, outshape = case
    tag = name.replace("/", "_")
    gp = os.path.join(T, "g_%s.txt" % tag)
    open(gp, "w").write(gtext)
    outp = os.path.join(T, "o_%s.bin" % tag)
    h = Host()
    try:
        r = h.cmd('{"cmd":"build","path":"%s"}' % gp)
        if '"ok":true' not in r:
            return {"group": grp, "name": name, "status": "build_fail", "detail": r[:80]}
        t0 = time.time()
        r = h.cmd('{"cmd":"compile"}')
        ct = time.time() - t0
        if '"ok":true' not in r:
            return {"group": grp, "name": name, "status": "unsupported", "detail": r[:80]}
        spec = "%d %s 1 %s" % (len(inputs),
                               " ".join("%d %s" % (i, p) for i, p in enumerate(inputs)), outp)
        r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
        if '"ok":true' not in r:
            return {"group": grp, "name": name, "status": "run_fail", "detail": r[:80]}
        best = None
        for _ in range(REPS):
            t0 = time.time()
            r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
            dt = (time.time() - t0) * 1000
            if '"ok":true' not in r:
                break
            best = dt if best is None else min(best, dt)
        nbytes = 1
        for d in outshape:
            nbytes *= d
        nbytes *= 2
        return {"group": grp, "name": name, "status": "ok",
                "ms": round(best, 2), "compile_s": round(ct, 2),
                "out_bytes": nbytes}
    finally:
        h.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=os.path.join(T, "op_table"))
    args = ap.parse_args()

    av = mem_avail()
    print("可用内存 %.0f MB" % av)
    if 0 < av < MIN_FREE_MB:
        print("内存不足，退出"); return 4

    cases = build_cases()
    if args.only:
        keep = set(x.strip() for x in args.only.split(","))
        cases = [c for c in cases if c[0] in keep]
    print("共 %d 个用例\n" % len(cases))

    rows = []
    for i, c in enumerate(cases):
        r = run_case(c)
        rows.append(r)
        mark = {"ok": "%.2f ms" % r["ms"] if r.get("ms") else "",
                "unsupported": "不支持", "build_fail": "build失败",
                "run_fail": "run失败"}.get(r["status"], r["status"])
        print("[%2d/%2d] %-10s %-24s %s" % (i + 1, len(cases), r["group"], r["name"], mark), flush=True)

    json.dump(rows, open(args.out + ".json", "w"), ensure_ascii=False, indent=1)

    # Markdown 表
    lines = ["# 算子性能表（MT6897 / 真机实测）", "",
             "| 组 | 算子/写法 | 时间 | 状态 |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["group"], x["name"])):
        t = ("%.2f ms" % r["ms"]) if r.get("ms") else "-"
        lines.append("| %s | %s | %s | %s |" % (r["group"], r["name"], t, r["status"]))
    lines += ["", "## 同组对比", ""]
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            g[r["group"]].append(r)
    for grp, rs in g.items():
        if len(rs) < 2:
            continue
        lines.append("### %s" % grp)
        base = min(rs, key=lambda x: x["ms"])
        for r in sorted(rs, key=lambda x: x["ms"]):
            lines.append("- %-24s %8.2f ms   (%.2fx vs 最快)"
                         % (r["name"], r["ms"], r["ms"] / base["ms"]))
        lines.append("")
    open(args.out + ".md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n已写出 %s.json / %s.md" % (args.out, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
