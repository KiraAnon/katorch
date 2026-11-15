"""单个 DiT block：fp16 图 vs W8A16 图，执行时间差多少？

背景
  · 早先的记录说 W8A16「17.3 s/步 > fp16 14.6 s/步」，但那是整模型口径，而且绕法很粗暴。
  · 后来 FC 单算子实测只有 1.14x。单算子测不出带宽收益（那是计算受限），
    所以真正的答案只能在「一层真实图」上量。
  · 这个脚本就是干这个：同一个 b00 图，fp16 版和 q16_convert 转出来的 W8A16 版，
    各自 build → compile → run，比较稳态执行时间。

注意输入格式不同
  fp16 版输入是 TENSOR_FLOAT16(type 8)
  W8A16 版输入是 QUANT16_SYMM(type 7)，带 scale（脚本从图里读出来自己换算）

用法（设备上）：
    python3 -P bench_block_fp16_vs_w8.py
环境变量：
    KA_FP16_GRAPH  fp16 图路径（默认 ~/npt/e2eper512miao/b00）
    KA_W8_GRAPH    W8A16 图路径（默认 /data/local/tmp/q8b00）
    KA_WORK        临时目录
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

H = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
HOST = os.environ.get("KA_HOST", H + "/npt/host_new")
FP16 = os.environ.get("KA_FP16_GRAPH", H + "/npt/e2eper512miao/b00")
W8 = os.environ.get("KA_W8_GRAPH", "/data/local/tmp/q8b00")
T = os.environ.get("KA_WORK", "/data/local/tmp/kabench")
REPS = 3
MIN_FREE_MB = 900

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
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        try:
            self.cmd('{"cmd":"quit"}'); self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def parse_io(gpath):
    """返回 [(operand_id, type, dims, scale)], out_ids"""
    ins, outs, decl = [], [], {}
    for ln in open(gpath, encoding="utf-8"):
        if ln.startswith("OPD "):
            t = ln.split()
            oid, ty, nd = int(t[1]), int(t[2]), int(t[3])
            dims = [int(x) for x in t[4:4 + nd]]
            scale = float(t[4 + nd]) if len(t) > 4 + nd else 0.0
            tail = t[6 + nd] if len(t) > 6 + nd else ""
            decl[oid] = (ty, dims, scale)
        elif ln.startswith("IO "):
            t = ln.split()
            ni = int(t[1])
            ids = [int(x) for x in t[2:2 + ni]]
            no = int(t[2 + ni])
            oids = [int(x) for x in t[3 + ni:3 + ni + no]]
            ins = ids
            outs = oids
    return [(i, decl[i][0], decl[i][1], decl[i][2]) for i in ins], outs, decl


def make_inputs(gpath, tag, rng):
    """按图声明的类型/形状生成输入文件，返回 [(slot, path)]"""
    ins, outs, decl = parse_io(gpath)
    files = []
    for idx, (oid, ty, dims, scale) in enumerate(ins):
        n = 1
        for d in dims:
            n *= d
        p = os.path.join(T, "%s_i%d.bin" % (tag, idx))
        if ty == 8:                       # float16
            a = (rng.standard_normal(n, dtype=np.float32) * np.float32(0.5)).astype(np.float16)
        elif ty == 7:                     # QUANT16_SYMM：整数码，scale 由 OPD 给
            a = np.clip(np.rint(rng.standard_normal(n, dtype=np.float32) * 3000.0),
                        -32767, 32767).astype(np.int16)
        else:
            a = np.zeros(n, np.int32)
        a.tofile(p)
        files.append((idx, p))
    return files, outs


def run_variant(gpath, tag, rng):
    print("\n[%s] %s" % (tag, gpath), flush=True)
    inputs, outs = make_inputs(gpath, tag, rng)
    out_paths = [os.path.join(T, "%s_o%d.bin" % (tag, k)) for k in range(len(outs))]
    h = Host(HOST)
    try:
        r = h.cmd('{"cmd":"build","path":"%s"}' % gpath)
        if '"ok":true' not in r:
            print("  build 失败:", r[:90]); return None
        t0 = time.time()
        r = h.cmd('{"cmd":"compile"}')
        ctime = time.time() - t0
        if '"ok":true' not in r:
            print("  compile 失败:", r[:90]); return None
        spec = "%d %s %d %s" % (len(inputs),
                                " ".join("%d %s" % (i, p) for i, p in inputs),
                                len(out_paths), " ".join(out_paths))
        r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)     # warm-up
        if '"ok":true' not in r:
            print("  run 失败:", r[:90]); return None
        best = None
        for _ in range(REPS):
            t0 = time.time()
            r = h.cmd('{"cmd":"run","spec":"%s"}' % spec)
            dt = (time.time() - t0) * 1000
            if '"ok":true' not in r:
                print("  run 失败:", r[:90]); return None
            best = dt if best is None else min(best, dt)
        print("  编译 %.1f s   稳态 run %.1f ms" % (ctime, best))
        return best
    finally:
        h.close()


def main():
    avail = mem_available_mb()
    print("可用内存 %.0f MB" % avail)
    if avail > 0 and avail < MIN_FREE_MB:
        print("拒绝运行：内存不足 %d MB" % MIN_FREE_MB); return 4
    if not os.path.exists(os.path.join(FP16, "g.txt")):
        print("找不到 fp16 图:", FP16); return 2
    if not os.path.exists(os.path.join(W8, "g.txt")):
        print("找不到 W8 图:", W8); return 2

    rng = np.random.default_rng(42)
    t16 = run_variant(os.path.join(FP16, "g.txt"), "fp16", rng)
    tw8 = run_variant(os.path.join(W8, "g.txt"), "w8a16", rng)
    print("\n" + "=" * 50)
    if t16 and tw8:
        print("fp16 : %.1f ms" % t16)
        print("W8A16: %.1f ms" % tw8)
        print("加速比: %.2fx  (>1 表示 INT8 更快)" % (t16 / tw8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
