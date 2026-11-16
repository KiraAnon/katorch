"""对比两版图在同一台设备上的执行时间。

用于验证「2D@2D 走 FC」这个改动是否真的更快。

输入按图的 IO 声明自动生成（形状 + 类型都从 OPD 读），所以两版图用同一套输入。

用法（设备上）：
    python3 -P bench_two_graphs.py <图A目录> <图B目录> [--reps 5]
"""
import os
import subprocess
import sys
import time

import numpy as np

HOST = os.environ.get("KA_HOST", "/data/data/com.termux/files/home/npt/host_new")
T = os.environ.get("KA_WORK", "/data/local/tmp/ka_two")
REPS = 5


def parse_io(gpath):
    ins, outs, decl = [], [], {}
    for ln in open(gpath, encoding="utf-8", errors="replace"):
        if ln.startswith("OPD "):
            t = ln.split()
            oid, ty, nd = int(t[1]), int(t[2]), int(t[3])
            dims = [int(x) for x in t[4:4 + nd]]
            decl[oid] = (ty, dims)
        elif ln.startswith("IO "):
            t = ln.split()
            ni = int(t[1])
            ins = [int(x) for x in t[2:2 + ni]]
            no = int(t[2 + ni])
            outs = [int(x) for x in t[3 + ni:3 + ni + no]]
    return [(i, decl[i][0], decl[i][1]) for i in ins], outs, decl


def make_inputs(gpath, tag):
    ins, outs, decl = parse_io(gpath)
    files = []
    rng = np.random.default_rng(11)
    for idx, (oid, ty, dims) in enumerate(ins):
        n = 1
        for d in dims:
            n *= d
        p = os.path.join(T, "%s_i%d.bin" % (tag, idx))
        if not os.path.exists(p):
            if ty == 8:
                a = (rng.standard_normal(n) * 0.5).astype(np.float16)
            elif ty == 7:
                a = np.clip(np.rint(rng.standard_normal(n) * 3000), -32767, 32767).astype(np.int16)
            else:
                a = np.zeros(n, np.int32)
            a.tofile(p)
        files.append((idx, p))
    return files, outs


def bench(gdir, tag):
    gp = os.path.join(gdir, "g.txt")
    if not os.path.exists(gp):
        print("  找不到", gp)
        return None
    files, outs = make_inputs(gp, tag)
    outp = os.path.join(T, "%s_o.bin" % tag)
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()
    def cmd(s):
        p.stdin.write(s + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()
    r = cmd('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        print("  build 失败:", r[:90]); p.kill(); return None
    t0 = time.time()
    r = cmd('{"cmd":"compile"}')
    ct = time.time() - t0
    if '"ok":true' not in r:
        print("  compile 失败:", r[:90]); p.kill(); return None
    spec = "%d %s 1 %s" % (len(files), " ".join("%d %s" % (i, f) for i, f in files), outp)
    r = cmd('{"cmd":"run","spec":"%s"}' % spec)
    if '"ok":true' not in r:
        print("  run 失败:", r[:90]); p.kill(); return None
    best = None
    for _ in range(REPS):
        t0 = time.time()
        r = cmd('{"cmd":"run","spec":"%s"}' % spec)
        dt = (time.time() - t0) * 1000
        if '"ok":true' not in r:
            break
        best = dt if best is None else min(best, dt)
    cmd('{"cmd":"quit"}'); p.wait(timeout=30)
    print("  编译 %.1f s   run %.1f ms" % (ct, best))
    return best


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    a, b = sys.argv[1], sys.argv[2]
    os.makedirs(T, exist_ok=True)
    for d in (a, b):
        m = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6
        print("%s  %.0f MB" % (d, m))
    print("\n[A] %s" % a)
    ta = bench(a, "A")
    print("[B] %s" % b)
    tb = bench(b, "B")
    print("\n" + "=" * 50)
    if ta and tb:
        print("A: %.1f ms" % ta)
        print("B: %.1f ms" % tb)
        if tb < ta:
            print("B 快 %.1f ms (%.1f%%)" % (ta - tb, (ta - tb) / ta * 100))
        else:
            print("A 快 %.1f ms (%.1f%%)" % (tb - ta, (tb - ta) / tb * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
