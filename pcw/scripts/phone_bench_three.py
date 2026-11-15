"""实测三种方案跑完 14 块的真实耗时。不估算，直接测。"""
import subprocess, time, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
E2E = D + "/e2eper256"
OUT = D + "/gen5"
CACHE = E2E + "/nc"
HOST = D + "/host_new"
B = "/data/local/tmp/blobs"

INS = [OUT + "/i_xb.bin", OUT + "/i_te.bin", OUT + "/i_lsh.bin", OUT + "/i_lsc.bin",
       OUT + "/i_lga.bin", OUT + "/i_ctx_pos.bin", OUT + "/i_cos.bin", OUT + "/i_sin.bin"]
SCRATCH = "/data/local/tmp/scratch_out.bin"

for f in INS:
    if not os.path.exists(f):
        print("缺少输入 %s" % f)
        raise SystemExit(1)

def spec(ins, out):
    return "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), out)

def spawn():
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    return p

def s(p, c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

def bench(label, fn):
    print("\n=== %s ===" % label, flush=True)
    t0 = time.time()
    ok = fn()
    dt = time.time() - t0
    print("  完成 %d/14 块   总耗时 %.2f s   每块 %.0f ms" % (ok, dt, dt / 14 * 1000), flush=True)
    return dt

# ---------- 方案 A：current（每块 build + compile + run） ----------
def path_a():
    p = spawn()
    s(p, '{"cmd":"cachedir","path":"%s"}' % CACHE)
    ok = 0
    for i in range(14):
        s(p, '{"cmd":"cachekey","v":"b%02d"}' % i)
        r = s(p, '{"cmd":"build","path":"%s/b%02d/g.txt"}' % (E2E, i))
        if '"ok":true' not in r: print("  b%02d build fail %s" % (i, r[:40])); break
        r = s(p, '{"cmd":"compile"}')
        if '"ok":true' not in r: print("  b%02d compile fail %s" % (i, r[:40])); break
        r = s(p, '{"cmd":"run","spec":"%s"}' % spec(INS, SCRATCH))
        if '"ok":true' not in r: print("  b%02d run fail %s" % (i, r[:40])); break
        ok += 1
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    return ok

# ---------- 方案 B：restore，不常驻（每块恢复后立即释放） ----------
def path_b():
    p = spawn()
    ok = 0
    for i in range(14):
        r = s(p, '{"cmd":"rpush","path":"%s/b%02d.dla"}' % (B, i))
        if '"ok":true' not in r: print("  b%02d restore fail %s" % (i, r[:50])); break
        r = s(p, '{"cmd":"run","spec":"%s"}' % spec(INS, SCRATCH))
        if '"ok":true' not in r: print("  b%02d run fail %s" % (i, r[:50])); break
        s(p, '{"cmd":"pop"}')
        ok += 1
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    return ok

# ---------- 方案 C：5 常驻 + 轮换 ----------
def path_c():
    p = spawn()
    ok = 0
    held = []          # 当前常驻的块编号
    for i in range(5):
        r = s(p, '{"cmd":"rpush","path":"%s/b%02d.dla"}' % (B, i))
        if '"ok":true' not in r: print("  预载 b%02d fail %s" % (i, r[:50])); break
        held.append(i)
    for i in range(14):
        if i not in held:
            # 换出最早的一个，换入 i
            old = held.pop(0)
            s(p, '{"cmd":"pop"}')
            r = s(p, '{"cmd":"rpush","path":"%s/b%02d.dla"}' % (B, i))
            if '"ok":true' not in r: print("  换入 b%02d fail %s" % (i, r[:50])); break
            held.append(i)
        else:
            # 常驻的块要 use 对应 slot
            s(p, '{"cmd":"use","slot":%d}' % held.index(i))
        r = s(p, '{"cmd":"run","spec":"%s"}' % spec(INS, SCRATCH))
        if '"ok":true' not in r: print("  b%02d run fail %s" % (i, r[:50])); break
        ok += 1
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    return ok

ta = bench("A: build + compile + run（现行方案）", path_a)
tb = bench("B: restore + run，不常驻", path_b)
tc = bench("C: 5 常驻 + 轮换", path_c)

print("\n========== 汇总（单步 14 块）==========")
print("  A 现行       : %.2f s" % ta)
print("  B 不常驻恢复 : %.2f s  (%.2fx)" % (tb, ta / tb))
print("  C 5常驻+轮换 : %.2f s  (%.2fx)" % (tc, ta / tc))
