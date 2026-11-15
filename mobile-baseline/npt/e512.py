"""导出 512×512 版的 16 个 blob（冷编译 → storeCompiledNetwork）。"""
import os, subprocess, time

H = "/data/data/com.termux/files/home"
D = H + "/npt"
DST = D + "/e2eper512"
HOST = D + "/host_new"
BLOBDIR = "/data/local/tmp/blobs512"
os.makedirs(BLOBDIR, exist_ok=True)

TARGETS = [("b%02d" % i, "%s/b%02d/g.txt" % (DST, i)) for i in range(14)]
TARGETS += [("pe", DST + "/pe/g.txt"), ("fl", DST + "/fl/g.txt")]

def spawn():
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    return p

def s(p, c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

t_all = time.time()
fail = 0
for name, g in TARGETS:
    blob = "%s/%s.dla" % (BLOBDIR, name)
    if os.path.exists(blob) and os.path.getsize(blob) > 1 << 18:
        print("  %-4s 已存在 %.0f MB" % (name, os.path.getsize(blob) / 1e6), flush=True)
        continue
    t0 = time.time()
    p = spawn()
    r = s(p, '{"cmd":"build","path":"%s"}' % g)
    if '"ok":true' not in r:
        print("  %-4s build失败 %s" % (name, r[:60]), flush=True); p.kill(); fail += 1; continue
    r = s(p, '{"cmd":"compile"}')
    if '"ok":true' not in r:
        print("  %-4s compile失败 %s" % (name, r[:60]), flush=True); p.kill(); fail += 1; continue
    r = s(p, '{"cmd":"export","path":"%s"}' % blob)
    ok = '"ok":true' in r
    if not ok: fail += 1
    sz = os.path.getsize(blob) if os.path.exists(blob) else 0
    print("  %-4s %5.1fs  %s  %.0f MB" % (name, time.time() - t0,
          "OK" if ok else r[:50], sz / 1e6), flush=True)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=180)
    except: p.kill()

tot = sum(os.path.getsize(os.path.join(BLOBDIR, f)) for f in os.listdir(BLOBDIR))
with open(BLOBDIR + "/list16.txt", "w") as f:
    for name, _ in TARGETS:
        pth = "%s/%s.dla" % (BLOBDIR, name)
        if os.path.exists(pth):
            f.write(pth + "\n")
print("\n总耗时 %.0fs，失败 %d 个，合计 %.2f GB" % (time.time() - t_all, fail, tot / 1e9))
