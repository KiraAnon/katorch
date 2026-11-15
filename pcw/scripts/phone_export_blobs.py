"""一次性导出所有块的编译产物 blob（冷编译 → storeCompiledNetwork）。"""
import os, subprocess, time

H = "/data/data/com.termux/files/home"
D = H + "/npt"
E2E = D + "/e2eper256"
OUT = D + "/gen5"
HOST = D + "/host_new"
BLOBDIR = "/data/local/tmp/blobs"
os.makedirs(BLOBDIR, exist_ok=True)

TARGETS = [("b%02d" % i, "%s/b%02d/g.txt" % (E2E, i)) for i in range(14)]
TARGETS += [("pe", OUT + "/pe/g.txt"), ("fl", OUT + "/fl/g.txt")]

def spawn():
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    return p

def s(p, c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

t_all = time.time()
for name, g in TARGETS:
    blob = "%s/%s.dla" % (BLOBDIR, name)
    if os.path.exists(blob) and os.path.getsize(blob) > 1 << 20:
        print("  %-4s 已存在 %.0f MB，跳过" % (name, os.path.getsize(blob) / 1e6), flush=True)
        continue
    t0 = time.time()
    p = spawn()
    r = s(p, '{"cmd":"build","path":"%s"}' % g)
    if '"ok":true' not in r:
        print("  %-4s build 失败 %s" % (name, r[:60]), flush=True)
        p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
        continue
    r = s(p, '{"cmd":"compile"}')          # 冷编译（未设 cachedir）
    if '"ok":true' not in r:
        print("  %-4s compile 失败 %s" % (name, r[:60]), flush=True)
        p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
        continue
    r = s(p, '{"cmd":"export","path":"%s"}' % blob)
    ok = '"ok":true' in r
    sz = os.path.getsize(blob) if os.path.exists(blob) else 0
    print("  %-4s %5.1fs  %s  %.0f MB" % (name, time.time() - t0, "OK" if ok else r[:50], sz / 1e6), flush=True)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()

total = sum(os.path.getsize(os.path.join(BLOBDIR, f)) for f in os.listdir(BLOBDIR))
print("\n总耗时 %.0fs，blob 合计 %.2f GB" % (time.time() - t_all, total / 1e9))
