"""对比：同一个量化 b00，build+compile 跑 vs restore blob 跑，输出是否一致。"""
import os, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
QG = os.environ.get("KA_G", "/data/local/tmp/q16i/g.txt")
BLOB = "/data/local/tmp/blobs512mix/b00.dla"
OUT = os.environ.get("KA_IN", D + "/gen5/cal/step00")
T = "/data/local/tmp/vblob"
os.makedirs(T, exist_ok=True)

# 用 gen_app 里最后一次生图的输入（正常数值）
INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
paths = []
for f in INS:
    p = os.path.join(OUT, f)
    if not os.path.exists(p):
        p = os.path.join(D + "/gen5", f)
    paths.append(p)
print("输入存在:", [os.path.exists(p) for p in paths])


def spawn():
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()
    return p


def s(p, c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()


def stat(path, tag):
    a = np.fromfile(path, dtype=np.float16).astype(np.float32)
    fin = np.isfinite(a)
    print("%-22s n=%-8d max=%-10.4g std=%-9.4g nan=%-7d inf=%d"
          % (tag, len(a), float(np.abs(a[fin]).max()) if fin.any() else -1,
             float(a[fin].std()) if fin.any() else -1,
             int(np.isnan(a).sum()), int(np.isinf(a).sum())))


# --- A: build + compile ---
p = spawn()
r = s(p, '{"cmd":"build","path":"%s"}' % QG)
print("build:", r[:50])
r = s(p, '{"cmd":"compile"}')
print("compile:", r[:50])
o1 = T + "/o_build.bin"
spec = "%d %s 1 %s" % (len(paths), " ".join("%d %s" % (i, f) for i, f in enumerate(paths)), o1)
r = s(p, '{"cmd":"run","spec":"%s"}' % spec)
print("run(build):", r[:50])
s(p, '{"cmd":"quit"}')
try: p.wait(timeout=60)
except: p.kill()
stat(o1, "build+compile")

# --- B: restore blob ---
p = spawn()
r = s(p, '{"cmd":"rpush","path":"%s"}' % BLOB)
print("rpush:", r[:70])
if '"ok":true' in r:
    o2 = T + "/o_restore.bin"
    spec = "%d %s 1 %s" % (len(paths), " ".join("%d %s" % (i, f) for i, f in enumerate(paths)), o2)
    r = s(p, '{"cmd":"run","spec":"%s"}' % spec)
    print("run(restore):", r[:60])
    s(p, '{"cmd":"quit"}')
    try: p.wait(timeout=60)
    except: p.kill()
    if os.path.exists(o2):
        stat(o2, "restore blob")
        a = np.fromfile(o1, dtype=np.float16).astype(np.float32)
        b = np.fromfile(o2, dtype=np.float16).astype(np.float32)
        n = min(len(a), len(b))
        d = np.abs(a[:n] - b[:n])
        d = d[np.isfinite(d)]
        print("两者最大差 %.4f  (参考量程 %.2f)" % (d.max() if d.size else -1, np.abs(a).max()))
