"""导出量化版 b00 的 blob，放进混合目录（其余仍是 fp16）。"""
import os, subprocess, shutil
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
QG = "/data/local/tmp/q16i/g.txt"
DST = "/data/local/tmp/blobs512mix"
os.makedirs(DST, exist_ok=True)

# 先建立混合目录：其余 15 个从 blobs512m 软链/复制，b00 用新的
SRC = "/data/local/tmp/blobs512m"
for n in os.listdir(SRC):
    if n.startswith("b00"):
        continue
    s = os.path.join(SRC, n)
    d = os.path.join(DST, n)
    if not os.path.exists(d):
        try:
            os.symlink(s, d)
        except OSError:
            shutil.copy(s, d)

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdout.readline()


def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()


r = s('{"cmd":"build","path":"%s"}' % QG)
print("build  :", r[:60])
if '"ok":true' not in r:
    raise SystemExit(1)
r = s('{"cmd":"compile"}')
print("compile:", r[:60])
if '"ok":true' not in r:
    raise SystemExit(1)
out = os.path.join(DST, "b00.dla")
r = s('{"cmd":"export","path":"%s"}' % out)
print("export :", r[:60])
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()
if os.path.exists(out):
    print("  %s  %.1f MB" % (out, os.path.getsize(out) / 1e6))
else:
    print("  导出失败")
    raise SystemExit(1)

tot = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)
          if os.path.isfile(os.path.join(DST, f)))
print("混合目录 %s 共 %.0f MB" % (DST, tot / 1e6))
