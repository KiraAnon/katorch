"""验证 bload/map/unmap 架构：blob 常驻内存，循环映射，跑完 14 块。"""
import subprocess, time, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
HOST = D + "/host_new"
B = "/data/local/tmp/blobs"
LIST = "/data/local/tmp/blobs/list14.txt"

with open(LIST, "w") as f:
    for i in range(14):
        f.write("%s/b%02d.dla\n" % (B, i))

INS = [OUT + "/i_xb.bin", OUT + "/i_te.bin", OUT + "/i_lsh.bin", OUT + "/i_lsc.bin",
       OUT + "/i_lga.bin", OUT + "/i_ctx_pos.bin", OUT + "/i_cos.bin", OUT + "/i_sin.bin"]
SCRATCH = "/data/local/tmp/scratch2.bin"

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

print("=== bload：把 14 个 blob 读进内存 ===")
t0 = time.time()
r = s('{"cmd":"bload","list":"%s"}' % LIST)
print("  %s   耗时 %.1f s" % (r[:50], time.time() - t0))

def spec(ins, out):
    return "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), out)

print("\n=== 循环 map → run → unmap ===")
t_all = time.time()
t_map = t_run = 0.0
ok = 0
for i in range(14):
    a = time.time()
    r = s('{"cmd":"map","idx":%d}' % i)
    t_map += time.time() - a
    if '"ok":true' not in r:
        print("  b%02d map FAIL %s" % (i, r[:60])); break
    a = time.time()
    r = s('{"cmd":"run","spec":"%s"}' % spec(INS, SCRATCH))
    t_run += time.time() - a
    if '"ok":true' not in r:
        print("  b%02d run FAIL %s" % (i, r[:60])); break
    s('{"cmd":"unmap"}')
    ok += 1

dt = time.time() - t_all
print("\n  完成 %d/14 块   总耗时 %.2f s" % (ok, dt))
print("  其中 map 合计 %.2f s (%.0f ms/块)" % (t_map, t_map / max(ok, 1) * 1000))
print("       run 合计 %.2f s (%.0f ms/块)" % (t_run, t_run / max(ok, 1) * 1000))
print("       其他     %.2f s" % (dt - t_map - t_run))

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()
err = p.stderr.read()
print("\n=== host 端映射耗时 ===")
for line in err.splitlines()[:16]:
    if line.startswith("[MAP]"):
        print(" ", line)
