"""测试 restore 路径：导出一个块的 blob，再用新进程恢复并执行，与常规路径对拍。"""
import os, subprocess, time, hashlib

H = "/data/data/com.termux/files/home"
D = H + "/npt"
E2E = D + "/e2eper256"
OUT = D + "/gen5"
CACHE = E2E + "/nc"
HOST = D + "/host_new"
TMP = "/data/local/tmp"

INS = ["i_xb.bin", "i_te.bin", "i_lsh.bin", "i_lsc.bin", "i_lga.bin",
       "i_ctx_pos.bin", "i_cos.bin", "i_sin.bin"]
BLOB = TMP + "/b00.dla"
OA = TMP + "/o_a.bin"
OB = TMP + "/o_b.bin"

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def spawn():
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    return p


def s(p, c):
    a = time.time()
    p.stdin.write(c + "\n"); p.stdin.flush()
    l = p.stdout.readline().strip()
    return l, (time.time() - a) * 1000


def spec(out):
    return "8 " + " ".join("%d %s" % (i, OUT + "/" + f) for i, f in enumerate(INS)) + " 1 " + out


# ---------- 路径 A：常规 build + compile + run，顺便导出 blob ----------
print("=== A: build + compile + export + run ===")
p = spawn()
r, ms = s(p, '{"cmd":"build","path":"%s/b00/g.txt"}' % E2E)
print("  build   %.0f ms  %s" % (ms, r[:50]))
r, ms = s(p, '{"cmd":"compile"}')
print("  compile %.0f ms  %s" % (ms, r[:50]))
r, ms = s(p, '{"cmd":"export","path":"%s"}' % BLOB)
print("  export  %.0f ms  %s" % (ms, r[:70]))
r, ms = s(p, '{"cmd":"run","spec":"%s"}' % spec(OA))
print("  run     %.0f ms  %s" % (ms, r[:50]))
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()

if not os.path.exists(BLOB):
    print("!! blob 没生成，后续跳过")
    raise SystemExit(1)
print("  blob %.1f MB" % (os.path.getsize(BLOB) / 1e6))

# ---------- 路径 B：rpush + run ----------
print("\n=== B: rpush(restore) + run ===")
p = spawn()
t0 = time.time()
r, ms = s(p, '{"cmd":"rpush","path":"%s"}' % BLOB)
print("  rpush   %.0f ms  %s" % (ms, r[:70]))
if '"ok":true' not in r:
    print("!! restore 失败")
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    raise SystemExit(1)
r, ms = s(p, '{"cmd":"run","spec":"%s"}' % spec(OB))
print("  run     %.0f ms  %s" % (ms, r[:50]))
print("  单块合计 %.0f ms" % ((time.time() - t0) * 1000))
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()

# ---------- 对拍 ----------
print("\n=== 对拍 ===")
if os.path.exists(OA) and os.path.exists(OB):
    sa, sb = sha(OA), sha(OB)
    print("  A(build+compile) %s  %d bytes" % (sa, os.path.getsize(OA)))
    print("  B(restore)       %s  %d bytes" % (sb, os.path.getsize(OB)))
    print("  一致: %s" % (sa == sb))
else:
    print("  缺少输出文件 A=%s B=%s" % (os.path.exists(OA), os.path.exists(OB)))
