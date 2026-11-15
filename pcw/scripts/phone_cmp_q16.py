"""W8A16 块图 vs FP16 块图 数值对拍 + 耗时对比。"""
import os, sys, json, struct, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
HOST = D + "/host_new"
QDIR = "/data/local/tmp/q16b00"
FPATH = D + "/e2eper512/b00/g.txt"
T = "/data/local/tmp/cmpq"

os.makedirs(T, exist_ok=True)

# 从转换后的 g.txt 读出每个输入 operand 的 scale
scale_of, ty_of = {}, {}
for ln in open(QDIR + "/g.txt"):
    if not ln.startswith("OPD "):
        continue
    tok = ln.split()
    oid, ty, nd = int(tok[1]), int(tok[2]), int(tok[3])
    scale_of[oid] = float(tok[4 + nd])
    ty_of[oid] = ty

INS = ["i_xb.bin", "i_te.bin", "i_lsh.bin", "i_lsc.bin", "i_lga.bin",
       "i_ctx_pos.bin", "i_cos.bin", "i_sin.bin"]
# 输入 operand 的顺序就是 0..7（图里就是按这个顺序定义的）
q_ins = []
for i, f in enumerate(INS):
    a = np.fromfile(os.path.join(OUT, f), dtype=np.float16).astype(np.float32)
    s = np.float32(scale_of.get(i, 1e-4))
    q = np.clip(np.rint(a / s), -32767, 32767).astype(np.int16)
    p = "%s/in%d.i16" % (T, i)
    q.tofile(p)
    q_ins.append(p)

def spawn():
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()
    return p

def s(p, c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

def spec(ins, out):
    return "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), out)

# ---- FP16 参考 ----
out16 = T + "/o_fp16.bin"
p = spawn()
r = s(p, '{"cmd":"build","path":"%s"}' % FPATH)
assert '"ok":true' in r, r
r = s(p, '{"cmd":"compile"}')
assert '"ok":true' in r, r
t0 = time.time()
r = s(p, '{"cmd":"run","spec":"%s"}' % spec([os.path.join(OUT, f) for f in INS], out16))
dt16 = (time.time() - t0) * 1000
assert '"ok":true' in r, r
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()

# ---- W8A16 ----
outq = T + "/o_q16.bin"
p = spawn()
r = s(p, '{"cmd":"build","path":"%s/g.txt"}' % QDIR)
assert '"ok":true' in r, r
r = s(p, '{"cmd":"compile"}')
assert '"ok":true' in r, r
t0 = time.time()
r = s(p, '{"cmd":"run","spec":"%s"}' % spec(q_ins, outq))
dtq = (time.time() - t0) * 1000
assert '"ok":true' in r, r
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()

ref = np.fromfile(out16, dtype=np.float16).astype(np.float32)
qraw = np.fromfile(outq, dtype=np.int16).astype(np.float32)
# 输出 operand 的 scale
last_op = None
for ln in open(QDIR + "/g.txt"):
    if ln.startswith("OP "):
        last_op = ln
tok = last_op.split()
no = int(tok[3 + int(tok[2])])
out_id = int(tok[4 + int(tok[2])])
so = np.float32(scale_of.get(out_id, 1e-4))
got = qraw * so

n = min(len(ref), len(got))
ref, got = ref[:n], got[:n]
nrmse = np.sqrt(np.mean((got - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-12)
print("FP16 输出   : max=%.4f std=%.4f  (%.0f ms)" % (np.abs(ref).max(), ref.std(), dt16))
print("W8A16 输出  : max=%.4f std=%.4f  (%.0f ms)" % (np.abs(got).max(), got.std(), dtq))
print("NRMSE       : %.4f%%" % (nrmse * 100))
print("速度        : %.2fx" % (dt16 / dtq if dtq else 0))
