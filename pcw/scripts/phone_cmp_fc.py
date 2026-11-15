"""FC 版归约图的 W8A16 vs 原 fp16 对拍。"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
OUT = D + "/gen5"
FP16_REF = D + "/e2eper512/b00/g.txt"     # 原始 fp16（地面真值）
QDIR = "/data/local/tmp/q16i"
T = "/data/local/tmp/cmpf"
os.makedirs(T, exist_ok=True)

scale_of = {}
for ln in open(QDIR + "/g.txt"):
    if ln.startswith("OPD "):
        tok = ln.split()
        scale_of[int(tok[1])] = float(tok[4 + int(tok[3])])

INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
paths = [os.path.join(OUT, f) for f in INS]

q_paths = []
for i, p in enumerate(paths):
    a = np.fromfile(p, dtype=np.float16).astype(np.float32)
    s = np.float32(scale_of.get(i, 0.02))
    q = np.clip(np.rint(a / s), -32767, 32767).astype(np.int16)
    qp = "%s/qin%d.i16" % (T, i)
    q.tofile(qp)
    q_paths.append(qp)


def run(gpath, ins, outp, dt):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gpath)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:50]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:50]
    t0 = time.time()
    spec = "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), outp)
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    dt_ms = (time.time() - t0) * 1000
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:50]
    return np.fromfile(outp, dtype=dt).astype(np.float32), dt_ms


last = None
for ln in open(QDIR + "/g.txt"):
    if ln.startswith("OP "):
        last = ln
tok = last.split()
ni = int(tok[2]); no = int(tok[3 + ni])
out_id = int(tok[4 + ni])
SO = np.float32(scale_of.get(out_id, 0.02))
print("输出 operand %d scale=%.6g" % (out_id, SO))

ref, _ = run(FP16_REF, paths, T + "/of.bin", np.float16)
got, ms = run(QDIR + "/g.txt", q_paths, T + "/oq.bin", np.int16)
if ref is None or got is None:
    print("失败", ref if ref is None else got)
    raise SystemExit(1)
got = got * SO
n = min(len(ref), len(got))
a, b = ref[:n], got[:n]
nrmse = np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((a ** 2).mean()) + 1e-12)
print("FP16 参考 : max=%.4f std=%.4f" % (np.abs(a).max(), a.std()))
print("W8A16     : max=%.4f std=%.4f  (%.0f ms)" % (np.abs(b).max(), b.std(), ms))
print("NRMSE     : %.3f%%" % (nrmse * 100))
