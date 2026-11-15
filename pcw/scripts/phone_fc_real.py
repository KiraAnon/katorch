"""用真实块权重做 FC 隔离验证：fp16 vs W8A16。"""
import os, sys, json, struct, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
W = D + "/e2eper256/b00/c4.bin"     # 256x2048 权重
B = D + "/e2eper256/b00/c5.bin"     # 256 bias
T = "/data/local/tmp/fcreal"
os.makedirs(T, exist_ok=True)

wt = np.fromfile(W, dtype=np.float16).astype(np.float32).reshape(256, 2048)
bs = np.fromfile(B, dtype=np.float16).astype(np.float32)
print("权重 %s  max=%.4f std=%.4f" % (wt.shape, np.abs(wt).max(), wt.std()))
print("bias %s  max=%.4f" % (bs.shape, np.abs(bs).max()))

rng = np.random.default_rng(7)
x = (rng.standard_normal((1, 2048)) * 0.05).astype(np.float32)
print("输入 max=%.4f std=%.4f" % (np.abs(x).max(), x.std()))

SX = float(np.abs(x).max() / 32767.0)
torch_ref_pre = (x @ wt.T + bs)[0]
SY = float(np.abs(torch_ref_pre).max() / 32767.0)

def mkfp16():
    x.astype(np.float16).tofile(T + "/x.bin")
    wt.astype(np.float16).tofile(T + "/w.bin")
    bs.astype(np.float16).tofile(T + "/b.bin")
    g = ["OPD 0 8 2 1 2048 0 0 i",
         "OPD 1 8 2 256 2048 0 0 w %s/w.bin" % T,
         "OPD 2 8 1 256 0 0 w %s/b.bin" % T,
         "OPD 3 1 1 1 0 0 a 0",
         "OPD 4 8 2 1 256 0 0 o",
         "IO 1 0 1 4", "OP 9 4 0 1 2 3 1 4", "END"]
    open(T + "/gf.txt", "w").write("\n".join(g) + "\n")

def mkw8():
    mx = np.abs(wt).max(axis=1)
    sw = np.where(mx > 0, mx / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(wt / sw[:, None]), -127, 127).astype(np.int8)
    q.tofile(T + "/w8.bin"); sw.tofile(T + "/w8.sc")
    qx = np.clip(np.rint(x / SX), -32767, 32767).astype(np.int16)
    qx.tofile(T + "/x16.bin")
    qb = np.clip(np.rint(bs / (SX * sw)), -2**31, 2**31-1).astype(np.int32)
    qb.tofile(T + "/b32.bin")
    g = ["OPD 0 7 2 1 2048 %.9g 0 i" % SX,
         "OPD 1 11 2 256 2048 0 0 w %s/w8.bin" % T,
         "QCH 1 0 256 %s/w8.sc" % T,
         "OPD 2 4 1 256 0 0 w %s/b32.bin" % T,
         "OPD 3 1 1 1 0 0 a 0",
         "OPD 4 7 2 1 256 %.9g 0 o" % SY,
         "IO 1 0 1 4", "OP 9 4 0 1 2 3 1 4", "END"]
    open(T + "/gq.txt", "w").write("\n".join(g) + "\n")

def run(gpath, ing, outg, dtype):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()
    r = s('{"cmd":"build","path":"%s"}' % gpath)
    assert '"ok":true' in r, r
    r = s('{"cmd":"compile"}')
    assert '"ok":true' in r, r
    out = T + "/" + outg
    r = s('{"cmd":"run","spec":"1 0 %s 1 0 %s"}' % (ing, out))
    assert '"ok":true' in r, r
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    return np.fromfile(out, dtype=dtype).astype(np.float32)

mkfp16(); mkw8()
ref = run(T + "/gf.txt", T + "/x.bin", "of.bin", np.float16)
q = run(T + "/gq.txt", T + "/x16.bin", "oq.bin", np.int16)
got = q * SY

torch_ref = (x @ wt.T + bs)[0]
print("\nPyTorch 期望 : max=%.4f" % np.abs(torch_ref).max())
print("NPU fp16     : max=%.4f  NRMSE(vs torch)=%.3f%%"
      % (np.abs(ref).max(), np.sqrt(((ref-torch_ref)**2).mean())/np.sqrt((torch_ref**2).mean())*100))
print("NPU W8A16    : max=%.4f  NRMSE(vs torch)=%.3f%%"
      % (np.abs(got).max(), np.sqrt(((got-torch_ref)**2).mean())/np.sqrt((torch_ref**2).mean())*100))
