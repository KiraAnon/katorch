"""用真实运行时的输入，对拍 b00（2层）的 NPU 输出与 PyTorch 参考。"""
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")

import sys, os, json, subprocess
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from anima_block import load
from bal_block2 import BalDiTBlock

H = _KA_HOME
D = _KA_WORK
OUT = D + "/gen5"
E2E = D + "/e2eper256"
S, M, DIM = 256, 512, 2048

cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]

def rd(name):
    return torch.from_numpy(np.fromfile(OUT + "/" + name, dtype=np.float16).astype(np.float32))

xb  = rd("i_xb.bin").reshape(S, DIM)
te  = rd("i_te.bin").reshape(1, DIM)
lsh = rd("i_lsh.bin").reshape(1, DIM)
lsc = rd("i_lsc.bin").reshape(1, DIM)
lga = rd("i_lga.bin").reshape(1, DIM)
ctx = rd("i_ctx_pos.bin").reshape(M, 1024)
cos = rd("i_cos.bin").reshape(S, 1, 128)
sin = rd("i_sin.bin").reshape(S, 1, 128)
print("输入: xb std=%.4f  ctx std=%.4f  te absmax=%.2f" % (xb.std(), ctx.std(), te.abs().max()))

class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))
    def forward(s, x, e, l1, l2, l3, c, co, si):
        x = s.a(x, e, l1, l2, l3, c, co, si) * s.ps
        return s.b(x, e, l1, l2, l3, c, co, si)

b0 = BalDiTBlock("net.blocks.0.", pb[0]["d_in"], pb[0]["gs"], S=S, M=M).eval()
b1 = BalDiTBlock("net.blocks.1.", pb[1]["d_in"], pb[1]["gs"], S=S, M=M).eval()
m = Pair(b0, b1, pb[0]["post_scale"]).eval()

with torch.no_grad():
    ref = m(xb, te, lsh, lsc, lga, ctx, cos, sin)
print("PyTorch 输出: max=%.4f  std=%.4f" % (ref.abs().max(), ref.std()))

# NPU：复用已存在的 b00 图和缓存
ins = ["i_xb.bin", "i_te.bin", "i_lsh.bin", "i_lsc.bin", "i_lga.bin", "i_ctx_pos.bin", "i_cos.bin", "i_sin.bin"]
of = OUT + "/o_verify_b00.bin"
p = subprocess.Popen([D + "/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def cmd(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    l = p.stdout.readline(); return l.strip() if l else ""

cmd('{"cmd":"cachedir","path":"%s/nc"}' % E2E)
cmd('{"cmd":"cachekey","v":"b00"}')
r = cmd('{"cmd":"build","path":"%s/b00/g.txt"}' % E2E)
print("build:", r[:50])
r = cmd('{"cmd":"compile"}')
print("compile:", r[:50])
if '"ok":true' in r:
    spec = "8 " + " ".join("%d %s" % (i, OUT + "/" + f) for i, f in enumerate(ins)) + " 1 " + of
    r = cmd('{"cmd":"run","spec":"%s"}' % spec)
    print("run:", r[:50])
    if '"ok":true' in r:
        got = np.fromfile(of, dtype=np.float16).astype(np.float32).reshape(ref.shape)
        er = np.abs(got - ref.numpy())
        print("\nNPU      max=%.4f std=%.4f" % (np.abs(got).max(), got.std()))
        print("PyTorch  max=%.4f std=%.4f" % (ref.abs().max(), ref.std()))
        print("误差 MAX_ABS=%.4f  相对=%.4f%%" % (er.max(), 100 * er.max() / (ref.abs().max() + 1e-12)))
        print("逐通道标准差比 NPU/torch = %.4f" % (got.std() / (ref.std() + 1e-12)))
        print("前5 torch:", [round(float(v), 3) for v in ref[0, :5]])
        print("前5 NPU  :", [round(float(v), 3) for v in got[0, :5]])
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=200)
except: p.kill()
