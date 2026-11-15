"""用干净输入测 NPU 的 Pair（b00）。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, json, subprocess
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn.functional as F
from anima_block import load
from bal_block2 import BalDiTBlock
import torch.nn as nn
from anima_head import PatchEmbed
from t_rope3d import rope3d_cos_sin

D = _KA_WORK
E = D + "/e2eper256"
CLEAN = D + "/clean"
OUT = D + "/pairtest"
os.makedirs(OUT, exist_ok=True)
S, M = 256, 512
cal = json.load(open(D+"/gen/calib6.json"))
pb = cal["per_block"]

# 干净输入
h = torch.from_numpy(np.fromfile(CLEAN+"/h.bin", dtype=np.float16).astype(np.float32)).reshape(S,2048)
h.numpy().astype(np.float16).tofile(OUT+"/i_h.bin")
print("h: max=%.4f" % h.abs().max())

def t_embed(sv):
    h_=1024
    e=-torch.log(torch.tensor(10000.0))*torch.arange(h_,dtype=torch.float32)/h_
    v=torch.tensor([float(sv)])[:,None]*torch.exp(e)[None,:]
    return torch.cat([torch.cos(v),torch.sin(v)],dim=-1)
ctx = torch.from_numpy(np.fromfile(D+"/gen/i_ctx.bin", dtype=np.float16).astype(np.float32)).reshape(M,1024)
te = t_embed(1.0)
W1, W2 = load("net.t_embedder.1.linear_1.weight"), load("net.t_embedder.1.linear_2.weight")
lora = F.linear(F.silu(F.linear(te, W1)), W2)
lsh, lsc, lga = lora[:,:2048], lora[:,2048:4096], lora[:,4096:6144]
cos, sin = rope3d_cos_sin(1, 16, 16)
cos = cos.reshape(S,1,128); sin = sin.reshape(S,1,128)
te.numpy().astype(np.float16).tofile(OUT+"/i_te.bin")
lsh.numpy().astype(np.float16).tofile(OUT+"/i_lsh.bin")
lsc.numpy().astype(np.float16).tofile(OUT+"/i_lsc.bin")
lga.numpy().astype(np.float16).tofile(OUT+"/i_lga.bin")
ctx.numpy().astype(np.float16).tofile(OUT+"/i_ctx.bin")
cos.numpy().astype(np.float16).tofile(OUT+"/i_cos.bin")
sin.numpy().astype(np.float16).tofile(OUT+"/i_sin.bin")

class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__(); s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))
    def forward(s, x, e, l1, l2, l3, c, co, si):
        x = s.a(x, e, l1, l2, l3, c, co, si) * s.ps
        return s.b(x, e, l1, l2, l3, c, co, si)

# torch 参考
with torch.no_grad():
    xb = h / pb[0]["d_in"]
    print("x_bal: max=%.4f" % xb.abs().max())
    xb.numpy().astype(np.float16).tofile(OUT+"/i_xb.bin")
    m = Pair(BalDiTBlock("net.blocks.0.", pb[0]["d_in"], pb[0]["gs"], S=S,M=M).eval(),
             BalDiTBlock("net.blocks.1.", pb[1]["d_in"], pb[1]["gs"], S=S,M=M).eval(),
             pb[0]["post_scale"]).eval()
    ref = m(xb, te, lsh, lsc, lga, ctx, cos, sin)
print("torch Pair: max=%.3f std=%.3f nan=%d inf=%d" % (ref.abs().max(), ref.std(), torch.isnan(ref).sum(), torch.isinf(ref).sum()))
ref.numpy().astype(np.float16).tofile(OUT+"/t_ref.bin")

# NPU
ins = ["i_xb.bin","i_te.bin","i_lsh.bin","i_lsc.bin","i_lga.bin","i_ctx.bin","i_cos.bin","i_sin.bin"]
p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def s(c):
    p.stdin.write(c+"\n"); p.stdin.flush()
    l = p.stdout.readline(); return l.strip() if l else ""
s('{"cmd":"cachedir","path":"%s/nc"}' % E)
s('{"cmd":"cachekey","v":"b00"}')
r = s('{"cmd":"build","path":"%s/b00/g.txt"}' % E)
print("build: %s" % r[:40])
r = s('{"cmd":"compile"}')
print("compile: %s" % r[:40])
if '"ok":true' in r:
    of = OUT+"/o.bin"
    spec = "8 " + " ".join("%d %s" % (i, OUT+"/"+f) for i,f in enumerate(ins)) + " 1 0 " + of
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    print("run: %s" % r[:40])
    if '"ok":true' in r:
        got = np.fromfile(of, dtype=np.float16).astype(np.float32).reshape(ref.shape)
        print("\nNPU   max=%.3f std=%.3f inf=%d" % (np.abs(np.nan_to_num(got)).max(), np.nan_to_num(got).std(), np.isinf(got).sum()))
        print("torch max=%.3f std=%.3f" % (ref.abs().max(), ref.std()))
        er = np.abs(np.nan_to_num(got) - ref.numpy())
        print("误差: MAX_ABS=%.4f  rel=%.4f%%" % (er.max(), 100*er.max()/(ref.abs().max()+1e-12)))
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=300)
except: p.kill()
