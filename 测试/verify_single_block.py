"""直接测 BalDiTBlock 单层 vs 探针版。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, json, subprocess, shutil
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn.functional as F
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering
from t_rope3d import rope3d_cos_sin

H = _KA_HOME
D = H + "/npt"
OUT = D + "/prb4"
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
S, M, DIM = 256, 512, 2048
cal = json.load(open(D+"/gen/calib6.json"))
pb = cal["per_block"][0]

def t_embed(sv):
    h=1024
    e=-torch.log(torch.tensor(10000.0))*torch.arange(h,dtype=torch.float32)/h
    v=torch.tensor([float(sv)])[:,None]*torch.exp(e)[None,:]
    return torch.cat([torch.cos(v),torch.sin(v)],dim=-1)
ctx = torch.from_numpy(np.fromfile(D+"/gen/i_ctx.bin", dtype=np.float16).astype(np.float32)).reshape(M,1024)
te = t_embed(1.0)
W1, W2 = load("net.t_embedder.1.linear_1.weight"), load("net.t_embedder.1.linear_2.weight")
lora = F.linear(F.silu(F.linear(te, W1)), W2)
lsh, lsc, lga = lora[:,:2048], lora[:,2048:4096], lora[:,4096:6144]
cos, sin = rope3d_cos_sin(1, 16, 16)
cos = cos.reshape(S,1,128); sin = sin.reshape(S,1,128)
xb = torch.from_numpy(np.fromfile(D+"/gen5/i_xb.bin", dtype=np.float16).astype(np.float32)).reshape(S,2048)

b0 = BalDiTBlock("net.blocks.0.", pb["d_in"], pb["gs"], S=S, M=M).eval()
with torch.no_grad():
    ref = b0(xb, te, lsh, lsc, lga, ctx, cos, sin)
print("BalDiTBlock torch 输出: max=%.4f std=%.4f" % (ref.abs().max(), ref.std()))

for nm, a in [("in_xb",xb),("in_te",te),("in_lsh",lsh),("in_lsc",lsc),("in_lga",lga),("in_ctx",ctx),("in_cos",cos),("in_sin",sin)]:
    a.numpy().astype(np.float16).tofile(OUT+"/%s.bin" % nm)

ins_shape = [(S,DIM),(1,1,DIM),(1,1,DIM),(1,1,DIM),(1,1,DIM),(M,1024),(S,1,128),(S,1,128)]
ins_files = ["in_xb.bin","in_te.bin","in_lsh.bin","in_lsc.bin","in_lga.bin","in_ctx.bin","in_cos.bin","in_sin.bin"]

d = OUT + "/g"
os.makedirs(d, exist_ok=True)
low = Lowering(d)
args = [low.input(sh) for sh in ins_shape]
low.lower_module(b0, args)
open(d+"/g.txt","w").write(low.graph_text())
errs = low.validate()
print("图: %d 算子 校验 %s" % (len(low.ops), "通过" if not errs else errs[:1]))

p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def s(c):
    p.stdin.write(c+"\n"); p.stdin.flush()
    l = p.stdout.readline(); return l.strip() if l else ""
s('{"cmd":"cachedir","path":"%s/nc"}' % OUT)
s('{"cmd":"cachekey","v":"b0direct"}')
r = s('{"cmd":"build","path":"%s/g.txt"}' % d)
print("build: %s" % r[:40])
r = s('{"cmd":"compile"}')
print("compile: %s" % r[:40])
if '"ok":true' in r:
    of = OUT+"/o.bin"
    spec = "8 " + " ".join("%d %s" % (i, OUT+"/"+f) for i,f in enumerate(ins_files)) + " 1 0 " + of
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    print("run: %s" % r[:40])
    if '"ok":true' in r:
        got = np.fromfile(of, dtype=np.float16).astype(np.float32).reshape(ref.shape)
        er = np.abs(got - ref.numpy())
        print("\nNPU   max=%.4f std=%.4f" % (np.abs(got).max(), got.std()))
        print("torch max=%.4f std=%.4f" % (ref.abs().max(), ref.std()))
        print("误差: MAX_ABS=%.4f  rel=%.4f%%" % (er.max(), 100*er.max()/(ref.abs().max()+1e-12)))
        # 前 5 个
        print("前5 torch: %s" % [round(float(v),3) for v in ref[0,:5]])
        print("前5 NPU:   %s" % [round(float(v),3) for v in got[0,:5]])
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=200)
except: p.kill()
