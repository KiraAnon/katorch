"""28 层：BalDiTBlock vs 手算，看哪个 NaN。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, json
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn.functional as F
from anima_block import load
from bal_block2 import BalDiTBlock
from anima_head import PatchEmbed
from t_rope3d import rope3d_cos_sin

D = _KA_WORK
S, M = 256, 512
LH = S_H = 16
cal = json.load(open(D+"/gen/calib6.json"))
pb = cal["per_block"]

torch.manual_seed(42)
x = torch.randn(1,16,32,32)
pp_ = torch.zeros(S,68); ll = x[0].permute(1,2,0)
for i in range(16):
    for j in range(16):
        pp_[i*16+j,:64] = ll[i*2:(i+1)*2, j*2:(j+1)*2, :].reshape(-1)
pe = PatchEmbed(load("net.x_embedder.proj.1.weight")).eval()

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

with torch.no_grad():
    h0 = pe(pp_)
print("h0: max=%.4f" % h0.abs().max())

print("\n=== BalDiTBlock 28 层 ===")
xb = h0 / pb[0]["d_in"]
for bi in range(28):
    b = BalDiTBlock("net.blocks.%d." % bi, pb[bi]["d_in"], pb[bi]["gs"], S=S, M=M).eval()
    with torch.no_grad():
        xb = b(xb, te, lsh, lsc, lga, ctx, cos, sin) * pb[bi]["post_scale"]
    if bi < 3 or bi % 9 == 8:
        nan = torch.isnan(xb).sum().item(); inf = torch.isinf(xb).sum().item()
        print("  块%02d: max=%12.3f std=%10.3f nan=%d inf=%d" % (bi, torch.nan_to_num(xb).abs().max(), torch.nan_to_num(xb).std(), nan, inf))
    if torch.isnan(xb).any() or torch.isinf(xb).any():
        print("  ^^^ 块%d 爆了" % bi)
        # 逐步查块 0
        print("\n  重跑块 0 逐步:")
        x0 = h0 / pb[0]["d_in"]
        b0 = BalDiTBlock("net.blocks.0.", pb[0]["d_in"], pb[0]["gs"], S=S, M=M).eval()
        with torch.no_grad():
            n = b0.ln_sa(x0)
            print("    ln_sa: max=%.3f nan=%d" % (n.abs().max(), torch.isnan(n).sum()))
        break
