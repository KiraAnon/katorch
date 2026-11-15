"""测量模型里 softmax 输入的取值范围（决定 clamp 上限）。"""
import os, sys, json
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from anima_block import load
from bal_block2 import BalDiTBlock
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
M, S = 512, 1024
cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]


class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))

    def forward(s, x, e, l1, l2, l3, c, co, si):
        return s.b(s.a(x, e, l1, l2, l3, c, co, si) * s.ps, e, l1, l2, l3, c, co, si)


mod = Pair(BalDiTBlock("net.blocks.0.", pb[0]["d_in"], pb[0]["gs"], S=S, M=M).eval(),
           BalDiTBlock("net.blocks.1.", pb[1]["d_in"], pb[1]["gs"], S=S, M=M).eval(),
           pb[0]["post_scale"]).eval()

rec = []
_real = F.softmax
def spy(x, dim=-1, **kw):
    rec.append((float(x.max()), float(x.min()), tuple(x.shape)))
    return _real(x, dim=dim, **kw)
F.softmax = spy
_ts = torch.softmax
def spy2(x, dim=-1, **kw):
    rec.append((float(x.max()), float(x.min()), tuple(x.shape)))
    return _ts(x, dim=dim, **kw)
torch.softmax = spy2
_tt = torch.Tensor.softmax
def spy3(self, dim=-1, **kw):
    rec.append((float(self.max()), float(self.min()), tuple(self.shape)))
    return _tt(self, dim=dim, **kw)
torch.Tensor.softmax = spy3

INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
shapes = [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (M, CTX), (S, 1, HD), (S, 1, HD)]
args = [torch.from_numpy(np.fromfile(os.path.join(OUT, f), dtype=np.float16).astype(np.float32).reshape(sh))
        for f, sh in zip(INS, shapes)]

with torch.no_grad():
    mod(*args)

print("softmax 调用次数: %d" % len(rec))
if rec:
    mx = max(r[0] for r in rec)
    mn = min(r[1] for r in rec)
    print("输入最大值 %.3f  最小值 %.3f  最大跨度 %.3f" % (mx, mn, mx - mn))
    for r in rec[:6]:
        print("   shape=%s max=%.2f min=%.2f span=%.2f" % (r[2], r[0], r[1], r[0] - r[1]))
