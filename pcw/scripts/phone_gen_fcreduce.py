"""用 FC 版归约重新生成 b00（fp16），并和原 fp16 结果对拍。"""
import os, sys, json, shutil
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import torch, torch.nn as nn, numpy as np
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
DST = D + "/e2eper512f"
SCR = "/data/local/tmp/gscr_f"
M = 512
S = 1024

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

if os.path.exists(SCR):
    shutil.rmtree(SCR)
os.makedirs(SCR)
low = Lowering(SCR)
shapes = [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (M, CTX), (S, 1, HD), (S, 1, HD)]
args = [low.input(sh) for sh in shapes]
low.lower_module(mod, args)
errs = low.validate()
print("校验:", errs if errs else "OK")
txt = low.graph_text()
open(SCR + "/g.txt", "w").write(txt)
os.makedirs(DST + "/b00", exist_ok=True)
for f in os.listdir(SCR):
    shutil.copy(os.path.join(SCR, f), DST + "/b00/" + f)
open(DST + "/b00/g.txt", "w").write(txt.replace(SCR, DST + "/b00"))
print("算子数 %d" % len(low.ops))
import collections
c = collections.Counter()
for ln in txt.splitlines():
    if ln.startswith("OP "):
        c[int(ln.split()[1])] += 1
print("算子分布:", dict(sorted(c.items())))
