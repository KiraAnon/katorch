"""全量生成 14 个块的量化图（带 FP16 岛）。"""
import os, sys, json, shutil, collections
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import torch, torch.nn as nn
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
DST = D + "/e2eper512i"
SCR = "/data/local/tmp/gscr_all"
M, S = 512, 1024

cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]
FOUR = [scalar for scalar in ()]


class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))

    def forward(s, x, e, l1, l2, l3, c, co, si):
        return s.b(s.a(x, e, l1, l2, l3, c, co, si) * s.ps, e, l1, l2, l3, c, co, si)


os.makedirs(DST, exist_ok=True)
for bi in range(14):
    outdir = "%s/b%02d" % (DST, bi)
    if os.path.exists(outdir + "/g.txt"):
        print("b%02d 已存在，跳过" % bi, flush=True)
        continue
    b0, b1 = 2 * bi, 2 * bi + 1
    mod = Pair(BalDiTBlock("net.blocks.%d." % b0, pb[b0]["d_in"], pb[b0]["gs"], S=S, M=M).eval(),
               BalDiTBlock("net.blocks.%d." % b1, pb[b1]["d_in"], pb[b1]["gs"], S=S, M=M).eval(),
               pb[b0]["post_scale"]).eval()
    if os.path.exists(SCR):
        shutil.rmtree(SCR)
    os.makedirs(SCR)
    low = Lowering(SCR)
    shapes = [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (M, CTX), (S, 1, HD), (S, 1, HD)]
    args = [low.input(sh) for sh in shapes]
    low.lower_module(mod, args)
    errs = low.validate()
    txt = low.graph_text()
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(SCR):
        shutil.copy(os.path.join(SCR, f), outdir + "/" + f)
    open(outdir + "/g.txt", "w").write(txt.replace(SCR, outdir))
    c = collections.Counter()
    for ln in txt.splitlines():
        if ln.startswith("OP "):
            c[int(ln.split()[1])] += 1
    print("b%02d  %d 算子  DQ=%d Q=%d  校验=%s"
          % (bi, len(low.ops), c.get(6, 0), c.get(72, 0), errs if errs else "OK"), flush=True)
shutil.rmtree(SCR, ignore_errors=True)
print("全部完成")
