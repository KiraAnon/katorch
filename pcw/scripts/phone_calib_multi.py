"""多步标定：在多个 sigma 的输入上分别跑，每个 operand 取所有步的最大值。

用法: KA_IN_DIRS=dir1,dir2,... python calibfx_multi.py
"""
import os, sys, json
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import torch
import torch.nn as nn
import torch.fx as fx
import numpy as np
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering, _IslandTracer
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
M, S = 512, 1024
OUT_JSON = os.environ.get("KA_CALIB_OUT", "/data/local/tmp/calib_i.json")

cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]


class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))

    def forward(s, x, e, l1, l2, l3, c, co, si):
        return s.b(s.a(x, e, l1, l2, l3, c, co, si) * s.ps, e, l1, l2, l3, c, co, si)


BI = int(os.environ.get("KA_BLOCK", "0"))
b0, b1 = 2 * BI, 2 * BI + 1
print("构建块 %d/%d ..." % (b0, b1), flush=True)
mod = Pair(BalDiTBlock("net.blocks.%d." % b0, pb[b0]["d_in"], pb[b0]["gs"], S=S, M=M).eval(),
           BalDiTBlock("net.blocks.%d." % b1, pb[b1]["d_in"], pb[b1]["gs"], S=S, M=M).eval(),
           pb[b0]["post_scale"]).eval()

low = Lowering("/data/local/tmp/calxm")
shapes = [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (M, CTX), (S, 1, HD), (S, 1, HD)]
args = [low.input(sh) for sh in shapes]
low.lower_module(mod, args)
node_ids = dict(low.node_ids)
node_extra = getattr(low, "node_extra", {})
print("lowering: %d 算子, %d node 映射" % (len(low.ops), len(node_ids)), flush=True)

gm = fx.GraphModule(mod, _IslandTracer().trace(mod))


class Cap(fx.Interpreter):
    def __init__(self, m):
        super().__init__(m)
        self.rec = {}

    def run_node(self, n):
        r = super().run_node(n)
        try:
            if torch.is_tensor(r) and r.numel() > 0:
                a = r.abs().flatten().float()
                mx = float(a.max())
                if mx == mx and mx != float("inf"):
                    self.rec[n.name] = mx
        except Exception:
            pass
        return r


INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]

dirs = [d for d in os.environ.get("KA_IN_DIRS", D + "/gen5").split(",") if d]
print("标定输入目录: %s" % dirs, flush=True)

worst = {}          # node name -> 跨所有输入的最大值
input_max = [0.0] * 8
used = 0
for di, d in enumerate(dirs):
    targs = []
    ok = True
    for f, sh in zip(INS, shapes):
        fp = os.path.join(d, f)
        if not os.path.exists(fp):
            ok = False
            break
        a = np.fromfile(fp, dtype=np.float16).astype(np.float32)
        targs.append(torch.from_numpy(a.reshape(sh)))
    if not ok:
        print("  跳过 %s（文件不全）" % d, flush=True)
        continue
    with torch.no_grad():
        it = Cap(gm)
        it.run(*targs)
    used += 1
    for k, v in it.rec.items():
        if v > worst.get(k, 0.0):
            worst[k] = v
    for i in range(8):
        input_max[i] = max(input_max[i], float(targs[i].abs().max()))
    print("  [%d/%d] %s  -> %d 个节点" % (di + 1, len(dirs), os.path.basename(d), len(it.rec)), flush=True)

print("共用了 %d 组输入" % used, flush=True)

tbl = {}
miss = 0
for name, oid in node_ids.items():
    v = worst.get(name)
    if v is None:
        miss += 1
        continue
    ids = list(node_extra.get(name, [])) or ([oid] if isinstance(oid, int) else [])
    for o in ids:
        tbl[str(o)] = max(v / 32767.0, 1e-8)

for i in range(8):
    tbl[str(i)] = max(input_max[i] / 32767.0, 1e-8)
    print("  输入 %d: max=%.5f  scale=%.4g" % (i, input_max[i], tbl[str(i)]))

print("映射到 operand: %d 个，缺 %d" % (len(tbl), miss))
if tbl:
    vals = sorted(tbl.values())
    print("  scale 最小 %.3g  中位 %.3g  最大 %.3g" % (vals[0], vals[len(vals) // 2], vals[-1]))
    json.dump(tbl, open(OUT_JSON, "w"))
    print("  -> %s" % OUT_JSON)
