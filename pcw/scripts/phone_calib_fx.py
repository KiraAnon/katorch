"""per-operand 标定：FX node 上跑真实张量，记录每个中间张量的 absmax，映射到 operand id。"""
import os, sys, json
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import torch
import torch.nn as nn
import torch.fx as fx
import numpy as np
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
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


SRC_DIR = os.environ.get("KA_SRC", D + "/e2eper512/b00")
OUT_JSON = os.environ.get("KA_CALIB_OUT", "/data/local/tmp/calib_scale.json")
print("标定源: %s -> %s" % (SRC_DIR, OUT_JSON), flush=True)
print("构建模块 b00 ...", flush=True)
mod = Pair(BalDiTBlock("net.blocks.0.", pb[0]["d_in"], pb[0]["gs"], S=S, M=M).eval(),
           BalDiTBlock("net.blocks.1.", pb[1]["d_in"], pb[1]["gs"], S=S, M=M).eval(),
           pb[0]["post_scale"]).eval()

# --- lowering 拿 node -> operand 映射 ---
low = Lowering("/data/local/tmp/calx")
shapes = [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID),
          (M, CTX), (S, 1, HD), (S, 1, HD)]
args = [low.input(sh) for sh in shapes]
low.lower_module(mod, args)
node_ids = dict(low.node_ids)
print("lowering: %d 算子, %d 个 node 映射" % (len(low.ops), len(node_ids)), flush=True)

# --- 真实张量执行，记录每节点 absmax ---
INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
targs = []
for f, sh in zip(INS, shapes):
    a = np.fromfile(os.path.join(OUT, f), dtype=np.float16).astype(np.float32)
    targs.append(torch.from_numpy(a.reshape(sh)))


class Cap(fx.Interpreter):
    def __init__(self, m):
        super().__init__(m)
        self.rec = {}

    def run_node(self, n):
        r = super().run_node(n)
        try:
            if torch.is_tensor(r):
                v = float(r.abs().max())
                if v == v and v != float("inf"):
                    self.rec[n.name] = v
        except Exception:
            pass
        return r


from lowering import _IslandTracer
gm = fx.GraphModule(mod, _IslandTracer().trace(mod))
print("执行 trace 图（真实张量）...", flush=True)
with torch.no_grad():
    it = Cap(gm)
    it.run(*targs)

print("记录了 %d 个节点的范围" % len(it.rec), flush=True)

tbl, miss = {}, 0
node_extra = getattr(low, "node_extra", {})
for name, oid in node_ids.items():
    v = it.rec.get(name)
    if v is None:
        miss += 1
        continue
    ids = list(node_extra.get(name, [])) or ([oid] if isinstance(oid, int) else [])
    for o in ids:
        tbl[str(o)] = max(v / 32767.0, 1e-8)

print("映射到 operand: %d 个，缺 %d" % (len(tbl), miss))

# 图输入的尺度：直接按数据自身范围算（它们不对应 FX 节点）
for i, t in enumerate(targs):
    mx = float(t.abs().max())
    tbl[str(i)] = max(mx / 32767.0, 1e-8)
    print("  输入 %d: max=%.5f  scale=%.4g" % (i, mx, tbl[str(i)]))

if tbl:
    vals = sorted(tbl.values())
    print("  scale 最小 %.3g  中位 %.3g  最大 %.3g" % (vals[0], vals[len(vals)//2], vals[-1]))
    json.dump(tbl, open(OUT_JSON, "w"))
    print("  -> %s" % OUT_JSON)
