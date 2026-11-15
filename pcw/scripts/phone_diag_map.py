"""诊断：why IslandOut 的输出 operand 没进标定表。"""
import os, sys, json
sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import torch, torch.nn as nn, torch.fx as fx
import numpy as np
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
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

low = Lowering("/data/local/tmp/calx2")
shapes = [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (M, CTX), (S, 1, HD), (S, 1, HD)]
args = [low.input(sh) for sh in shapes]
low.lower_module(mod, args)

isl = {k: v for k, v in low.node_ids.items() if "isl" in k.lower()}
print("node_ids 里含 isl 的项: %d" % len(isl))
for k, v in list(isl.items())[:8]:
    print("   %-28s -> operand %s" % (k, v))

# FX 图里岛节点的名字
gm = fx.GraphModule(mod, __import__("lowering")._IslandTracer().trace(mod))
isl_nodes = [n.name for n in gm.graph.nodes
             if n.op == "call_module" and "isl" in str(n.target).lower()]
print("\nFX 图里岛节点: %d" % len(isl_nodes))
for n in isl_nodes[:8]:
    print("   ", n)

# 这些名字是否在 node_ids 里
miss = [n for n in isl_nodes if n not in low.node_ids]
print("\n不在 node_ids 里的岛节点: %d" % len(miss))
for n in miss[:5]:
    print("   ", n)
