"""Phase 1: Anima DiT 多层拼接验证。

从 safetensors 按层号加载，用 AnimaBlock 串成 N 层，和 torch 对比。
"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys
import time

sys.path.insert(0, _KA_WORK)
import numpy as np
import torch
import torch.nn as nn

from anima_block import load, has
from npt_blocks import AnimaBlock

HID, HD, NH, CTX = 2048, 128, 16, 1024


def make_block(idx, S, M):
    pfx = "net.blocks.%d." % idx
    w = dict(
        hidden=HID,
        q_proj=load(pfx + "self_attn.q_proj.weight"), k_proj=load(pfx + "self_attn.k_proj.weight"),
        v_proj=load(pfx + "self_attn.v_proj.weight"), o_proj=load(pfx + "self_attn.output_proj.weight"),
        q_norm=load(pfx + "self_attn.q_norm.weight"), k_norm=load(pfx + "self_attn.k_norm.weight"),
        cq_proj=load(pfx + "cross_attn.q_proj.weight"), ck_proj=load(pfx + "cross_attn.k_proj.weight"),
        cv_proj=load(pfx + "cross_attn.v_proj.weight"), co_proj=load(pfx + "cross_attn.output_proj.weight"),
        cq_norm=load(pfx + "cross_attn.q_norm.weight"), ck_norm=load(pfx + "cross_attn.k_norm.weight"),
        mlp_in=load(pfx + "mlp.layer1.weight"), mlp_out=load(pfx + "mlp.layer2.weight"),
        ad_sa_1=load(pfx + "adaln_modulation_self_attn.1.weight"),
        ad_sa_2=load(pfx + "adaln_modulation_self_attn.2.weight"),
        ad_ca_1=load(pfx + "adaln_modulation_cross_attn.1.weight"),
        ad_ca_2=load(pfx + "adaln_modulation_cross_attn.2.weight"),
        ad_mlp_1=load(pfx + "adaln_modulation_mlp.1.weight"),
        ad_mlp_2=load(pfx + "adaln_modulation_mlp.2.weight"),
    )
    return AnimaBlock(w, n_head=NH, head_dim=HD, S=S, M=M)


class DiTRange(nn.Module):
    """N 层串联。层的 forward 签名一致，只有 x 在流动。"""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, emb, lsh, lsc, lga, context, cos, sin):
        for b in self.blocks:
            x = b(x, emb, lsh, lsc, lga, context, cos, sin)
        return x


def build_range(first, n, S, M):
    return DiTRange([make_block(first + i, S, M) for i in range(n)])


if __name__ == "__main__":
    S, M = 4, 4
    print("构建 2 层...", flush=True)
    t0 = time.time()
    mod = build_range(0, 2, S, M).eval()
    print("  %.1fs" % (time.time() - t0), flush=True)

    torch.manual_seed(0)
    x = torch.randn(S, HID) * 0.5
    emb = torch.randn(1, 1, HID) * 0.5
    lsh = torch.randn(1, 1, HID) * 0.5
    lsc = torch.randn(1, 1, HID) * 0.5
    lga = torch.randn(1, 1, HID) * 0.5
    ctx = torch.randn(M, CTX) * 0.5
    cos = torch.randn(S, 1, HD) * 0.1
    sin = torch.randn(S, 1, HD) * 0.1

    with torch.no_grad():
        out = mod(x, emb, lsh, lsc, lga, ctx, cos, sin)
    print("2 层 torch: std=%.4f max=%.2f" % (out.std(), out.abs().max()), flush=True)

    # 逐层对比
    with torch.no_grad():
        h = x
        for i, b in enumerate(mod.blocks):
            h2 = b(h, emb, lsh, lsc, lga, ctx, cos, sin)
            print("  层 %d: in std=%.3f -> out std=%.3f" % (i, h.std(), h2.std()), flush=True)
            h = h2
    print("最终输出一致:", bool(torch.allclose(h, out)))
