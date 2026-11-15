"""Anima DiT 首尾：PatchEmbed / TimestepEmbedding / FinalLayer。

设计约束（都踩过）：
  - SLICE 不可用 → adaln 的 chunk 用权重切分实现（和 AdaLN 里 n=3 一样）
  - lora 的切片在 CPU 侧预切好传入
  - 单边广播 [S,D] * [1,D] 可用（Block 已验证）
"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys
sys.path.insert(0, _KA_WORK)
import torch
import torch.nn as nn

from npt_layers import Lin, Norm


class PatchEmbed(nn.Module):
    """重排在 CPU 做，NPU 只做 Linear(68 -> 2048)。"""

    def __init__(self, w):
        super().__init__()
        self.proj = Lin(w)

    def forward(self, x):
        return self.proj(x)


class TimestepEmbedding(nn.Module):
    """sin/cos 由 CPU 算好传入 [1, 2048]，输出 adaln_lora [1, 6144]。"""

    def __init__(self, w1, w2):
        super().__init__()
        self.l1 = Lin(w1)
        self.l2 = Lin(w2)

    def forward(self, t_emb):
        h = self.l1(t_emb)
        h = torch.nn.functional.silu(h)
        return self.l2(h)


class FinalLayer(nn.Module):
    """LayerNorm + adaln(2 路) + Linear(2048 -> 64)。

    原实现 adaln_modulation 输出 4096 再 chunk(2)，这里改成两个独立 Linear，
    权重按行切开，图里没有切片。
    lora 的 4096 段由 CPU 切成 shift/scale 两半传入。
    """

    def __init__(self, hidden, w_l1, w_l2, w_lin, n_chunks=2):
        super().__init__()
        # K 要足够大：sum((x/K)^2) over 2048 维不能超过 fp16 上限 65504
        # 输入 std 可达 1200+（28 层后），var=1.4e6，需要 K > 209，取 512
        self.ln = Norm(None, kind="ln", dim=hidden, K=512.0)
        self.ad_l1 = Lin(w_l1)
        D = w_l2.shape[0] // n_chunks
        self.ad_parts = nn.ModuleList([Lin(w_l2[i * D:(i + 1) * D]) for i in range(n_chunks)])
        self.lin = Lin(w_lin)
        self.D = D

    def forward(self, x, emb, lora_sh, lora_sc):
        h = torch.nn.functional.silu(emb)
        h = self.ad_l1(h)
        sh = self.ad_parts[0](h) + lora_sh      # [1, 2048]
        sc = self.ad_parts[1](h) + lora_sc
        y = self.ln(x) * (1.0 + sc) + sh
        return self.lin(y)


# ---------- 参考实现（纯 torch，用于对拍）----------
def ref_timestep_embedding(t_emb, w1, w2):
    h = torch.nn.functional.linear(t_emb, w1)
    h = torch.nn.functional.silu(h)
    return torch.nn.functional.linear(h, w2)


def ref_final_layer(x, emb, lora, w_l1, w_l2, w_lin, eps=1e-6):
    h = torch.nn.functional.silu(emb)
    h = torch.nn.functional.linear(h, w_l1)
    out = torch.nn.functional.linear(h, w_l2)          # [1, 4096]
    out = out + lora[:4096].unsqueeze(0)               # lora [1,6144] 的前 4096
    sh, sc = out.chunk(2, dim=-1)
    m = x.mean(-1, keepdim=True)
    d = x - m
    v = (d * d).mean(-1, keepdim=True)
    y = d / torch.sqrt(v + eps) * (1.0 + sc) + sh
    return torch.nn.functional.linear(y, w_lin)
