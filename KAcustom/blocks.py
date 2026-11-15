"""用 npt_layers 组装 Qwen3 / Anima 的 block。

模型接入只剩「哪层接哪层」，算子细节全在层库里。
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

from npt_layers import Lin, Norm, MLP, Attention, AdaLN


class QwenBlock(nn.Module):
    """Qwen3 一层：RMSNorm → GQA attention（RoPE + causal）→ residual → RMSNorm → SwiGLU。"""

    def __init__(self, w, n_head, n_kv, head_dim, S):
        super().__init__()
        self.S = S
        self.ln1 = Norm(w["ln1"], kind="rms")
        rep = n_head // n_kv
        # k/v 的权重提前按 head 重复成 n_head 个（GQA 在图里就不需要 expand）
        kw = w["k_proj"].reshape(n_kv, head_dim, -1).repeat_interleave(rep, 0).reshape(n_head * head_dim, -1)
        vw = w["v_proj"].reshape(n_kv, head_dim, -1).repeat_interleave(rep, 0).reshape(n_head * head_dim, -1)
        self.attn = Attention(w["q_proj"], kw, vw, w["o_proj"], n_head, head_dim, S,
                              qn=w["q_norm"], kn=w["k_norm"], rope=True, causal=True)
        self.ln2 = Norm(w["ln2"], kind="rms")
        self.mlp = MLP("swiglu", w["up_proj"], w["down_proj"], w_gate=w["gate_proj"])

    def forward(self, x, mask, cos, sin):
        x = x + self.attn(self.ln1(x), None, cos, sin, mask)
        x = x + self.mlp(self.ln2(x))
        return x


class AnimaBlock(nn.Module):
    """Cosmos DiT 一层：三次 adaln 门控残差（self-attn / cross-attn / mlp）。"""

    def __init__(self, w, n_head, head_dim, S, M):
        super().__init__()
        self.ln_sa = Norm(None, kind="ln", dim=w["hidden"], K=64.0)
        self.ln_ca = Norm(None, kind="ln", dim=w["hidden"], K=64.0)
        self.ln_mlp = Norm(None, kind="ln", dim=w["hidden"], K=64.0)
        self.attn_sa = Attention(w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
                                n_head, head_dim, S, qn=w["q_norm"], kn=w["k_norm"], rope=True)
        self.attn_ca = Attention(w["cq_proj"], w["ck_proj"], w["cv_proj"], w["co_proj"],
                                n_head, head_dim, S, M, qn=w["cq_norm"], kn=w["ck_norm"], rope=False)
        self.mlp = MLP("gelu", w["mlp_in"], w["mlp_out"])
        self.ad_sa = AdaLN(w["ad_sa_1"], w["ad_sa_2"], n=3)
        self.ad_ca = AdaLN(w["ad_ca_1"], w["ad_ca_2"], n=3)
        self.ad_mlp = AdaLN(w["ad_mlp_1"], w["ad_mlp_2"], n=3)

    def forward(self, x, emb, lsh, lsc, lga, context, cos, sin):
        sh, sc, ga = self.ad_sa(emb, lsh, lsc, lga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        x = x + ga * self.attn_sa(self.ln_sa(x) * (1.0 + sc) + sh, None, cos, sin)

        sh, sc, ga = self.ad_ca(emb, lsh, lsc, lga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        x = x + ga * self.attn_ca(self.ln_ca(x) * (1.0 + sc) + sh, context, cos, sin)

        sh, sc, ga = self.ad_mlp(emb, lsh, lsc, lga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        x = x + ga * self.mlp(self.ln_mlp(x) * (1.0 + sc) + sh)
        return x
