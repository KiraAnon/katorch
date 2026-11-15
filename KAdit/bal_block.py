"""Balanced DiT block v2：每块独立 d_in + 3 个 Norm 各自的 g + 层间 2 的幂转换。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, math
sys.path.insert(0, _KA_WORK)
import torch
import torch.nn as nn
from anima_block import Lin, RMS, GELU, AdaLN3, Attn, load
from bal_norm import BalancedNorm


class BalDiTBlock(nn.Module):
    def __init__(self, pfx, d_in, gs, S=4, M=4, n_head=16, head_dim=128, hidden=2048):
        super().__init__()
        self.S, self.M = S, M
        r_sa, r_ca, r_mlp = (d_in / float(g) for g in gs)
        self.ln_sa = BalancedNorm(torch.full((hidden,), r_sa), torch.ones(hidden), kind="ln")
        self.ln_ca = BalancedNorm(torch.full((hidden,), r_ca), torch.ones(hidden), kind="ln")
        self.ln_mlp = BalancedNorm(torch.full((hidden,), r_mlp), torch.ones(hidden), kind="ln")
        inv_d = 1.0 / float(d_in)
        self.register_buffer("inv_d", torch.full((1, hidden), inv_d))
        self.attn_sa = Attn(load(pfx + "self_attn.q_proj.weight"), load(pfx + "self_attn.k_proj.weight"),
                            load(pfx + "self_attn.v_proj.weight"), load(pfx + "self_attn.output_proj.weight"),
                            load(pfx + "self_attn.q_norm.weight"), load(pfx + "self_attn.k_norm.weight"),
                            n_head, head_dim, True, S, S)
        self.attn_ca = Attn(load(pfx + "cross_attn.q_proj.weight"), load(pfx + "cross_attn.k_proj.weight"),
                            load(pfx + "cross_attn.v_proj.weight"), load(pfx + "cross_attn.output_proj.weight"),
                            load(pfx + "cross_attn.q_norm.weight"), load(pfx + "cross_attn.k_norm.weight"),
                            n_head, head_dim, False, S, M)
        self.mlp_in = Lin(load(pfx + "mlp.layer1.weight"))
        self.mlp_out = Lin(load(pfx + "mlp.layer2.weight"))
        self.mlp_act = GELU()
        self.ad_sa = AdaLN3(load(pfx + "adaln_modulation_self_attn.1.weight"), load(pfx + "adaln_modulation_self_attn.2.weight"))
        self.ad_ca = AdaLN3(load(pfx + "adaln_modulation_cross_attn.1.weight"), load(pfx + "adaln_modulation_cross_attn.2.weight"))
        self.ad_mlp = AdaLN3(load(pfx + "adaln_modulation_mlp.1.weight"), load(pfx + "adaln_modulation_mlp.2.weight"))

    def forward(self, x_bal, emb, lora_sh, lora_sc, lora_ga, context, cos, sin):
        sh, sc, ga = self.ad_sa(emb, lora_sh, lora_sc, lora_ga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        n = self.ln_sa(x_bal) * (1.0 + sc) + sh
        # 必须先把 ga*inv_d 合成一个标度再乘：ga 能到 800+，m 也能到上千，
        # 写成 (m * ga) * inv_d 的话中间乘积几十万，fp16 直接溢出。
        x_bal = x_bal + self.attn_sa(n, None, cos, sin) * (ga * self.inv_d)

        sh, sc, ga = self.ad_ca(emb, lora_sh, lora_sc, lora_ga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        n = self.ln_ca(x_bal) * (1.0 + sc) + sh
        x_bal = x_bal + self.attn_ca(n, context, cos, sin) * (ga * self.inv_d)

        sh, sc, ga = self.ad_mlp(emb, lora_sh, lora_sc, lora_ga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        n = self.ln_mlp(x_bal) * (1.0 + sc) + sh
        x_bal = x_bal + self.mlp_out(self.mlp_act(self.mlp_in(n))) * (ga * self.inv_d)
        return x_bal
