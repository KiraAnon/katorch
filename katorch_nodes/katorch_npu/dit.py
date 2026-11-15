"""DiT 前向：latent → 速度场，全部在 NPU 上跑（14 块逐块映射）。

数值路径与已验证的 `npt/app/gen.py` 完全一致：
  patch 化 [S,68] → pe → 逐块（每块后乘 post_scale）→ fl → 反 patch 化成 latent

和 gen.py 的区别只有「块从哪个目录来」——这里按资产 manifest 里的文件清单走，
所以同一份代码能跑不同分辨率/不同块数的编译产物。
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .st import SafeTensorFile
from .rope import rope3d_cos_sin

HOME = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
WORK = os.environ.get("KA_WORK", os.path.join(HOME, "npt"))
TMP = os.environ.get("KA_TMP", os.path.join(WORK, "gen_app"))


def _to_fp16_file(t, path):
    t.contiguous().numpy().astype(np.float16).tofile(path)


def _read_fp16(path, shape=None):
    a = np.fromfile(path, dtype=np.float16).astype(np.float32)
    if shape is not None:
        return torch.from_numpy(a).reshape(*shape)
    return torch.from_numpy(a)


class DitRuntime(object):
    """一份编译好的 DiT 图 + 它的时间嵌入/位置编码。"""

    def __init__(self, asset, host, model_path=None, lora=None, lora_strength=1.0, tmp_dir=None):
        self.asset = asset
        self.host = host
        self.tmp = tmp_dir or TMP
        os.makedirs(self.tmp, exist_ok=True)

        self.width = int(asset["width"])
        self.height = int(asset["height"])
        self.n_blocks = int(asset["blocks"])
        self.LH, self.LW = self.height // 8, self.width // 8     # latent 尺寸
        self.S_H, self.S_W = self.LH // 2, self.LW // 2
        self.S = self.S_H * self.S_W
        self.order = list(asset["order"])

        calib = asset.get("calib")
        if not calib or not os.path.isfile(calib):
            raise RuntimeError("资产 %s 缺少校准文件（per_block post_scale 必需）" % asset.get("id"))
        with open(calib, "r", encoding="utf-8") as f:
            self.cal = json.load(f)
        self.pb = self.cal["per_block"]
        if len(self.pb) < 2 * self.n_blocks:
            raise RuntimeError("校准文件与块数不匹配：%s 有 %d 项，%d 块需要 %d 项"
                               % (calib, len(self.pb), self.n_blocks, 2 * self.n_blocks))

        # ---- 时间嵌入 / adaLN 权重（只读 3 个张量） ----
        model_path = model_path or self._find_model()
        if not model_path or not os.path.isfile(model_path):
            raise RuntimeError("找不到 DiT 权重文件（%s），可用 KA_ANIMA 指定" % model_path)
        self.model_path = model_path
        st = SafeTensorFile(model_path)
        self.W1 = self._w(st, "net.t_embedder.1.linear_1.weight", lora, lora_strength)
        self.W2 = self._w(st, "net.t_embedder.1.linear_2.weight", lora, lora_strength)
        self.TNORM = st.get("net.t_embedding_norm.weight").float()
        del st

        # ---- RoPE 预算（只与分辨率有关，一次算好） ----
        self.f_cos = os.path.join(self.tmp, "i_cos.bin")
        self.f_sin = os.path.join(self.tmp, "i_sin.bin")
        if not (os.path.isfile(self.f_cos) and os.path.isfile(self.f_sin)):
            cos, sin = rope3d_cos_sin(1, self.S_H, self.S_W)
            cos.reshape(self.S, 1, 128).numpy().astype(np.float16).tofile(self.f_cos)
            sin.reshape(self.S, 1, 128).numpy().astype(np.float16).tofile(self.f_sin)
        self._rope_ok = self._check_rope()

        # ---- 每步复用的临时文件 ----
        self.f_patch = os.path.join(self.tmp, "i_patch.bin")
        self.f_h = os.path.join(self.tmp, "o_h.bin")
        self.f_te = os.path.join(self.tmp, "i_te.bin")
        self.f_lsh = os.path.join(self.tmp, "i_lsh.bin")
        self.f_lsc = os.path.join(self.tmp, "i_lsc.bin")
        self.f_lga = os.path.join(self.tmp, "i_lga.bin")
        self.f_tok = os.path.join(self.tmp, "o_tok.bin")

    # ---------- 初始化 ----------
    @staticmethod
    def _w(st, name, lora, strength):
        w = st.get(name).float()
        if lora is not None:
            try:
                d = lora.delta(name, strength)
            except Exception:
                d = None
            if d is not None and tuple(d.shape) == tuple(w.shape):
                w = w + d
        return w

    def _find_model(self):
        model = self.asset.get("model", "anima_v1")
        cands = [
            os.environ.get("KA_ANIMA"),
            os.path.join(HOME, "ComfyUI", "models", "diffusion_models", model + ".safetensors"),
            os.path.join(HOME, "ComfyUI", "models", "diffusion_models", "anima_v1.safetensors"),
        ]
        for c in cands:
            if c and os.path.isfile(c):
                return c
        return None

    def _check_rope(self):
        """校验缓存下来的 cos/sin 和当前分辨率是否匹配，不匹配就重算。"""
        want = self.S * 128 * 2
        try:
            if os.path.getsize(self.f_cos) != want or os.path.getsize(self.f_sin) != want:
                raise ValueError("size")
            return True
        except Exception:
            cos, sin = rope3d_cos_sin(1, self.S_H, self.S_W)
            cos.reshape(self.S, 1, 128).numpy().astype(np.float16).tofile(self.f_cos)
            sin.reshape(self.S, 1, 128).numpy().astype(np.float16).tofile(self.f_sin)
            return True

    # ---------- 时间嵌入 ----------
    def t_embed(self, sigma):
        """[1,2048]：fourier 时间嵌入。"""
        h = 1024
        e = -torch.log(torch.tensor(10000.0)) * torch.arange(h, dtype=torch.float32) / h
        v = torch.tensor([float(sigma)])[:, None] * torch.exp(e)[None, :]
        return torch.cat([torch.cos(v), torch.sin(v)], dim=-1)

    def write_cond(self, sigma):
        """把这一步的 adaLN 条件写盘（只依赖 sigma）。返回 4 个文件路径。"""
        raw = self.t_embed(sigma)
        lora = F.linear(F.silu(F.linear(raw, self.W1)), self.W2)
        te = raw * torch.rsqrt((raw * raw).mean(-1, keepdim=True) + 1e-6) * self.TNORM
        _to_fp16_file(te, self.f_te)
        _to_fp16_file(lora[:, :2048], self.f_lsh)
        _to_fp16_file(lora[:, 2048:4096], self.f_lsc)
        _to_fp16_file(lora[:, 4096:6144], self.f_lga)
        return self.f_te, self.f_lsh, self.f_lsc, self.f_lga

    # ---------- 前向 ----------
    def _patchify(self, x):
        """latent [1,16,LH,LW] -> token 矩阵 [S,68] 写盘。"""
        ll = x[0].permute(1, 2, 0)          # [LH,LW,16]
        pp = torch.zeros(self.S, 68)
        for i in range(self.S_H):
            for j in range(self.S_W):
                pp[i * self.S_W + j, :64] = ll[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] \
                    .permute(2, 0, 1).reshape(-1)
        _to_fp16_file(pp, self.f_patch)
        return self.f_patch

    def _unpatchify(self, tok):
        """token 矩阵 [S,64] -> latent [1,16,LH,LW]。"""
        v_lat = torch.zeros(1, 16, self.LH, self.LW)
        for i in range(self.S_H):
            for j in range(self.S_W):
                v_lat[0, :, i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = \
                    tok[i * self.S_W + j].reshape(2, 2, 16).permute(2, 0, 1)
        return v_lat

    def forward(self, x, ctx_path, sigma):
        """x: [1,16,LH,LW]  →  velocity 同形状。ctx_path: fp16 [512,1024] 文件。"""
        host = self.host
        idx = {n: i for i, n in enumerate(self.order)}
        te, lsh, lsc, lga = self.write_cond(sigma)

        tail = [te, lsh, lsc, lga, ctx_path, self.f_cos, self.f_sin]

        # patch embed
        host.run_block(idx["pe"], [self._patchify(x)], self.f_h)
        h = _read_fp16(self.f_h)
        if h.numel() != self.S * 2048:
            raise RuntimeError(
                "pe 输出 %d 个元素，与当前分辨率（%dx%d，S=%d）不符。\n"
                "这个 blob 目录多半是别的分辨率编的、或者几份图拼在一起的残件。"
                % (h.numel(), self.width, self.height, self.S))
        h = h.reshape(self.S, 2048)
        cur = os.path.join(self.tmp, "i_xb.bin")
        _to_fp16_file(h / float(self.pb[0]["d_in"]), cur)

        # 逐块
        for bi in range(self.n_blocks):
            name = "b%02d" % bi
            nxt = os.path.join(self.tmp, "o_%s.bin" % name)
            host.run_block(idx[name], [cur] + tail, nxt)
            v = _read_fp16(nxt, (self.S, 2048)) * float(self.pb[2 * bi + 1]["post_scale"])
            nb = os.path.join(self.tmp, "i_xb_%02d.bin" % bi)
            _to_fp16_file(v, nb)
            cur = nb

        # final layer
        host.run_block(idx["fl"], [cur, te, lsh, lsc], self.f_tok)
        tok = _read_fp16(self.f_tok)
        if tok.numel() != self.S * 64:
            raise RuntimeError("fl 输出 %d 个元素，与 S=%d 不符（token×64=%d）"
                               % (tok.numel(), self.S, self.S * 64))
        tok = tok.reshape(self.S, 64)
        return self._unpatchify(tok)


class _LazyMods(object):
    pass
