"""给 gen.py 加两个可切换的采样器：euler（原版）和 er_sde（对齐 ComfyUI）。

对齐依据：comfy/k_diffusion/sampling.py::sample_er_sde（ER-SDE-Solver-3, arXiv:2309.06169）
    er_lambdas  = exp(-half_log_snr) = sigma / alpha        (flow matching 下 alpha = 1 - sigma)
    noise_scaler(x) = x * (exp(x**0.3) + 10)
    Stage1: x = r_alpha*r*x + alpha_t*(1-r)*denoised
    Stage2/3: 二阶/三阶修正（用历史 denoised 差分，200 个积分点）
    加噪:   x += alpha_t * eps * s_noise * sqrt(el_t^2 - el_s^2*r^2)
其中 denoised = x - v * sigma（flow matching 的 x0 预测）。

切换：KA_SAMPLER=euler | er_sde（默认 er_sde，和 PC 工作流一致）
      KA_SDE_STAGE=1|2|3（默认 3）
      KA_S_NOISE=1.0
"""
import re

P = "/data/data/com.termux/files/home/npt/app/gen.py"
src = open(P, encoding="utf-8").read()

if "KA_SAMPLER" in src:
    print("已经装过采样器补丁，跳过")
    raise SystemExit

# ---------- 1) 在文件里插入采样器实现 ----------
HELPERS = '''

# ============ 采样器实现（euler / er_sde）============
SAMPLER = os.environ.get("KA_SAMPLER", "er_sde")
SDE_STAGE = int(os.environ.get("KA_SDE_STAGE", "3"))
S_NOISE = float(os.environ.get("KA_S_NOISE", "1.0"))


def _noise_scaler(x):
    """ComfyUI 的 default_er_sde_noise_scaler: x * (exp(x**0.3) + 10)"""
    return x * ((x ** 0.3).exp() + 10.0)


def _er_lambda(sigma):
    """sigma/(1-sigma)，对应 ComfyUI 的 exp(-half_log_snr)"""
    return sigma / (1.0 - sigma) if sigma < 1.0 else 1e4


def sample_euler(x, sigs, v_fn, **kw):
    """原版：x_{t+1} = x_t - v * (sigma_t - sigma_{t+1})"""
    for i in range(len(sigs) - 1):
        v = v_fn(x, sigs[i], i)
        x = x - v * (sigs[i] - sigs[i + 1])
    return x


def sample_er_sde(x, sigs, v_fn, stage=None, s_noise=None, **kw):
    """ER-SDE-Solver-3，与 ComfyUI 的 er_sde 对齐。"""
    stage = SDE_STAGE if stage is None else stage
    s_noise = S_NOISE if s_noise is None else s_noise
    num_pts = 200.0
    els = [_er_lambda(s) for s in sigs]
    old_d = None
    old_dd = None
    for i in range(len(sigs) - 1):
        v = v_fn(x, sigs[i], i)
        denoised = x - v * sigs[i]           # flow matching: x0 = x - v*sigma
        stage_used = min(stage, i + 1)
        if sigs[i + 1] == 0:
            x = denoised
        else:
            el_s, el_t = els[i], els[i + 1]
            alpha_s = sigs[i] / el_s if el_s > 0 else 1.0
            alpha_t = sigs[i + 1] / el_t if el_t > 0 else 1.0
            r_alpha = alpha_t / alpha_s
            r = _noise_scaler(torch.tensor(el_t)) / _noise_scaler(torch.tensor(el_s))
            r = float(r)

            # Stage 1: Euler
            x = r_alpha * r * x + alpha_t * (1.0 - r) * denoised

            if stage_used >= 2 and old_d is not None:
                dt = el_t - el_s
                lam_step = -dt / num_pts
                lam_pos = [el_t + k * lam_step for k in range(int(num_pts))]
                scaled = torch.stack([_noise_scaler(torch.tensor(p)) for p in lam_pos])
                s_int = float(torch.sum(1.0 / scaled) * lam_step)
                denoised_d = (denoised - old_d) / (el_s - els[i - 1])
                x = x + alpha_t * (dt + s_int * float(_noise_scaler(torch.tensor(el_t)))) * denoised_d

                if stage_used >= 3 and old_dd is not None:
                    su = sum((p - el_s) / float(_noise_scaler(torch.tensor(p))) for p in lam_pos)
                    su = su * lam_step
                    denoised_u = (denoised_d - old_dd) / ((el_s - els[i - 2]) / 2.0)
                    x = x + alpha_t * ((dt ** 2) / 2.0 + su * float(_noise_scaler(torch.tensor(el_t)))) * denoised_u
                old_dd = denoised_d

            if s_noise > 0:
                var = el_t ** 2 - (el_s ** 2) * (r ** 2)
                if var > 0:
                    eps = torch.randn_like(x)
                    x = x + alpha_t * eps * s_noise * (var ** 0.5)
        old_d = denoised
    return x


SAMPLERS = {"euler": sample_euler, "er_sde": sample_er_sde}

'''

# 插到 import 之后、常量之前的位置（找 STEPS 定义前）
anchor = 'STEPS = int(os.environ.get("KA_STEPS", "8"))'
if anchor in src:
    src = src.replace(anchor, HELPERS + anchor, 1)
else:
    print("找不到 STEPS 锚点")
    raise SystemExit

# ---------- 2) 改 generate() 用采样器 ----------
OLD_GEN = '''        torch.manual_seed(int(seed))
        x = torch.randn(1, 16, LH, LW)
        self._start_host()
        t0 = time.time()
        for step in range(STEPS):
            sv = self.sigs[step]
            raw = self._t_embed(sv)
            lora = F.linear(F.silu(F.linear(raw, self.W1)), self.W2)
            te = raw * torch.rsqrt((raw * raw).mean(-1, keepdim=True) + 1e-6) * self.TNORM
            te.numpy().astype(np.float16).tofile(TMP + "/i_te.bin")
            lora[:, :2048].numpy().astype(np.float16).tofile(TMP + "/i_lsh.bin")
            lora[:, 2048:4096].numpy().astype(np.float16).tofile(TMP + "/i_lsc.bin")
            lora[:, 4096:6144].numpy().astype(np.float16).tofile(TMP + "/i_lga.bin")
            v = self._stack(x, ctx_path)
            x = x - v * (self.sigs[step] - self.sigs[step + 1])
            if progress:
                progress(step + 1, STEPS, time.time() - t0)
        return x'''

NEW_GEN = '''        torch.manual_seed(int(seed))
        x = torch.randn(1, 16, LH, LW)
        self._start_host()
        t0 = time.time()
        _cache = {}

        def _v_fn(xx, sigma, step):
            """给定当前 x 和 sigma，跑一遍 DiT 拿速度场。"""
            key = step
            if key in _cache:
                return _cache[key]
            sv = self.sigs[step]
            raw = self._t_embed(sv)
            lora = F.linear(F.silu(F.linear(raw, self.W1)), self.W2)
            te = raw * torch.rsqrt((raw * raw).mean(-1, keepdim=True) + 1e-6) * self.TNORM
            te.numpy().astype(np.float16).tofile(TMP + "/i_te.bin")
            lora[:, :2048].numpy().astype(np.float16).tofile(TMP + "/i_lsh.bin")
            lora[:, 2048:4096].numpy().astype(np.float16).tofile(TMP + "/i_lsc.bin")
            lora[:, 4096:6144].numpy().astype(np.float16).tofile(TMP + "/i_lga.bin")
            v = self._stack(xx, ctx_path)
            _cache[key] = v
            if progress:
                progress(step + 1, STEPS, time.time() - t0)
            return v

        _fn = SAMPLERS.get(SAMPLER, sample_er_sde)
        print("[gen] 采样器 %s (stage=%d, s_noise=%.2f)" % (SAMPLER, SDE_STAGE, S_NOISE), flush=True)
        x = _fn(x, self.sigs, _v_fn)
        return x'''

if OLD_GEN in src:
    src = src.replace(OLD_GEN, NEW_GEN)
    print("已替换 generate() 的采样循环")
else:
    print("generate() 锚点没找到！")

open(P, "w", encoding="utf-8").write(src)
print("完成。KA_SAMPLER=euler 可回退到原版。")
