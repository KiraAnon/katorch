"""采样器与调度器。

调度器决定 sigma 序列（噪声强度），采样器决定怎么从 x_t 走到 x_{t+1}。
PC 上的 ComfyUI 工作流用的是 `er_sde` + `beta`，手机默认对齐这套。

采样器签名统一为 `fn(x, sigmas, v_fn, **kw) -> x`，`v_fn(x, sigma, step)` 返回速度场。
"""
import math
import os

import torch

DEFAULT_SHIFT = float(os.environ.get("KA_SHIFT", "3.0"))


# ============ 调度器 ============

def _flow_shift(t, a):
    return t if a == 1.0 else a * t / (1 + (a - 1) * t)


def sched_linear(steps, shift=DEFAULT_SHIFT):
    """原版：线性时间 + flow-matching shift（3.0）。"""
    ts = [_flow_shift(shift, (k + 1) / 1000.0) for k in range(1000)]
    return [ts[-(1 + int(x * (1000.0 / steps)))] for x in range(steps)] + [0.0]


def sched_beta(steps, a=0.6, b=0.6):
    """ComfyUI 的 beta 调度器：Beta 分布逆分位。前期降得快、后期在高精度区补步进。"""
    from scipy.stats import beta as _beta
    sig = [float(_beta.ppf(1.0 - (i + 1.0) / steps, a, b)) for i in range(steps)]
    return [min(max(v, 0.0), 1.0) for v in sig] + [0.0]


def sched_normal(steps, smax=1.0, smin=0.0):
    return [smax + (smin - smax) * i / max(steps - 1, 1) for i in range(steps)] + [0.0]


def sched_karras(steps, smax=1.0, smin=0.001, rho=7.0):
    inv = 1.0 / rho
    lo, hi = smin ** inv, smax ** inv
    return [(hi + (lo - hi) * i / max(steps - 1, 1)) ** rho for i in range(steps)] + [0.0]


def sched_sgm_uniform(steps, smax=1.0, smin=0.0):
    # 时间步均匀，再换算成 sigma（flow matching 下 sigma 与时间同构）
    return [smax + (smin - smax) * (i + 1) / steps for i in range(steps)] + [0.0]


def sched_simple(steps):
    return sched_normal(steps)


SCHEDULERS = {
    "beta": sched_beta,
    "linear": sched_linear,
    "normal": sched_normal,
    "karras": sched_karras,
    "sgm_uniform": sched_sgm_uniform,
    "simple": sched_simple,
}
SCHEDULER_NAMES = ["beta", "linear", "normal", "karras", "sgm_uniform", "simple"]


def build_sigmas(steps, scheduler="beta", shift=DEFAULT_SHIFT, beta_a=0.6, beta_b=0.6):
    """返回长度 steps+1 的 sigma 序列，末尾一定是 0。"""
    steps = max(1, int(steps))
    fn = SCHEDULERS.get(scheduler, sched_beta)
    if scheduler == "beta":
        sig = fn(steps, beta_a, beta_b)
    elif scheduler == "linear":
        sig = fn(steps, shift)
    else:
        sig = fn(steps)
    sig = [float(v) for v in sig]
    sig[-1] = 0.0
    # 单调递减保护：非单调的序列会让采样发散
    for i in range(1, len(sig)):
        if sig[i] > sig[i - 1]:
            sig[i] = sig[i - 1]
    return sig


def apply_denoise(sigmas, denoise):
    """denoise<1 时只用后半段（图生图的常规做法）。"""
    if denoise >= 0.999:
        return sigmas
    n = len(sigmas) - 1
    k = max(1, min(int(round(n * float(denoise))), n))
    return sigmas[n - k:]


# ============ 采样器 ============

def _noise_scaler(x):
    """ComfyUI 的 default_er_sde_noise_scaler: x * (exp(x**0.3) + 10)"""
    return x * ((x ** 0.3).exp() + 10.0)


def _er_lambda(sigma):
    """sigma/(1-sigma)，对应 ComfyUI 的 exp(-half_log_snr)（flow matching 下 alpha = 1-sigma）。"""
    return sigma / (1.0 - sigma) if sigma < 1.0 else 1e4


def sample_euler(x, sigmas, v_fn, **kw):
    """一阶 Euler：x_{t+1} = x_t - v * (sigma_t - sigma_{t+1})。"""
    for i in range(len(sigmas) - 1):
        v = v_fn(x, sigmas[i], i)
        x = x - v * (sigmas[i] - sigmas[i + 1])
    return x


def sample_heun(x, sigmas, v_fn, **kw):
    """二阶 Heun（确定性，每步两次 DiT，慢一倍但更稳）。"""
    n = len(sigmas) - 1
    for i in range(n):
        s0, s1 = sigmas[i], sigmas[i + 1]
        v0 = v_fn(x, s0, i)
        x_e = x - v0 * (s0 - s1)
        if s1 == 0.0:
            x = x_e
            continue
        v1 = v_fn(x_e, s1, i)
        x = x - (v0 + v1) * 0.5 * (s0 - s1)
    return x


def sample_er_sde(x, sigmas, v_fn, stage=3, s_noise=1.0, **kw):
    """ER-SDE-Solver-3，与 ComfyUI 的 er_sde 严格对齐（arXiv:2309.06169）。

    Stage1 = Euler，Stage2/3 用历史 denoised 做二阶/三阶修正，每步注入噪声。
    注意：会调用 torch.randn_like，所以同 seed 多次跑结果不会完全一致。
    """
    num_pts = 200.0
    els = [_er_lambda(s) for s in sigmas]
    old_d = None
    old_dd = None
    for i in range(len(sigmas) - 1):
        v = v_fn(x, sigmas[i], i)
        denoised = x - v * sigmas[i]
        stage_used = min(stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            el_s, el_t = els[i], els[i + 1]
            alpha_s = sigmas[i] / el_s if el_s > 0 else 1.0
            alpha_t = sigmas[i + 1] / el_t if el_t > 0 else 1.0
            r_alpha = alpha_t / alpha_s
            r = float(_noise_scaler(torch.tensor(el_t)) / _noise_scaler(torch.tensor(el_s)))

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
                    x = x + alpha_t * ((dt ** 2) / 2.0
                                       + su * float(_noise_scaler(torch.tensor(el_t)))) * denoised_u
                old_dd = denoised_d

            if s_noise > 0:
                var = el_t ** 2 - (el_s ** 2) * (r ** 2)
                if var > 0:
                    eps = torch.randn_like(x)
                    x = x + alpha_t * eps * s_noise * (var ** 0.5)
        old_d = denoised
    return x


SAMPLERS = {
    "euler": sample_euler,
    "er_sde": sample_er_sde,
    "heun": sample_heun,
}
SAMPLER_NAMES = ["er_sde", "euler", "heun"]
