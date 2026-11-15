"""KAtorch 的 ComfyUI 节点：每个节点只做一件事。

数据流：
    KAtorchUNETLoader ─KAT_MODEL─┐
    KAtorchCLIPLoader ─KAT_CLIP─┬┴─ KAtorchCLIPTextEncode ─KAT_COND─┐
    KAtorchEmptyLatentImage ─LATENT─────────────────────────────────┤
    KAtorchVAELoader ─KAT_VAE───────────────────────────────────────┤
                                                                    ▼
                                                        KAtorchKSampler ─LATENT─► KAtorchVAEDecode ─IMAGE─► SaveImage
"""
import os
import time

import numpy as np
import torch

from . import assets as ka_assets
from . import runtime as ka_rt
from .samplers import SAMPLER_NAMES, SCHEDULER_NAMES, SAMPLERS, build_sigmas, apply_denoise

TMP = os.environ.get("KA_TMP", os.path.join(ka_rt.WORK, "gen_app"))


def _progress(total):
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        class _P(object):
            def update(self, n):
                pass
        return _P()


def _interrupt_check():
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
        return True
    except ImportError:
        return False


def _asset_choices():
    try:
        opts = ka_assets.asset_options(ka_rt.manifest())
    except Exception as e:
        ka_rt.log("扫描资产失败：%s" % e)
        opts = []
    return [name for name, _ in opts] or ["(没有可用资产)"]


def _asset_from_choice(choice):
    mf = ka_rt.manifest()
    for name, aid in ka_assets.asset_options(mf):
        if name == choice:
            return aid
    return choice


def _file_choices(kind, fallback):
    try:
        mf = ka_rt.manifest()
        names = [os.path.basename(c["name"]) for c in mf.get(kind, [])]
    except Exception:
        names = []
    return names or [fallback]

# ============ 加载器 ============

class KAtorchUNETLoader(object):
    """选一份已编译好的 DiT 资产（blob 集）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"asset": (_asset_choices(), {"default": _asset_choices()[0]})}}

    RETURN_TYPES = ("KAT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "KAtorch/loaders"

    def load(self, asset):
        t0 = time.time()
        model = ka_rt.load_model(_asset_from_choice(asset))
        ka_rt.log("装载模型 %s（%.2fs）" % (model, time.time() - t0))
        return (model,)


class KAtorchCLIPLoader(object):
    """文本编码器（Qwen3-0.6B + Anima 的 LLMAdapter）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_name": (_file_choices("clips", "qwen3_06b"), {})}}

    RETURN_TYPES = ("KAT_CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "KAtorch/loaders"

    def load(self, clip_name):
        return (ka_rt.load_clip(clip_name),)


class KAtorchVAELoader(object):
    """VAE（解码在 CPU 跑，约 3-5 秒）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (_file_choices("vaes", "qwen_image_vae.safetensors"), {})}}

    RETURN_TYPES = ("KAT_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "KAtorch/loaders"

    def load(self, vae_name):
        return (ka_rt.load_vae(vae_name),)


class KAtorchLoraLoader(object):
    """LoRA：NPU 是静态图，只能在编译前把 LoRA 并进权重，所以这里走「预合并组合」查表。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("KAT_MODEL",),
                "lora_name": (_file_choices("loras", "(没有 LoRA)"), {}),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("KAT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "KAtorch/loaders"

    def apply(self, model, lora_name, strength):
        path, s = ka_rt.load_lora(lora_name, strength)
        ka_rt.log("提示：LoRA %s 需要在编译期已合并进 %s，否则效果不变" % (lora_name, model.asset_id))
        return (model.with_loras([(path, s)]),)


# ============ 条件 ============

class KAtorchCLIPTextEncode(object):
    """prompt → ctx（512×1024 fp16）。

    连上 model 时用该模型 checkpoint 里的 llm_adapter —— 微调底模（如 miaomiaoHarem）
    连文本编码器都改过，不跟着换会导致条件空间错位。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("KAT_CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "text_in": ("STRING", {"forceInput": True}),
                "model": ("KAT_MODEL",),
            },
        }

    RETURN_TYPES = ("KAT_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "KAtorch/conditioning"

    def encode(self, clip, text, text_in=None, model=None):
        t = (text_in or "").strip() or text
        t0 = time.time()
        cond = ka_rt.encode_text(clip, t, model)
        ka_rt.log("  [计时] CLIPTextEncode 节点整体 %.2fs（%d 字符）" % (time.time() - t0, len(t)))
        return (cond,)


# ============ 潜空间 ============

class KAtorchEmptyLatentImage(object):
    """空 latent。分辨率必须是已经编译过的资产，否则 KSampler 会直接报错。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "KAtorch/latent"

    def generate(self, width, height, batch_size=1):
        return ({"samples": torch.zeros(batch_size, 16, height // 8, width // 8)},)


# ============ 采样（核心） ============

class KAtorchKSampler(object):
    """在 NPU 上逐块跑 DiT 的采样器。接口对齐 ComfyUI 原生的 KSampler。

    cfg>1 需要正负两个分支各跑一遍 DiT，耗时翻倍；cfg=1 时只跑一遍。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("KAT_MODEL",),
                "positive": ("KAT_COND",),
                "negative": ("KAT_COND",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 64}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (SAMPLER_NAMES, {"default": "er_sde"}),
                "scheduler": (SCHEDULER_NAMES, {"default": "beta"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.1}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "sde_stage": ("INT", {"default": 3, "min": 1, "max": 3}),
                "beta_a": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.05}),
                "beta_b": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "KAtorch/sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, shift=3.0, s_noise=1.0, sde_stage=3,
               beta_a=0.6, beta_b=0.6):
        t_all = time.time()
        samples = latent_image["samples"]
        if samples.dim() == 5:      # ComfyUI 的 Wan VAE 会带一维 T
            samples = samples[:, :, 0]
        if samples.shape[0] != 1:
            raise RuntimeError("NPU 是静态图，一次只能跑一张（batch=%d）。请把 batch_size 设为 1。"
                               % samples.shape[0])
        b, c, lh, lw = samples.shape
        if c != 16:
            raise RuntimeError("latent 通道数必须是 16，收到 %d" % c)

        asset = model.asset
        if int(asset.get("width", 0)) != lw * 8 or int(asset.get("height", 0)) != lh * 8:
            raise RuntimeError(
                "分辨率不匹配：latent 是 %dx%d（图 %dx%d），当前资产 %s 是 %sx%s。\n"
                "NPU 是静态图，换分辨率必须先用新分辨率重新编译 blob 并把目录放进 blob 根目录。"
                % (lw * 8, lh * 8, lw * 8, lh * 8, asset["id"], asset.get("width"), asset.get("height")))

        t_dit = time.time()
        rt = ka_rt.get_dit(model)
        ka_rt.log("  [计时] get_dit（NPU 宿主+blob 装载） %.2fs" % (time.time() - t_dit))
        sigmas = build_sigmas(int(steps), scheduler, shift, beta_a, beta_b)
        sigmas = apply_denoise(sigmas, denoise)
        n_steps = len(sigmas) - 1
        ka_rt.log("采样 %s/%s 步数 %d cfg=%.2f seed=%d sigmas=%s"
                  % (sampler_name, scheduler, n_steps, cfg, seed,
                     [round(s, 4) for s in sigmas]))

        t_ctx = time.time()
        os.makedirs(TMP, exist_ok=True)
        ctx_pos = os.path.join(TMP, "i_ctx_pos.bin")
        positive.write(ctx_pos)
        ctx_neg = None
        if cfg != 1.0:
            ctx_neg = os.path.join(TMP, "i_ctx_neg.bin")
            negative.write(ctx_neg)
        ka_rt.log("  [计时] 写 ctx 文件 %.2fs" % (time.time() - t_ctx))

        t_noise = time.time()
        torch.manual_seed(int(seed) & 0x7FFFFFFF)
        noise = torch.randn_like(samples)
        if denoise >= 0.999:
            # 纯噪声起手：Anima 的 flow matching 在 t=1 时 x 本身就是标准正态（与旧管线一致）
            x = samples + noise
        else:
            x = samples + noise * max(sigmas[0], 1e-3)
        ka_rt.log("  [计时] 生成初始噪声 %.2fs" % (time.time() - t_noise))
        ka_rt.log("  [计时] 采样前累计 %.2fs" % (time.time() - t_all))

        bar = _progress(n_steps)
        state = {"i": 0}

        def v_fn(xx, sigma, step):
            t0 = time.time()
            v_pos = rt.forward(xx, ctx_pos, sigma)
            if ctx_neg is not None:
                v_neg = rt.forward(xx, ctx_neg, sigma)
                v = v_neg + float(cfg) * (v_pos - v_neg)
            else:
                v = v_pos
            state["i"] += 1
            bar.update(1)
            ka_rt.log("  步 %d/%d  sigma=%.4f  %.1fs" % (state["i"], n_steps, sigma, time.time() - t0))
            _interrupt_check()
            return v

        fn = SAMPLERS.get(sampler_name, SAMPLERS["er_sde"])
        x = fn(x, sigmas, v_fn, stage=int(sde_stage), s_noise=float(s_noise))
        ka_rt.log("采样完成 %.1fs" % (time.time() - t_all))
        return ({"samples": x},)


# ============ VAE ============

class KAtorchVAEDecode(object):
    """latent → IMAGE。输出是标准 ComfyUI IMAGE，可以直接接 SaveImage/PreviewImage。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("KAT_VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "KAtorch/vae"

    def decode(self, samples, vae):
        z = samples["samples"]
        lh, lw = z.shape[-2:]
        t0 = time.time()
        img = ka_rt.vae_decode(vae, z, lh, lw)
        ka_rt.log("VAE 解码 %.1fs → %dx%d" % (time.time() - t0, img.shape[2], img.shape[1]))
        return (img,)


class KAtorchVAEEncode(object):
    """IMAGE → latent（图生图/重绘）。用 ComfyUI 自带的 Wan VAE 编码器。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pixels": ("IMAGE",), "vae": ("KAT_VAE",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "KAtorch/vae"

    def encode(self, pixels, vae):
        t0 = time.time()
        z = ka_rt.vae_encode(vae, pixels)
        ka_rt.log("VAE 编码 %.1fs → %s" % (time.time() - t0, tuple(z.shape)))
        return ({"samples": z},)


# ============ 工具节点 ============

class KAtorchModelInfo(object):
    """把资产详情吐成文本，方便在界面上确认跑的是哪份图。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("KAT_MODEL",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "info"
    CATEGORY = "KAtorch/utils"

    def info(self, model):
        a = model.asset
        lines = [
            "资产    : %s" % a.get("id"),
            "分辨率  : %sx%s（latent %sx%s，%d token）"
            % (a.get("width"), a.get("height"), (a.get("width") or 0) // 8,
               (a.get("height") or 0) // 8, a.get("tokens")),
            "块数    : %s（%s）" % (a.get("blocks"), " → ".join(a.get("order", [])[:3]) + " → …"),
            "体积    : %.2f GB" % ((a.get("bytes") or 0) / 1e9),
            "blob    : %s" % a.get("blob_dir"),
            "校准    : %s" % a.get("calib"),
        ]
        return ("\n".join(lines),)


class KAtorchFreeMemory(object):
    """释放 NPU 宿主（blob 常驻内存约 4GB），以及 DiT 缓存。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "free"
    CATEGORY = "KAtorch/utils"
    OUTPUT_NODE = True

    def free(self):
        ka_rt.clear_dit_cache()
        ka_rt.log("已释放 NPU 宿主")
        return ()


class KAtorchClearCondCache(object):
    """清空文本编码缓存。

    文本编码一次约 25 秒，同样的 prompt 第二次会直接命中缓存。
    调完提示词如果发现出图没变化、或者想强制重算，就跑一下这个。
    换底模时不需要清 —— 缓存 key 里带了模型路径，不会串。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "clear"
    CATEGORY = "KAtorch/utils"
    OUTPUT_NODE = True

    def clear(self):
        n = ka_rt.clear_enc_cache()
        msg = "已清空文本编码缓存（%d 条）" % n
        return (msg,)


# ============ 兼容旧工作流的黑盒节点 ============

class KAtorchNPUGenerate(object):
    """一步出图（兼容旧工作流）。内部就是 UNETLoader+CLIP+KSampler+VAEDecode 的串联。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 64}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (SAMPLER_NAMES, {"default": "er_sde"}),
                "scheduler": (SCHEDULER_NAMES, {"default": "beta"}),
            },
            "optional": {"prompt_in": ("STRING", {"forceInput": True})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "KAtorch"

    def generate(self, prompt, negative, seed, width, height, steps, cfg,
                 sampler_name, scheduler, prompt_in=None):
        text = (prompt_in or "").strip() or prompt
        mf = ka_rt.manifest()
        asset = ka_assets.pick_asset_for(mf, "anima_v1", width, height)
        model = ka_rt.KAtorchModel(asset)
        clip = ka_rt.load_clip()
        vae = ka_rt.load_vae()

        pos = ka_rt.encode_text(clip, text, model)
        neg = ka_rt.encode_text(clip, negative or "", model)
        latent = {"samples": torch.zeros(1, 16, height // 8, width // 8)}

        sampler = KAtorchKSampler()
        (out,) = sampler.sample(model, pos, neg, latent, seed, steps, cfg,
                                sampler_name, scheduler, 1.0)
        (img,) = KAtorchVAEDecode().decode(out, vae)
        return (img,)


NODE_CLASS_MAPPINGS = {
    "KAtorchUNETLoader": KAtorchUNETLoader,
    "KAtorchCLIPLoader": KAtorchCLIPLoader,
    "KAtorchVAELoader": KAtorchVAELoader,
    "KAtorchLoraLoader": KAtorchLoraLoader,
    "KAtorchCLIPTextEncode": KAtorchCLIPTextEncode,
    "KAtorchEmptyLatentImage": KAtorchEmptyLatentImage,
    "KAtorchKSampler": KAtorchKSampler,
    "KAtorchVAEDecode": KAtorchVAEDecode,
    "KAtorchVAEEncode": KAtorchVAEEncode,
    "KAtorchModelInfo": KAtorchModelInfo,
    "KAtorchFreeMemory": KAtorchFreeMemory,
    "KAtorchClearCondCache": KAtorchClearCondCache,
    "KAtorchNPUGenerate": KAtorchNPUGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KAtorchUNETLoader": "KAtorch 模型加载 (NPU)",
    "KAtorchCLIPLoader": "KAtorch 文本编码器加载",
    "KAtorchVAELoader": "KAtorch VAE 加载",
    "KAtorchLoraLoader": "KAtorch LoRA 加载",
    "KAtorchCLIPTextEncode": "KAtorch 提示词编码",
    "KAtorchEmptyLatentImage": "KAtorch 空 Latent",
    "KAtorchKSampler": "KAtorch 采样器 (NPU)",
    "KAtorchVAEDecode": "KAtorch VAE 解码",
    "KAtorchVAEEncode": "KAtorch VAE 编码",
    "KAtorchModelInfo": "KAtorch 模型信息",
    "KAtorchFreeMemory": "KAtorch 释放显存",
    "KAtorchClearCondCache": "KAtorch 清空文本缓存",
    "KAtorchNPUGenerate": "KAtorch NPU 生图 (一体)",
}
