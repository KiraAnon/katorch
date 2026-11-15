"""运行时对象与缓存。

三种句柄在节点之间流动：
    KAT_MODEL  → 一份编译好的 DiT 图（资产）+ 可选 LoRA
    KAT_CLIP   → 文本编码器（Qwen3-0.6B + LLMAdapter）
    KAT_VAE    → VAE（CPU torch）

所有重对象都做进程级缓存：ComfyUI 一次会话里反复出图不会重复加载。
"""
import hashlib
import collections
import gc
import json
import os
import threading
import time

import numpy as np
import torch

from . import assets as ka_assets
from .host import HostManager
from .dit import DitRuntime
from .st import LoraSet

HOME = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
WORK = os.environ.get("KA_WORK", os.path.join(HOME, "npt"))
MODELS = os.environ.get("KA_MODELS", os.path.join(HOME, "ComfyUI", "models"))
APP_MODELS = os.environ.get("KA_APP_MODELS", os.path.join(WORK, "app", "models"))

# Wan2.1 / Qwen-Image VAE 的 latent 归一化常数（与 ComfyUI 的 Wan latent_format 一致）
LAT_MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1)
LAT_STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1)

_lock = threading.RLock()
_dit_cache = {}
_clip_cache = {}
_vae_cache = {}


def log(msg):
    print("[katorch] " + str(msg), flush=True)


# ============ 句柄 ============

class KAtorchModel(object):
    """一份编译好的 DiT 图。"""

    def __init__(self, asset, model_path=None, loras=None):
        self.asset = dict(asset)
        self.asset_id = asset["id"]
        self.model_path = model_path
        self.loras = loras or []           # [(name, strength)]
        self.width = int(asset.get("width") or 0)
        self.height = int(asset.get("height") or 0)

    def __repr__(self):
        return "<KAtorchModel %s %dx%d %d块>" % (self.asset_id, self.width, self.height,
                                                self.asset.get("blocks", 0))

    def key(self):
        return (self.asset_id, tuple(self.loras))

    def with_loras(self, loras):
        return KAtorchModel(self.asset, self.model_path, loras)


class KAtorchClip(object):
    """文本编码器（Qwen3-0.6B + LLMAdapter）。

    LLMAdapter 的权重在每个 checkpoint 里，而且微调版（比如 miaomiaoHarem）连这部分都改了，
    所以编码器要按「用哪个 checkpoint 的 llm_adapter」分别缓存。
    """

    def __init__(self, name, qwen_path, anima_path, tok_dir):
        self.name = name
        self.qwen_path = qwen_path
        self.anima_path = anima_path
        self.tok_dir = tok_dir
        self._enc = {}

    def __repr__(self):
        return "<KAtorchClip %s>" % self.name

    def encoder(self, anima_path=None):
        key = anima_path or self.anima_path
        if key not in self._enc:
            from .enc import Encoder
            t0 = time.time()
            log("加载文本编码器（llm_adapter ← %s）…" % os.path.basename(key))
            self._enc[key] = Encoder(qwen_st=self.qwen_path, anima_st=key, tok_dir=self.tok_dir)
            log("文本编码器就绪 %.1fs" % (time.time() - t0))
        return self._enc[key]


class KAtorchVae(object):
    """VAE（解码走 CPU torch；编码优先用 ComfyUI 自带的 Wan VAE）。"""

    def __init__(self, path, name=None):
        self.path = path
        self.name = name or os.path.basename(path)
        self._dec = {}
        self._comfy = None

    def __repr__(self):
        return "<KAtorchVae %s>" % self.name

    def decoder(self, lh, lw):
        key = (lh, lw)
        if key not in self._dec:
            from .vae_decoder import WanVaeDecoder
            from safetensors.torch import load_file
            t0 = time.time()
            log("加载 VAE %s (latent %dx%d) …" % (self.name, lh, lw))
            sd = load_file(self.path, device="cpu")
            self._dec[key] = WanVaeDecoder(sd, H0=lh, W0=lw).eval()
            del sd
            log("VAE 就绪 %.1fs" % (time.time() - t0))
        return self._dec[key]

    def comfy_vae(self):
        """ComfyUI 原生 VAE 对象（图生图编码用；解码仍用我们自己的实现）。"""
        if self._comfy is None:
            import comfy.sd
            import comfy.utils
            sd = comfy.utils.load_torch_file(self.path, safe_load=True)
            self._comfy = comfy.sd.VAE(sd=sd)
        return self._comfy


# ============ 缓存 ============

def get_dit(model):
    """按 (资产, LoRA 组合) 取 DiT 运行时；内部会保证 NPU 宿主与资产匹配。"""
    key = model.key()
    with _lock:
        if key in _dit_cache:
            return _dit_cache[key]
        host = HostManager.get(model.asset)
        lora = None
        if model.loras:
            lora_files = [ln for ln, _ in model.loras]
            if len(lora_files) != 1:
                raise RuntimeError("当前只支持单个预合并 LoRA（资产里需要对应编译产物）")
            lora = LoraSet(lora_files[0])
        rt = DitRuntime(model.asset, host, model_path=model.model_path, lora=lora,
                        lora_strength=model.loras[0][1] if model.loras else 1.0)
        _dit_cache[key] = rt
        return rt


def clear_dit_cache():
    _dit_cache.clear()
    HostManager.close()


def clear_all():
    with _lock:
        _dit_cache.clear()
        _clip_cache.clear()
        _vae_cache.clear()
        _enc_cache.clear()
        HostManager.close()
        gc.collect()


# ============ 文本编码 ============

class KAtorchCond(object):
    """条件：已经算好的 ctx 张量 + 原文。"""

    def __init__(self, ctx, text=""):
        self.ctx = ctx               # np.float16 [512,1024]
        self.text = text

    def __repr__(self):
        return "<KAtorchCond %r len=%d>" % (self.text[:24], len(self.text))

    def write(self, path):
        arr = np.ascontiguousarray(self.ctx, dtype=np.float16)
        arr.tofile(path)
        return path


# 文本编码缓存：一次 25 秒，同样的 prompt 第二次直接命中。
# key 里带 anima 路径 —— 不同底模的 llm_adapter 不一样，不能混用。
# ctx 是 [512,1024] float16 ≈ 1MB，默认存 16 个也就 16MB。
_enc_cache = collections.OrderedDict()
_ENC_CACHE_MAX = int(os.environ.get("KA_ENC_CACHE", "16"))


def encode_text(clip, text, model=None, use_cache=True):
    """文本 → ctx。给了 model 就用该模型自己的 llm_adapter（微调版的文本编码器是改过的）。"""
    anima = model.model_path if model is not None else None
    key = (anima or "", text or "")

    if use_cache and _ENC_CACHE_MAX > 0:
        hit = _enc_cache.get(key)
        if hit is not None:
            _enc_cache.move_to_end(key)
            log("文本编码命中缓存（%d 字符，省 ~25s）" % len(text or ""))
            return KAtorchCond(hit, text)

    enc = clip.encoder(anima)
    t0 = time.time()
    ctx = enc.encode(text or "")
    log("文本编码完成 %.1fs（%d 字符）" % (time.time() - t0, len(text or "")))

    if use_cache and _ENC_CACHE_MAX > 0:
        _enc_cache[key] = ctx
        while len(_enc_cache) > _ENC_CACHE_MAX:
            _enc_cache.popitem(last=False)

    return KAtorchCond(ctx, text)


def clear_enc_cache():
    n = len(_enc_cache)
    _enc_cache.clear()
    if n:
        log("已清空文本编码缓存（%d 条）" % n)
    return n


# ============ VAE ============

# ---- VAE 的 NPU 通道 ----------------------------------------------------
# 4 段 blob 把 512×512 解码从 37s 压到 1.3s（精度：8bit 最大差 2 个色阶）。
# 任何一步不满足（分辨率不是 512、blob 缺失、host 异常）都静默退回 CPU 版，
# 保证出图不会因为加速路径挂掉。
_VAE_NPU = None
_VAE_NPU_TRIED = False


def _vae_npu_decode(z_nhwc, lh, lw):
    global _VAE_NPU, _VAE_NPU_TRIED
    if os.environ.get("KA_VAE_NPU", "1") == "0":
        return None
    if lh != 64 or lw != 64:          # 只有 512×512 编了 blob
        return None
    if _VAE_NPU is None:
        if _VAE_NPU_TRIED:
            return None
        _VAE_NPU_TRIED = True
        try:
            from .vae_npu import VaeNpu
            cand = VaeNpu(hw=lh)
            if cand.available():
                _VAE_NPU = cand
                log("VAE: 使用 NPU 解码（4 段 blob）")
            else:
                log("VAE: 未找到 NPU blob，用 CPU 解码")
        except Exception as e:
            log("VAE: NPU 初始化失败，用 CPU 解码（%s）" % e)
    if _VAE_NPU is None:
        return None
    try:
        out = _VAE_NPU.decode(z_nhwc.numpy())
        return torch.from_numpy(out)
    except Exception as e:
        log("VAE: NPU 执行失败，本次改用 CPU（%s）" % e)
        try:
            _VAE_NPU.close()
        except Exception:
            pass
        _VAE_NPU = None
        return None


def vae_decode(vae, samples, lh, lw):
    """归一化 latent -> IMAGE [1,H,W,3] float 0..1。

    注意 decoder 内部是 NHWC，输出已经是 [H,W,3]，不要再 permute。
    ComfyUI 的 Wan VAE 走视频路径，latent 可能带一维 T（[B,C,1,H,W]），这里兼容掉。
    """
    z = samples.detach().float()
    if z.dim() == 5:
        z = z[:, :, 0]
    z = z * LAT_STD + LAT_MEAN
    z_nhwc = z[0].permute(1, 2, 0).unsqueeze(0)
    img = _vae_npu_decode(z_nhwc, lh, lw)
    if img is None:
        dec = vae.decoder(lh, lw)
        with torch.no_grad():
            img = dec(z_nhwc).squeeze(0)
    img = torch.clamp((img + 1.0) * 127.5, 0, 255).to(torch.uint8)
    return img.unsqueeze(0).float() / 255.0


def vae_encode(vae, image):
    """IMAGE [B,H,W,3] -> 归一化 latent [B,16,H/8,W/8]。

    用 ComfyUI 的 Wan VAE 编码器。它返回的是**未归一化**的 latent（实测：
    直接拿它去乘 latents_std 会过曝，说明归一化不在 VAE.encode 里做），
    所以这里按 Wan21 的 latent_format 自己归一化，和我们 DiT 的空间对齐。
    视频路径会多一维 T，压掉。
    """
    v = vae.comfy_vae()
    img = image.detach().float()[:1]
    with torch.no_grad():
        z = v.encode(img[:, :, :, :3].contiguous())
    z = z.detach().float()
    if z.dim() == 5:
        z = z[:, :, 0]
    return (z - LAT_MEAN) / LAT_STD


# ============ 资产/清单 ============

def manifest(force=False):
    return ka_assets.get_manifest(force=force)


def load_model(asset_id, loras=None):
    mf = manifest()
    asset = ka_assets.find_asset(mf, asset_id)
    if asset is None:
        raise RuntimeError("资产清单里没有 %s（可用：%s）"
                           % (asset_id, ", ".join(a["id"] for a in mf.get("assets", []))))
    if not asset.get("width"):
        raise RuntimeError("资产 %s 的分辨率未知：请在 %s 里补 width/height" % (asset_id, ka_assets.OVERRIDES))
    return KAtorchModel(asset, loras=loras or [])


def load_clip(name="qwen3_06b"):
    mf = manifest()
    # 优先级：显式变量 > 已验证的 npt 目录 > ComfyUI/models 扫描结果
    qwen = os.environ.get("KA_QWEN")
    if not qwen:
        for cand in (os.path.join(APP_MODELS, "qwen3_06b.safetensors"),
                     os.path.join(APP_MODELS, "qwen3.safetensors")):
            if os.path.isfile(cand):
                qwen = cand
                break
    if not qwen:
        for c in mf.get("text_encoders", []) + mf.get("clip", []):
            if name in c["name"] or "qwen" in c["name"].lower():
                qwen = c["path"]
                break
    if not qwen or not os.path.isfile(qwen):
        raise RuntimeError("找不到 Qwen3 编码器权重（用 KA_QWEN 指定，或放到 %s）" % APP_MODELS)

    anima = os.environ.get("KA_ANIMA")
    if not anima:
        for c in mf.get("diffusion", []):
            if "anima" in c["name"].lower():
                anima = c["path"]
                break
    if not anima:
        anima = os.path.join(MODELS, "diffusion_models", "anima_v1.safetensors")
    if not os.path.isfile(anima):
        raise RuntimeError("找不到 DiT 权重（llm_adapter 也在里面）：%s" % anima)

    tok_dir = os.environ.get("KA_TOK", APP_MODELS)
    key = (qwen, anima, tok_dir)
    with _lock:
        if key not in _clip_cache:
            _clip_cache[key] = KAtorchClip(name, qwen, anima, tok_dir)
        return _clip_cache[key]


def load_vae(name=None):
    mf = manifest()
    path = os.environ.get("KA_VAE")
    if not path:
        want = name or ""
        for c in mf.get("vaes", []):
            if want and want in c["name"]:
                path = c["path"]
                break
        if not path and mf.get("vaes"):
            # 优先 qwen 的 VAE
            for c in mf["vaes"]:
                if "qwen" in c["name"].lower():
                    path = c["path"]
                    break
            path = path or mf["vaes"][0]["path"]
    if not path or not os.path.isfile(path):
        path = os.path.join(MODELS, "vae", "qwen_image_vae.safetensors")
    if not os.path.isfile(path):
        raise RuntimeError("找不到 VAE 权重：%s" % path)
    with _lock:
        if path not in _vae_cache:
            _vae_cache[path] = KAtorchVae(path, os.path.basename(path))
        return _vae_cache[path]


def load_lora(name, strength=1.0):
    """LoRA 走「预合并组合」：manifest 里有一个已把它编进 blob 的资产才能用。"""
    mf = manifest()
    path = None
    for c in mf.get("loras", []):
        if name in c["name"]:
            path = c["path"]
            break
    if path is None:
        raise RuntimeError("models/loras 里没有 %s" % name)
    return path, float(strength)
