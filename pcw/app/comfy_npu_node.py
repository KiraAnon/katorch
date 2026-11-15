"""ComfyUI 节点：调用 KAtorch 的 NPU 生图管线（512x512 约 2 分钟）。

两种用法：
1. 直接在 prompt 文本框写提示词；
2. 把 prompt_in 连到 StringConcatenate 之类节点的输出上（画师串拼接后喂进来），
   此时 prompt 文本框内容会被忽略。

为什么不是「把 DiT 的 Linear 层挂到 NPU」：那样每层输出误差 22%~78%
（见 npu_hook.log 里 blocks.*.mlp.layer* 的 rel 值），图像会崩。
这里改成整条采样在 NPU 上跑完，ComfyUI 只负责界面和工作流编排。
"""
import os
import sys
import time
import numpy as np
import torch

HOME = "/data/data/com.termux/files/home"
APP = os.path.join(HOME, "npt", "app")
for _p in (APP, os.path.join(HOME, "npt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENC = None
_PIPE = None
_LOG = os.path.join(HOME, "npt", "comfy_node.log")


def _log(msg):
    line = "[katorch] " + str(msg)
    print(line, flush=True)
    try:
        with open(_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _lazy():
    """延迟加载：ComfyUI 启动时不碰 NPU，第一次出图才初始化。"""
    global _ENC, _PIPE
    if _ENC is None:
        from enc import Encoder
        mdir = os.path.join(APP, "models")
        _log("加载 Qwen 编码器…")
        _ENC = Encoder(
            qwen_st=os.path.join(mdir, "qwen3_06b.safetensors"),
            anima_st=os.environ.get("KA_ANIMA", os.path.join(
                HOME, "ComfyUI/models/diffusion_models/anima_v1.safetensors")),
            tok_dir=mdir,
        )
    if _PIPE is None:
        from gen import Pipeline
        _log("初始化 NPU 管线…")
        _PIPE = Pipeline()
    return _ENC, _PIPE


class KAtorchNPUGenerate:
    """整条 DiT 采样走 MTK NPU。512x512 约 2 分钟。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "width": ("INT", {"default": 512, "min": 512, "max": 512}),
                "height": ("INT", {"default": 512, "min": 512, "max": 512}),
            },
            "optional": {
                # 连了线就用连线来的文本（画师串拼接的结果）
                "prompt_in": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "KAtorch"

    def generate(self, prompt, negative, seed, width, height, prompt_in=None):
        text = (prompt_in or "").strip() or prompt
        enc, pipe = _lazy()
        t0 = time.time()
        _log("开始 seed=%d len=%d prompt=%s" % (seed, len(text), text[:80]))
        ctx = enc.encode(text)
        x = pipe.generate(ctx, seed=seed)
        os.makedirs(os.path.join(HOME, "npt", "outputs"), exist_ok=True)
        out = os.path.join(HOME, "npt", "outputs", "comfy_%d.png" % int(time.time()))
        path = pipe.decode(x, out)
        _log("完成 %.1fs -> %s" % (time.time() - t0, path))

        from PIL import Image
        arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        return (torch.from_numpy(arr)[None, ...],)


NODE_CLASS_MAPPINGS = {"KAtorchNPUGenerate": KAtorchNPUGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"KAtorchNPUGenerate": "KAtorch NPU 生图"}
