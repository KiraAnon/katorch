"""用 ComfyUI 的文本编码器生成 NPU 链路需要的 ctx（512x1024 fp16，尾部零填充）。
   正/负两份，用于 CFG。
"""
import sys, os
sys.path.insert(0, r"D:\comfyui\ComfyUI-master")

import numpy as np
import torch
import comfy.sd
import comfy.model_management
from comfy.cli_args import args as comfy_args

CLIP_PATH = r"D:\comfyui\ComfyUI-master\models\text_encoders\qwen_3_06b_base.safetensors"
OUT = r"F:\KAtorch\pcw\ctx"
M = 512

POS = ("masterpiece, best quality, very aesthetic, 1girl, solo, white dress, "
       "long white hair, standing in shallow water, soft lighting, full body, detailed eyes")
NEG = ""

os.makedirs(OUT, exist_ok=True)

print("loading clip ...", flush=True)
clip = comfy.sd.load_clip([CLIP_PATH], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
print("clip loaded", flush=True)


def encode(text, name):
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    t = cond[0][0]                      # (B, seq, dim)
    if isinstance(t, list):
        t = t[0]
    arr = t[0].float().numpy()          # (seq, dim)
    seq, dim = arr.shape
    print("  %-8s seq=%d dim=%d  (%-20s)" % (name, seq, dim, text[:20]), flush=True)
    if seq > M:
        arr = arr[:M]
        seq = M
    out = np.zeros((M, dim), np.float32)
    out[:seq] = arr
    p = os.path.join(OUT, name + ".bin")
    out.astype(np.float16).tofile(p)
    print("        -> %s  %d bytes  nonzero_rows=%d" % (p, os.path.getsize(p), seq), flush=True)
    return dim


dp = encode(POS, "ctx_pos")
dn = encode(NEG, "ctx_neg")
assert dp == dn == 1024, "维度不对: %s %s" % (dp, dn)
print("OK")
