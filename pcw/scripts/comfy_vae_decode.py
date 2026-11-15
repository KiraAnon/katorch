"""用 ComfyUI 自己的 VAE 解码我们 NPU 产出的 latent，判定 VAE 实现是否一致。"""
import sys, os
sys.path.insert(0, r"D:\comfyui\ComfyUI-master")

import numpy as np
import torch
import comfy.sd
import comfy.utils
from PIL import Image

LAT = r"F:\KAtorch\pcw\laten.bin"
VAE_P = r"D:\comfyui\ComfyUI-master\models\vae\Qwen\qwen_image_vae.safetensors"
OUT = r"F:\KAtorch\pcw\comfy_vae_decode.png"

lat_path = sys.argv[1] if len(sys.argv) > 1 else r"F:\KAtorch\pcw\final_pc_latent.bin"
out_path = sys.argv[2] if len(sys.argv) > 2 else OUT

# NPU 存的 final_latent.bin: (16, 32, 32) fp16，是「已归一化」的潜变量
a = np.fromfile(lat_path, dtype=np.float16).astype(np.float32)
print("latent bytes=%d" % a.size)
x = torch.from_numpy(a).reshape(1, 16, 1, 32, 32)
print("latent std=%.4f min=%.3f max=%.3f" % (x.std(), x.min(), x.max()))

print("加载 ComfyUI VAE ...", flush=True)
vae_sd = comfy.utils.load_torch_file(VAE_P)
vae = comfy.sd.VAE(sd=vae_sd)

with torch.no_grad():
    img = vae.decode(x)
print("decode out", tuple(img.shape), "std=%.3f min=%.3f max=%.3f"
      % (img.std(), img.min(), img.max()))

o = img[0, 0] if img.dim() == 5 else img[0]
img8 = torch.clamp((o + 1) * 127.5, 0, 255).to(torch.uint8).numpy()
Image.fromarray(img8).save(out_path)
print("->", out_path)
