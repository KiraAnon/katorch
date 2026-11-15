"""端到端时间分解：从提交到出图，每一段都计时。"""
import sys, os, time
sys.path.insert(0, "/data/data/com.termux/files/home/ComfyUI/custom_nodes")

import torch

T0 = time.time()
marks = []


def mark(label):
    marks.append((label, time.time() - T0))
    print("  %-34s %7.2f s" % (label, time.time() - T0))


mark("脚本启动")

for name in list(sys.modules):
    if name.startswith("katorch_npu"):
        del sys.modules[name]

t = time.time()
from katorch_npu import runtime as ka_rt
from katorch_npu import assets as ka_assets
mark("import katorch_npu")

t = time.time()
mf = ka_rt.manifest()
asset = ka_assets.pick_asset_for(mf, "anima_v1", 512, 512)
mark("manifest 扫描")

t = time.time()
model = ka_rt.KAtorchModel(asset)
mark("构造 KAtorchModel")

t = time.time()
clip = ka_rt.load_clip()
mark("load_clip（只建对象）")

t = time.time()
vae = ka_rt.load_vae()
mark("load_vae（只建对象）")

POS = "masterpiece, best quality, score_9, score_8, highres, absurdres, anime screenshot, official art, 1girl, solo, pink hair, blue eyes"
NEG = "worst quality, low quality, score_1, score_2, score_3, simple background, lowres"

t = time.time()
pos = ka_rt.encode_text(clip, POS, model)
mark("编码 POS（含编码器加载）")

t = time.time()
neg = ka_rt.encode_text(clip, NEG, model)
mark("编码 NEG")

t = time.time()
latent = {"samples": torch.zeros(1, 16, 64, 64)}
mark("空 latent")

t = time.time()
dit = ka_rt.get_dit(model)
mark("get_dit（含 NPU 宿主+blob 装载）")

from katorch_npu.nodes import KAtorchKSampler, KAtorchVAEDecode

t = time.time()
(out,) = KAtorchKSampler().sample(model, pos, neg, latent, 12345, 8, 1.0,
                                   "er_sde", "beta", 1.0)
mark("★ 8 步采样")

t = time.time()
(img,) = KAtorchVAEDecode().decode(out, vae)
mark("VAE 解码")

print()
print("=== 汇总 ===")
total = time.time() - T0
print("  总计 %.2f s" % total)
prev = 0
for label, tt in marks:
    d = tt - prev
    prev = tt
    if d > 0.3:
        print("    %-30s %6.2f s (%.1f%%)" % (label, d, d / total * 100))
