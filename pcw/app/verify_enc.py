"""在 PC 上验证纯 Python 编码器与 ComfyUI 产出的 ctx 是否一致。"""
import sys, os
sys.path.insert(0, r"F:\KAtorch\pcw\app")
sys.path.insert(0, r"D:\comfyui\ComfyUI-master")
import numpy as np
import torch

from enc import Encoder

CM = r"D:\comfyui\ComfyUI-master"
TEXT = ("masterpiece, best quality, very aesthetic, 1girl, solo, white dress, "
        "long white hair, standing in shallow water, soft lighting, full body, detailed eyes")

enc = Encoder(
    qwen_st=os.path.join(CM, "models", "text_encoders", "qwen_3_06b_base.safetensors"),
    anima_st=os.path.join(CM, "models", "diffusion_models", "anima-base-v1.0.safetensors"),
    tok_dir=os.path.join(CM, "comfy", "text_encoders"),
)

print("\n[1] 单测 Qwen tokenizer ...", flush=True)
from transformers import Qwen2Tokenizer
ref = Qwen2Tokenizer.from_pretrained(os.path.join(CM, "comfy", "text_encoders", "qwen25_tokenizer"))
a = enc.qtok.encode(TEXT)
b = ref.encode(TEXT, add_special_tokens=False)
print("   我 %d 个 token，官方 %d 个，一致=%s" % (len(a), len(b), a == b))

print("[2] 编码 ...", flush=True)
ctx = enc.encode(TEXT)
ref_ctx = np.fromfile(r"F:\KAtorch\pcw\ctx\ctx_pos_m.bin", dtype=np.float16).reshape(512, 1024)
n = 64
d = np.abs(ctx[:n].astype(np.float32) - ref_ctx[:n].astype(np.float32))
print("   ctx shape=%s  std=%.4f" % (ctx.shape, ctx.astype(np.float32).std()))
print("   官方    shape=%s  std=%.4f" % (ref_ctx.shape, ref_ctx.astype(np.float32).std()))
print("   前 %d 行 最大差 %.5f  NRMSE %.3f%%"
      % (n, d.max(), np.sqrt((d ** 2).mean()) / (np.sqrt((ref_ctx[:n].astype(np.float32) ** 2).mean()) + 1e-12) * 100))
np.save(r"F:\KAtorch\pcw\ctx\ctx_mine.npy", ctx)
print("   已存 ctx_mine.npy")

# ---- 逐段对比：Qwen3 hidden vs ComfyUI ----
print("[3] 对比 Qwen3 hidden ...", flush=True)
import comfy.sd
clip = comfy.sd.load_clip([os.path.join(CM,"models","text_encoders","qwen_3_06b_base.safetensors")],
                          embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
cond = clip.encode_from_tokens_scheduled(clip.tokenize(TEXT))
href = cond[0][0]
if isinstance(href, list): href = href[0]
href = href.float().numpy().reshape(-1, 1024)
mine = enc.qwen.forward(torch.tensor(enc.qtok.encode(TEXT), dtype=torch.long)).numpy()
print("   官方 hidden std=%.4f  我的 std=%.4f" % (href.std(), mine.std()))
dd = np.abs(mine[:len(href)] - href[:len(mine)])
rel = np.sqrt((dd**2).mean()) / (np.sqrt((href**2).mean())+1e-12)
print("   最大差 %.5f  NRMSE %.2f%%" % (dd.max(), rel*100))
print("   官方[0,:6] =", np.round(href[0,:6],4))
print("   我的[0,:6] =", np.round(mine[0,:6],4))
# ---- 对比 t5 ids 与 adapter ----
print("[4] 对比 t5 ids ...", flush=True)
t5_ref = cond[0][1]["t5xxl_ids"].tolist()
t5_mine = enc.ttok.encode(TEXT, add_eos=False)
print("   官方 %d 个: %s" % (len(t5_ref), t5_ref[:10]))
print("   我的 %d 个: %s" % (len(t5_mine), t5_mine[:10]))
print("   一致: %s" % (t5_ref == t5_mine))

print("[5] 用官方 hidden + 官方 t5 ids 跑我的 adapter ...", flush=True)
with torch.no_grad():
    hid = torch.from_numpy(href).unsqueeze(0) if href.ndim == 2 else href
    got = enc.adapter(hid, torch.tensor([t5_ref], dtype=torch.long))[0].numpy()
rr = ref_ctx[:len(got)].astype(np.float32)
rel2 = np.sqrt(((got[:64]-rr[:64])**2).mean())/(np.sqrt((rr[:64]**2).mean())+1e-12)
print("   adapter 输出 std=%.4f  参考 std=%.4f  NRMSE(前64行) %.2f%%" % (got.std(), rr.std(), rel2*100))
print("   我的[0,:5] =", np.round(got[0,:5],5))
print("   参考[0,:5] =", np.round(rr[0,:5],5))