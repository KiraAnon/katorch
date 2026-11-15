"""用 ComfyUI 在 256x256、无 LoRA、欧拉采样下出图，作为 NPU 链路的公平对照组。"""
import json, os, sys, random, urllib.request

SRV = "http://127.0.0.1:8188"
OUTDIR = r"D:\comfyui\ComfyUI-master\output"

POS = ("masterpiece, best quality, very aesthetic, 1girl, solo, white dress, "
       "long white hair, standing in shallow water, soft lighting, full body, detailed eyes")
NEG = ""

W = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SAMPLER = sys.argv[2] if len(sys.argv) > 2 else "euler"
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "cmp256"

wf = {
    "1": {"class_type": "UNETLoader",
          "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader",
          "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
    "3": {"class_type": "VAELoader",
          "inputs": {"vae_name": "Qwen\\qwen_image_vae.safetensors"}},
    "6": {"class_type": "ModelSamplingAuraFlow",
          "inputs": {"model": ["1", 0], "shift": 3.0}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 0], "text": POS}},
    "8": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 0], "text": NEG}},
    "9": {"class_type": "EmptyLatentImage",
          "inputs": {"width": W, "height": W, "batch_size": 1}},
    "10": {"class_type": "KSampler",
           "inputs": {"model": ["6", 0], "positive": ["7", 0], "negative": ["8", 0],
                      "latent_image": ["9", 0],
                      "seed": random.randint(0, 2**31),
                      "steps": 30, "cfg": 5.0,
                      "sampler_name": SAMPLER, "scheduler": "simple", "denoise": 1.0}},
    "11": {"class_type": "VAEDecode",
           "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
    "12": {"class_type": "SaveImage",
           "inputs": {"images": ["11", 0], "filename_prefix": PREFIX}},
}

data = json.dumps({"prompt": wf, "client_id": "cmp"}).encode()
req = urllib.request.Request(SRV + "/prompt", data=data,
                             headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=30).read())
print("queued %s  W=%d sampler=%s" % (r.get("prompt_id"), W, SAMPLER))
