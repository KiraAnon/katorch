"""用 turbo LoRA + 8 步 + cfg=1 在 GPU 上验证，作为 NPU 路线的目标参考。"""
import json, random, sys, urllib.request

SRV = "http://127.0.0.1:8188"
POS = ("masterpiece, best quality, very aesthetic, 1girl, solo, white dress, "
       "long white hair, standing in shallow water, soft lighting, full body, detailed eyes")
NEG = ""
W = 512
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CFG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TAG = sys.argv[3] if len(sys.argv) > 3 else "turbo8"

wf = {
    "1": {"class_type": "UNETLoader",
          "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader",
          "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
    "3": {"class_type": "VAELoader",
          "inputs": {"vae_name": "Qwen\\qwen_image_vae.safetensors"}},
    "4": {"class_type": "LoraLoaderModelOnly",
          "inputs": {"model": ["1", 0], "lora_name": "anima-turbo.safetensors",
                     "strength_model": 1.0}},
    "6": {"class_type": "ModelSamplingAuraFlow",
          "inputs": {"model": ["4", 0], "shift": 3.0}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": POS}},
    "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": NEG}},
    "9": {"class_type": "EmptyLatentImage",
          "inputs": {"width": W, "height": W, "batch_size": 1}},
    "10": {"class_type": "KSampler",
           "inputs": {"model": ["6", 0], "positive": ["7", 0], "negative": ["8", 0],
                      "latent_image": ["9", 0], "seed": 12345,
                      "steps": STEPS, "cfg": CFG, "sampler_name": "euler",
                      "scheduler": "simple", "denoise": 1.0}},
    "11": {"class_type": "VAEDecode",
           "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
    "12": {"class_type": "SaveImage",
           "inputs": {"images": ["11", 0], "filename_prefix": TAG}},
}
data = json.dumps({"prompt": wf, "client_id": "turbo"}).encode()
req = urllib.request.Request(SRV + "/prompt", data=data,
                             headers={"Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print("queued", r.get("prompt_id"), "steps=%d cfg=%.1f" % (STEPS, CFG))
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:1500])
