"""按「Anima 正式版工作流」的参数提交一次生成，用于对比质量。
   构图刻意与 NPU 版（256x256）保持一致，只比较质量差异。
"""
import json
import urllib.request
import random

SRV = "http://127.0.0.1:8188"

POS = ("masterpiece,very aesthetic,best quality, score_9, score_8,highres, absurdres, "
       "anime screenshot, official art, "
       "1girl, solo, white dress, long white hair, standing in shallow water, "
       "soft lighting, full body, centered, sunlight from above, water reflection, "
       "detailed face, beautiful detailed eyes")

NEG = ("lowres,bad,error,fewer,extra,missing,worst quality,jpeg artifacts,bad quality,"
       "watermark,signature,extra digits,artistic error,username,abstract,artist logo,"
       "artist name,bad fingers,bad hand,simple background,"
       "(worst quality, low quality:1.4),(deformed, distorted, disfigured:1.3),"
       "poorly drawn hands,extra fingers,mutated,text,watermark,signature,"
       "messy drawing,doll,plastic skin,bad anatomy,inverted limbs,"
       "missing leg,missing arm,extra leg,extra arm,crowd,flat color,deformed")

wf = {
    "1": {"class_type": "UNETLoader",
          "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader",
          "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
    "3": {"class_type": "VAELoader",
          "inputs": {"vae_name": "Qwen\\qwen_image_vae.safetensors"}},

    # 两个美学 LoRA
    "4": {"class_type": "LoraLoaderModelOnly",
          "inputs": {"model": ["1", 0],
                     "lora_name": "anima-highres-aesthetic-boost.safetensors",
                     "strength_model": 1.0}},
    "5": {"class_type": "LoraLoaderModelOnly",
          "inputs": {"model": ["4", 0],
                     "lora_name": "anima-preview-3-masterpieces-v5.safetensors",
                     "strength_model": 1.0}},

    # ModelSamplingAuraFlow: shift = 3
    "6": {"class_type": "ModelSamplingAuraFlow",
          "inputs": {"model": ["5", 0], "shift": 3.0}},

    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 0], "text": POS}},
    "8": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 0], "text": NEG}},
    "9": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1536, "batch_size": 1}},

    "10": {"class_type": "KSampler",
           "inputs": {"model": ["6", 0],
                      "positive": ["7", 0],
                      "negative": ["8", 0],
                      "latent_image": ["9", 0],
                      "seed": random.randint(0, 2**31),
                      "steps": 30,
                      "cfg": 5.0,
                      "sampler_name": "er_sde",
                      "scheduler": "simple",
                      "denoise": 1.0}},
    "11": {"class_type": "VAEDecode",
           "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
    "12": {"class_type": "SaveImage",
           "inputs": {"images": ["11", 0], "filename_prefix": "anima_ref"}},
}

data = json.dumps({"prompt": wf, "client_id": "kat"}).encode()
req = urllib.request.Request(SRV + "/prompt", data=data,
                             headers={"Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print("queued:", r)
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:3000])
