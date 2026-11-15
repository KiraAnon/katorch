import json, urllib.request

API = "http://127.0.0.1:8188"
POS = ("masterpiece, best quality, score_9, score_8, highres, absurdres, "
       "anime screenshot, official art, 1girl, solo, pink hair, blue eyes, "
       "@necomi, @banned artist")
NEG = ("worst quality, low quality, score_1, score_2, score_3, artist name, "
       "simple background, lowres, (bad), text, error, fewer, extra, missing, "
       "jpeg artifacts, watermark, unfinished, displeasing, signature, "
       "artistic error, username, scan, english text, shiny hair")


def post(label, prompt):
    req = urllib.request.Request(API + "/prompt",
                                 data=json.dumps({"prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=40)
        j = json.loads(r.read().decode())
        print("%-28s 提交成功 %s" % (label, j.get("prompt_id", "")[:8]))
        if j.get("node_errors"):
            print("   节点错误:", json.dumps(j["node_errors"], ensure_ascii=False)[:300])
        return True
    except urllib.error.HTTPError as e:
        print("%-28s 失败 %d" % (label, e.code))
        print("  ", e.read().decode()[:500])
        return False


# ---- 1. 完整原生版（800x1424，CPU）----
native = {
    "1": {"class_type": "UNETLoader",
          "inputs": {"unet_name": "anima_v1.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader",
          "inputs": {"clip_name": "qwen3.safetensors", "type": "qwen_image"}},
    "3": {"class_type": "VAELoader",
          "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
    "4": {"class_type": "LoraLoader",
          "inputs": {"model": ["1", 0], "clip": ["2", 0],
                     "lora_name": "turbo.safetensors",
                     "strength_model": 0.6, "strength_clip": 0.6}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": POS}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": NEG}},
    "8": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 800, "height": 1424, "batch_size": 1}},
    "9": {"class_type": "KSampler",
          "inputs": {"model": ["4", 0], "seed": 12345, "steps": 8, "cfg": 1.0,
                     "sampler_name": "er_sde", "scheduler": "beta",
                     "positive": ["6", 0], "negative": ["7", 0],
                     "latent_image": ["8", 0], "denoise": 1.0}},
    "10": {"class_type": "VAEDecode",
           "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
    "11": {"class_type": "SaveImage",
           "inputs": {"images": ["10", 0], "filename_prefix": "Anima手动/out"}},
}

# ---- 2. NPU 版 ----
npu = {
    "1": {"class_type": "KAtorchNPUGenerate",
          "inputs": {"prompt": POS, "negative": NEG, "seed": 12345,
                     "width": 512, "height": 512}},
    "2": {"class_type": "SaveImage",
          "inputs": {"images": ["1", 0], "filename_prefix": "Anima手动/npu"}},
}

ok1 = post("原生版 800x1424 (CPU)", native)
