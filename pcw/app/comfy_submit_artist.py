import json, urllib.request

API = "http://127.0.0.1:8188"

ARTISTS = ("@chigusa minori, @yoneyama mai, @mochizuki kei, @hiten, "
           "@wlop, @takeuchi takashi, @sos adult, @kobinbin")
QUALITY = ("masterpiece, best quality, score_9, score_8, highres, absurdres, "
           "anime screenshot, official art")
NEG = ("worst quality, low quality, score_1, score_2, score_3, artist name, "
       "simple background, lowres, (bad), text, error, fewer, extra, missing, "
       "jpeg artifacts, watermark, unfinished, displeasing, oldest, early, "
       "signature, artistic error, username, scan, [abstract], english text, "
       "shiny hair, extra toes")

prompt = {
    "1": {"class_type": "UNETLoader",
          "inputs": {"unet_name": "anima_v1.safetensors", "weight_dtype": "default"}},
    "3": {"class_type": "CLIPLoader",
          "inputs": {"clip_name": "qwen3.safetensors", "type": "qwen_image"}},
    "2": {"class_type": "LoraLoader",
          "inputs": {"model": ["1", 0], "clip": ["3", 0],
                     "lora_name": "turbo.safetensors",
                     "strength_model": 0.6, "strength_clip": 0.6}},
    "4": {"class_type": "VAELoader",
          "inputs": {"vae_name": "qwen_image_vae.safetensors"}},

    "5": {"class_type": "AnimaV4RandomPrompt",
          "inputs": {"nsfw_level": "sfw", "seed": -1, "tag_cap": "自动",
                     "base_prob": 0.65, "disabled_sections": "", "lock_fields": "",
                     "force_role": "", "weights_json": "", "extra_body": "",
                     "appearance_follow": True, "selected_roles": "",
                     "big_breasts": False, "role_works": "", "breast_size": "不指定",
                     "outfit_style": "随机", "role_fav_only": False,
                     "hair_follow": True, "hair_len_pick": "角色标签发型",
                     "hair_tie_pick": "角色标签发型", "hair_curl_pick": "角色标签发型",
                     "hair_acc_pick": "角色标签发型"}},

    "6": {"class_type": "StringConcatenate",
          "inputs": {"string_a": ARTISTS, "string_b": ["5", 0], "delimiter": ", "}},
    "7": {"class_type": "StringConcatenate",
          "inputs": {"string_a": QUALITY, "string_b": ["6", 0], "delimiter": ", "}},

    "8": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 1], "text": ["7", 0]}},
    "9": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 1], "text": NEG}},

    "10": {"class_type": "EmptyLatentImage",
           "inputs": {"width": 800, "height": 1424, "batch_size": 1}},
    "11": {"class_type": "KSampler",
           "inputs": {"model": ["2", 0], "seed": 42, "steps": 8, "cfg": 1.0,
                      "sampler_name": "er_sde", "scheduler": "beta",
                      "positive": ["8", 0], "negative": ["9", 0],
                      "latent_image": ["10", 0], "denoise": 1.0}},
    "12": {"class_type": "VAEDecode",
           "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
    "13": {"class_type": "SaveImage",
           "inputs": {"images": ["12", 0], "filename_prefix": "katorch_artist"}},
}

req = urllib.request.Request(API + "/prompt", data=json.dumps({"prompt": prompt}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=40)
    body = r.read().decode()
    j = json.loads(body)
    print("提交成功:", j.get("prompt_id"))
    if j.get("node_errors"):
        print("节点错误:", json.dumps(j["node_errors"], ensure_ascii=False)[:500])
except urllib.error.HTTPError as e:
    print("提交失败:", e.code)
    print(e.read().decode()[:1500])
