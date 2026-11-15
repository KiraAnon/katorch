import json, urllib.request, time

API = "http://127.0.0.1:8188"


def post(path, obj):
    req = urllib.request.Request(API + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return -1, str(e)


def build(clip_type, vae_name, w=512, h=512, steps=8, cfg=1.0):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "anima_v1.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3.safetensors", "type": clip_type}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "masterpiece, best quality, 1girl, silver hair", "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "lowres, bad anatomy", "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": 42, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple",
                         "positive": ["3", 0], "negative": ["4", 0],
                         "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["7", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "katorch"}},
    }


for ct in ("qwen_image", "flux2"):
    for vae in ("flux2-vae.safetensors", "qwen_image_vae.safetensors"):
        st, body = post("/prompt", {"prompt": build(ct, vae)})
        ok = '"prompt_id"' in body
        note = ""
        if not ok:
            try:
                j = json.loads(body)
                note = str(j.get("error") or j.get("node_errors") or "")[:130]
            except Exception:
                note = body[:130]
        print("clip=%-10s vae=%-30s -> %s %s" % (ct, vae, "提交成功" if ok else "失败", note))
        if ok:
            print("   命中组合: clip=%s vae=%s" % (ct, vae))
            break
    else:
        continue
    break
