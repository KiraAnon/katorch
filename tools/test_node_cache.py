import json, urllib.request, time

API = "http://127.0.0.1:8188"


def build(seed, pos_text):
    return {
        "1": {"class_type": "KAtorchUNETLoader", "inputs": {"asset": "blobs512m"}},
        "2": {"class_type": "KAtorchCLIPLoader", "inputs": {"clip_name": "qwen3.safetensors"}},
        "3": {"class_type": "KAtorchVAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "KAtorchCLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": pos_text, "model": ["1", 0]}},
        "5": {"class_type": "KAtorchCLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": "worst quality", "model": ["1", 0]}},
        "6": {"class_type": "KAtorchEmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "7": {"class_type": "KAtorchKSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": 1, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}},
        "8": {"class_type": "KAtorchVAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "c1"}},
    }


def run(label, seed, text, wait=200):
    req = urllib.request.Request(API + "/prompt",
                                 data=json.dumps({"prompt": build(seed, text)}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=40)
        pid = json.loads(r.read().decode()).get("prompt_id")
    except urllib.error.HTTPError as e:
        print("%-40s 提交失败 %s" % (label, e.read().decode()[:200]))
        return
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            h = json.load(urllib.request.urlopen(API + "/history", timeout=20))
            if pid in h:
                print("%-40s %.1fs" % (label, time.time() - t0))
                return
        except Exception:
            pass
        time.sleep(2)
    print("%-40s 超时" % label)


# 用 1 步采样，只关心「编码节点有没有跑」
T1 = "a pink haired girl, test A"
T2 = "a blue haired boy, test B"

run("① 首次（text=A, seed=1）", 1, T1)
run("② 同 text 换 seed（应只重跑采样）", 2, T1)
run("③ 换 text（应重跑编码）", 1, T2)
run("④ 完全重复 ③（应全命中）", 1, T2)
