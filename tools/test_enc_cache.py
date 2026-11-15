import json, urllib.request, time

API = "http://127.0.0.1:8188"

POS = ("masterpiece, best quality, score_9, score_8, highres, absurdres, "
       "anime screenshot, official art, 1girl, solo, pink hair, blue eyes, "
       "@necomi, @banned artist")
NEG = ("worst quality, low quality, score_1, score_2, score_3, artist name, "
       "simple background, lowres, (bad), text, error, fewer, extra, missing, "
       "jpeg artifacts, watermark, unfinished, displeasing, signature, "
       "artistic error, username, scan, english text, shiny hair")


def build(seed):
    return {
        "1": {"class_type": "KAtorchUNETLoader",
              "inputs": {"asset": "blobs512m"}},
        "2": {"class_type": "KAtorchCLIPLoader",
              "inputs": {"clip_name": "qwen3.safetensors"}},
        "3": {"class_type": "KAtorchVAELoader",
              "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "KAtorchCLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": POS, "model": ["1", 0]}},
        "5": {"class_type": "KAtorchCLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": NEG, "model": ["1", 0]}},
        "6": {"class_type": "KAtorchEmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "7": {"class_type": "KAtorchKSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": 8, "cfg": 1.0,
                         "sampler_name": "er_sde", "scheduler": "beta", "denoise": 1.0}},
        "8": {"class_type": "KAtorchVAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "cachetest"}},
    }


def submit(seed):
    req = urllib.request.Request(API + "/prompt",
                                 data=json.dumps({"prompt": build(seed)}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=40)
        j = json.loads(r.read().decode())
        return j.get("prompt_id"), j.get("node_errors")
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:600]


def wait_done(pid, timeout=500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.load(urllib.request.urlopen(API + "/history", timeout=20))
            if pid in h:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    return None


print("=== 第 1 次（seed=111）—— 编码不命中 ===")
pid, err = submit(111)
if not pid:
    print("  提交失败:", err)
    raise SystemExit(1)
print("  prompt_id:", pid)
dt1 = wait_done(pid)
print("  完成 %.1fs" % (dt1 or -1))

print()
print("=== 第 2 次（同 prompt，seed=222）—— 编码应命中 ===")
pid2, err2 = submit(222)
if not pid2:
    print("  提交失败:", err2)
    raise SystemExit(1)
print("  prompt_id:", pid2)
dt2 = wait_done(pid2)
print("  完成 %.1fs" % (dt2 or -1))

if dt1 and dt2:
    print()
    print("=== 对比 ===")
    print("  第 1 次: %.1fs" % dt1)
    print("  第 2 次: %.1fs" % dt2)
    print("  省下:   %.1fs" % (dt1 - dt2))
