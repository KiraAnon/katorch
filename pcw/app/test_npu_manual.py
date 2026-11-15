import json, urllib.request, time

API = "http://127.0.0.1:8188"

# 中断当前
try:
    req = urllib.request.Request(API + "/interrupt", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    print("已中断原生版任务（800x1424 在 CPU 上太慢）")
except Exception as e:
    print("中断失败:", e)

time.sleep(2)

POS = ("masterpiece, best quality, score_9, score_8, highres, absurdres, "
       "anime screenshot, official art, 1girl, solo, pink hair, blue eyes, "
       "@necomi, @banned artist")
NEG = ("worst quality, low quality, score_1, score_2, score_3, artist name, "
       "simple background, lowres, (bad), text, error, fewer, extra, missing, "
       "jpeg artifacts, watermark, unfinished, displeasing, signature, "
       "artistic error, username, scan, english text, shiny hair")

npu = {
    "1": {"class_type": "KAtorchNPUGenerate",
          "inputs": {"prompt": POS, "negative": NEG, "seed": 12345,
                     "width": 512, "height": 512}},
    "2": {"class_type": "SaveImage",
          "inputs": {"images": ["1", 0], "filename_prefix": "Anima手动/npu"}},
}
req = urllib.request.Request(API + "/prompt",
                             data=json.dumps({"prompt": npu}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=40)
    j = json.loads(r.read().decode())
    print("NPU 版提交成功:", j.get("prompt_id"))
except urllib.error.HTTPError as e:
    print("NPU 版失败:", e.code, e.read().decode()[:400])
