"""生成用 KAtorch NPU 节点的 ComfyUI 工作流（UI 格式 + 直接提交测试）。"""
import json, os, urllib.request

POS = "masterpiece, best quality, very aesthetic, 1girl, silver hair, blue eyes, white dress, standing in water, moonlight"
NEG = "lowres, bad anatomy, bad hands, worst quality, low quality"

# ---------- UI 格式 ----------
nodes = [
    {"id": 1, "type": "KAtorchNPUGenerate", "pos": [80, 120], "size": [420, 260],
     "flags": {}, "order": 0, "mode": 0,
     "inputs": [],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [1], "slot_index": 0}],
     "properties": {"Node name for S&R": "KAtorchNPUGenerate"},
     "widgets_values": [POS, NEG, 42, 512, 512], "title": "KAtorch NPU 生图"},
    {"id": 2, "type": "SaveImage", "pos": [560, 120], "size": [420, 460],
     "flags": {}, "order": 1, "mode": 0,
     "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
     "outputs": [],
     "properties": {"Node name for S&R": "SaveImage"},
     "widgets_values": ["katorch"], "title": "保存图像"},
]
links = [[1, 1, 0, 2, 0, "IMAGE"]]

wf = {"last_node_id": 2, "last_link_id": 1, "nodes": nodes, "links": links,
      "groups": [], "config": {}, "extra": {"ds": {"scale": 0.9, "offset": [60, 40]}},
      "version": 0.4}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katorch_npu.json")
json.dump(wf, open(out, "w"), ensure_ascii=False, indent=1)
print("工作流已生成: %s (%d 字节)" % (out, os.path.getsize(out)))

# ---------- API 格式，直接提交 ----------
api = {
    "1": {"class_type": "KAtorchNPUGenerate",
          "inputs": {"prompt": POS, "negative": NEG, "seed": 42,
                     "width": 512, "height": 512}},
    "2": {"class_type": "SaveImage",
          "inputs": {"images": ["1", 0], "filename_prefix": "katorch_npu"}},
}
req = urllib.request.Request("http://127.0.0.1:8188/prompt",
                             data=json.dumps({"prompt": api}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=30)
    print("提交结果:", r.status, r.read().decode()[:120])
except urllib.error.HTTPError as e:
    print("提交失败:", e.code, e.read().decode()[:300])
