"""把「Anima手动-带画师串-8步.json」适配到手机。

原流 11 个节点全是原生节点，只需替换模型文件名。
另外额外提供一个 NPU 加速变体（原版 KSampler 在 CPU 上 800x1424 要 30-40 分钟）。
"""
import json, os

SRC = r"D:\comfyui\ComfyUI-master\user\default\workflows\Anima手动-带画师串-8步.json"
OUTDIR = os.path.dirname(os.path.abspath(__file__))

wf = json.loads(open(SRC, "rb").read().decode("utf-8-sig"))

# 模型名替换
RENAME = {
    "anima-base-v1.0.safetensors": "anima_v1.safetensors",
    "qwen_3_06b_base.safetensors": "qwen3.safetensors",
    "anima-turbo-lora-v0.2.safetensors": "turbo.safetensors",
}

changed = []
for n in wf.get("nodes", []):
    t = n.get("type")
    wv = n.get("widgets_values")
    if not isinstance(wv, list) or not wv:
        continue
    if t in ("UNETLoader", "VAELoader", "LoraLoader", "CLIPLoader"):
        if wv[0] in RENAME:
            changed.append("%s: %s -> %s" % (t, wv[0], RENAME[wv[0]]))
            wv[0] = RENAME[wv[0]]
    if t == "CLIPLoader" and len(wv) >= 2:
        # 手机端没有 stable_diffusion 对应 anima，实测要 qwen_image
        if wv[1] != "qwen_image":
            changed.append("CLIPLoader type: %s -> qwen_image" % wv[1])
            wv[1] = "qwen_image"

print("替换项:")
for c in changed:
    print("  ", c)

# 保存手机版（保持 800x1424 原尺寸，用原生 KSampler）
out1 = os.path.join(OUTDIR, "anima_manual_phone.json")
json.dump(wf, open(out1, "w"), ensure_ascii=False, indent=1)
print("\n已生成 %s (%.0f 字节)" % (out1, os.path.getsize(out1)))

# ---------- 变体：NPU 加速 ----------
nodes = wf.get("nodes", [])
links = wf.get("links", [])
note = next((n for n in nodes if n.get("type") == "Note"), None)
pos_text = next((n for n in nodes if n.get("type") == "CLIPTextEncode" and "正向" in (n.get("title") or "")), None)
pos_str = (pos_text or {}).get("widgets_values", [""])[0]
neg_text = next((n for n in nodes if n.get("type") == "CLIPTextEncode" and "负向" in (n.get("title") or "")), None)
neg_str = (neg_text or {}).get("widgets_values", [""])[0]

npu_nodes = [
    {"id": 1, "type": "KAtorchNPUGenerate", "pos": [80, 100], "size": [440, 280],
     "flags": {}, "order": 0, "mode": 0, "inputs": [],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [1], "slot_index": 0}],
     "properties": {"Node name for S&R": "KAtorchNPUGenerate"},
     "widgets_values": [pos_str, neg_str, 12345, 512, 512],
     "title": "KAtorch NPU 生图（原版参数内嵌，约2分钟）"},
    {"id": 2, "type": "SaveImage", "pos": [580, 100], "size": [420, 460],
     "flags": {}, "order": 1, "mode": 0,
     "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
     "outputs": [],
     "properties": {"Node name for S&R": "SaveImage"},
     "widgets_values": ["Anima手动/out"], "title": "保存图像"},
    {"id": 3, "type": "Note", "pos": [80, 420], "size": [440, 340],
     "flags": {}, "order": 2, "mode": 0, "inputs": [], "outputs": [],
     "properties": {"Node name for S&R": "Note"},
     "widgets_values": [
         "【NPU 加速版】\n\n"
         "原版工作流 800x1424 在 CPU 上要 30-40 分钟，"
         "这个版本把整条采样交给 NPU，512x512 约 2 分钟。\n\n"
         "正向 / 负向提示词已经内嵌进节点（含画师串 @necomi, @banned artist）。\n"
         "改画师直接改节点里的 prompt 文本框。\n\n"
         "想要原尺寸就加载 anima_manual_phone.json（走 CPU，很慢）。"
     ], "title": "说明"},
]
npu_wf = {"last_node_id": 3, "last_link_id": 1, "nodes": npu_nodes,
          "links": [[1, 1, 0, 2, 0, "IMAGE"]], "groups": [], "config": {},
          "extra": {"ds": {"scale": 0.8, "offset": [40, 30]}}, "version": 0.4}
out2 = os.path.join(OUTDIR, "anima_manual_npu.json")
json.dump(npu_wf, open(out2, "w"), ensure_ascii=False, indent=1)
print("已生成 %s (%.0f 字节)" % (out2, os.path.getsize(out2)))
