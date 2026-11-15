import json, urllib.request
oi = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=40))
print("总节点数:", len(oi))
k = oi.get("KAtorchNPUGenerate")
if k:
    print("KAtorch NPU 节点已注册 ✓")
    print("  分类:", k.get("category"))
    print("  输入:", list(k.get("input", {}).get("required", {}).keys()))
    print("  输出:", k.get("output_name"))
else:
    print("未找到 KAtorchNPUGenerate")
    print("含 katorch 的节点:", [n for n in oi if "katorch" in n.lower() or "KAtorch" in n])
