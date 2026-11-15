import json, urllib.request

r = urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=40)
oi = json.load(r)

d = oi.get("CLIPLoader", {})
req = d.get("input", {}).get("required", {})
print("=== CLIPLoader type 完整列表 ===")
t = req.get("type", [])
if isinstance(t, list) and t and isinstance(t[0], list):
    print(t[0])
else:
    print(t)

# 找所有跟 anima 有关的节点
print()
print("=== 含 anima 的节点 ===")
for k, v in oi.items():
    s = json.dumps(v, ensure_ascii=False).lower()
    if "anima" in s:
        print("  ", k)

print()
print("=== 所有 Loader 类节点的名字 ===")
for k in oi:
    if "Loader" in k or "Load" in k:
        print("  ", k)
