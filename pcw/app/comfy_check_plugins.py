import json, urllib.request

oi = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=60))
print("总节点数:", len(oi))

want = ["CR Text", "Seed (rgthree)", "Context (rgthree)", "AnimaV4RandomPrompt",
        "StringConcatenate", "LoraLoader", "UNETLoader", "CLIPLoader", "VAELoader",
        "KSampler", "EmptyLatentImage", "SaveImage", "CLIPTextEncode", "Note"]
print()
for w in want:
    print("  %-24s %s" % (w, "OK" if w in oi else "缺失"))

# 新出现的节点（对比之前的 1032）
base = {"KSampler", "CLIPTextEncode"}
print()
print("=== 含 Concatenate / Anima / rgthree 的节点 ===")
for k in oi:
    if "Concatenate" in k or "AnimaV4" in k or "rgthree" in k or k.startswith("CR "):
        print("  ", k)
