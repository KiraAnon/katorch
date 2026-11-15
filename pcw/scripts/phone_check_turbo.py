"""检查 turbo.safetensors 的结构，判断是否适配 anima。"""
import json, struct

P = "/data/data/com.termux/files/home/ComfyUI/models/loras/turbo.safetensors"
with open(P, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))

meta = hdr.get("__metadata__", {})
print("=== metadata ===")
for k, v in meta.items():
    print("  %s: %s" % (k, str(v)[:120]))

keys = [k for k in hdr if k != "__metadata__"]
print("\n=== 张量数: %d ===" % len(keys))
print("=== 前 12 个名字 ===")
for k in keys[:12]:
    print("  %s  %s" % (k, hdr[k]["shape"]))

# 抽样分析：LoRA 的 A/B 矩阵形状
import re
suffix = set()
for k in keys:
    m = re.search(r"\.(lora_[AB]|alpha|diff|diff_b)$", k)
    suffix.add(m.group(1) if m else "其他")
print("\n=== 后缀种类 ===", suffix)

# 判断是 anima 还是 qwen 结构
has_anima = any("x_embedder" in k or "final_layer" in k for k in keys)
has_t_emb = any("t_embedder" in k for k in keys)
print("\n=== 结构判断 ===")
print("  含 x_embedder/final_layer: %s" % has_anima)
print("  含 t_embedder: %s" % has_t_emb)

# 统计 rank
ranks = {}
for k in keys:
    if k.endswith("lora_A") or k.endswith("lora_down"):
        ranks[tuple(hdr[k]["shape"])] = ranks.get(tuple(hdr[k]["shape"]), 0) + 1
print("\n=== lora_A 形状分布（前 5）===")
for sh, c in sorted(ranks.items(), key=lambda x: -x[1])[:5]:
    print("  %s x %d" % (list(sh), c))
