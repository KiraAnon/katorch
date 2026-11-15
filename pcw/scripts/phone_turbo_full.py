"""完整分析 turbo LoRA：覆盖哪些模块、有没有 alpha、缩放系数怎么算。"""
import json, struct, re
from collections import Counter

P = "/data/data/com.termux/files/home/ComfyUI/models/loras/turbo.safetensors"
with open(P, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))

keys = [k for k in hdr if k != "__metadata__"]

print("=== 有没有 alpha ===")
alphas = [k for k in keys if "alpha" in k.lower()]
print("  alpha 张量数: %d" % len(alphas))
print("  metadata: %s" % hdr.get("__metadata__", {}))

print("\n=== 模块类型统计 ===")
pat = Counter()
for k in keys:
    m = re.match(r"diffusion_model\.blocks\.\d+\.(.+?)\.lora_[AB]\.weight", k)
    if m:
        pat[m.group(1)] += 1
    else:
        pat["(其他) " + k[:50]] += 1
for name, c in sorted(pat.items()):
    print("  %-45s %d" % (name, c))

print("\n=== 覆盖到哪些 block ===")
blk = set()
for k in keys:
    m = re.match(r"diffusion_model\.blocks\.(\d+)\.", k)
    if m:
        blk.add(int(m.group(1)))
print("  block 编号: %s" % sorted(blk))

print("\n=== rank 分布 ===")
ranks = Counter()
for k in keys:
    if ".lora_A." in k:
        ranks[hdr[k]["shape"][0]] += 1
print("  rank: %s" % dict(ranks))

print("\n=== 抽样：B 矩阵形状 ===")
cnt = 0
for k in keys:
    if ".lora_B." in k:
        print("  %-62s %s" % (k, hdr[k]["shape"]))
        cnt += 1
        if cnt >= 6:
            break
