"""验证 LoRA 合并：同一个权重，合与不合应有明显差异。"""
import os, sys
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

LORA = "/data/data/com.termux/files/home/ComfyUI/models/loras/turbo.safetensors"

NAMES = ["net.blocks.0.self_attn.q_proj.weight",
         "net.blocks.0.mlp.layer1.weight",
         "net.final_layer.linear.weight"]

os.environ.pop("KA_MERGE_LORA", None)
from anima_block import load as load_plain
plain = {n: load_plain(n).clone() for n in NAMES}

# 强制重载模块，让 load 看到新的环境变量
for m in list(sys.modules):
    if m in ("anima_block", "lora_merge"):
        del sys.modules[m]
os.environ["KA_MERGE_LORA"] = LORA
from anima_block import load as load_merged
merged = {n: load_merged(n) for n in NAMES}

print("%-45s %10s %10s %10s" % ("权重", "原始std", "合并后std", "相对变化"))
for n in NAMES:
    a, b = plain[n], merged[n]
    rel = (b - a).norm() / (a.norm() + 1e-12)
    print("%-45s %10.6f %10.6f %9.2f%%" % (n.split(".", 2)[-1], a.std(), b.std(), rel * 100))

print("\n=== 覆盖统计 ===")
from lora_merge import lora_for
hit = sum(1 for n in NAMES if lora_for(n) is not None)
print("  测试的 %d 个权重中，%d 个命中 LoRA" % (len(NAMES), hit))
