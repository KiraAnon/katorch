"""用 transformers 加载 Qwen3-0.6B 并编码提示词。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, time
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
from safetensors.torch import load_file

H = _KA_HOME
P = H + "/ComfyUI/models/clip/qwen3.safetensors"

print("=== 加载 Qwen3-0.6B ===", flush=True)
t0 = time.time()
sd = load_file(P, device="cpu")
print("  权重 %d keys, %.2f GB" % (len(sd), sum(v.numel()*v.element_size() for k,v in sd.items() if hasattr(v,'numel'))/1e9), flush=True)

from transformers import Qwen3Config, Qwen3ForCausalLM

cfg = Qwen3Config(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)
model = Qwen3ForCausalLM(cfg)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("  missing: %d, unexpected: %d" % (len(missing), len(unexpected)))
if missing[:5]: print("    missing 示例:", missing[:5])
if unexpected[:5]: print("    unexpected 示例:", unexpected[:5])
model.eval()
print("  加载 %.1fs" % (time.time()-t0), flush=True)

# tokenizer
from tokenizers import Tokenizer
import json
QTOK = H + "/ComfyUI/comfy/text_encoders/qwen25_tokenizer"
# Qwen2Tokenizer 是 BPE（vocab.json + merges.txt），tokenizers 库可以直接用
from tokenizers import Tokenizer as Tk
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, decoders
with open(QTOK + "/vocab.json") as f:
    vocab = json.load(f)
with open(QTOK + "/merges.txt") as f:
    raw = [l.strip() for l in f if l.strip() and not l.startswith("#version")]
merges = [tuple(r.split(" ")) for r in raw if " " in r]
tk = Tk(BPE(vocab=vocab, merges=merges, unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
print("  Qwen tokenizer 词表: %d" % tk.get_vocab_size())

text = "a cat sitting on a mat, masterpiece, best quality"
enc = tk.encode(text)
print("\n=== 编码 ===")
print("  text: %s" % text)
print("  ids: %s (len=%d)" % (enc.ids, len(enc.ids)))

ids = torch.tensor([enc.ids])
t0 = time.time()
with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True)
hs = out.hidden_states[-1]     # 最后一层
print("  hidden: %s  std=%.4f  耗时 %.2fs" % (tuple(hs.shape), hs.std(), time.time()-t0))

# 保存
hs.numpy().astype(np.float16).tofile(H + "/npt/e2e/o_qwen_hs.bin")
np.array(enc.ids, dtype=np.int32).tofile(H + "/npt/e2e/o_qwen_ids.bin")
print("  已存 o_qwen_hs.bin (%s) / o_qwen_ids.bin" % (hs.shape,))

# 顺便验证：只用最后一层 vs 平均
print("\n=== 参考：最后一层 logits 的 top-5 ===")
with torch.no_grad():
    lg = model(input_ids=ids).logits
top5 = lg[0, -1].topk(5)
print("  top5 ids: %s" % top5.indices.tolist())
