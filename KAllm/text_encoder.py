"""完整文本编码：Qwen3 + T5 + LLMAdapter -> context [1, 512, 1024]。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, time, json
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn.functional as F
from safetensors.torch import load_file

H = _KA_HOME
sys.path.insert(0, H + "/npt")

# 复用 LLMAdapter 的定义
exec(open(H + "/npt/t_llmadapter2.py").read().split("# ---- 加载权重 ----")[0])

print("=== 加载模型 ===", flush=True)
t0 = time.time()
anima_sd = load_file(H + "/ComfyUI/models/diffusion_models/anima_v1.safetensors", device="cpu")
adapter = LLMAdapter(anima_sd).eval()
print("  LLMAdapter %.1fs" % (time.time()-t0), flush=True)

t0 = time.time()
qsd = load_file(H + "/ComfyUI/models/clip/qwen3.safetensors", device="cpu")
from transformers import Qwen3Config, Qwen3ForCausalLM
# rope_theta 必须显式给：transformers 默认是 10000，ComfyUI 给 Qwen3-0.6B 写死的是 1000000。
# 差 100 倍会让 hidden 偏 10.8%，最终 context 偏 6.7%。
cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                  num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                  head_dim=128, max_position_embeddings=40960, rms_norm_eps=1e-6,
                  tie_word_embeddings=True, rope_theta=1000000.0)
qwen = Qwen3ForCausalLM(cfg)
qwen.load_state_dict(qsd, strict=False)
qwen.eval()
print("  Qwen3 %.1fs" % (time.time()-t0), flush=True)

# tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
# Qwen2 (BPE)
QTOK = H + "/ComfyUI/comfy/text_encoders/qwen25_tokenizer"
with open(QTOK + "/vocab.json") as f: vocab = json.load(f)
with open(QTOK + "/merges.txt") as f:
    raw = [l.strip() for l in f if l.strip() and not l.startswith("#version")]
merges = [tuple(r.split(" ")) for r in raw if " " in r]
qtk = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None))
qtk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
qtk.decoder = decoders.ByteLevel()

# T5
ttk = Tokenizer.from_file(H + "/ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json")
print("  tokenizers: qwen %d, t5 %d" % (qtk.get_vocab_size(), ttk.get_vocab_size()))

# ---- 完整编码 ----
text = "a cat sitting on a mat, masterpiece, best quality"
print("\n=== 编码: %s ===" % text, flush=True)
qids = qtk.encode(text).ids
tids = ttk.encode(text).ids
print("  qwen ids (len=%d): %s" % (len(qids), qids))
print("  t5   ids (len=%d): %s" % (len(tids), tids))

t0 = time.time()
with torch.no_grad():
    hs = qwen(input_ids=torch.tensor([qids]), output_hidden_states=True).hidden_states[-1]
t_q = time.time() - t0
print("  Qwen3 hidden: %s std=%.3f  %.2fs" % (tuple(hs.shape), hs.std(), t_q))

t0 = time.time()
with torch.no_grad():
    ctx = adapter(hs, torch.tensor([tids]))
t_a = time.time() - t0
print("  Adapter:      %s std=%.4f  %.2fs" % (tuple(ctx.shape), ctx.std(), t_a))

# pad 到 512
if ctx.shape[1] < 512:
    ctx = F.pad(ctx, (0, 0, 0, 512 - ctx.shape[1]))
print("  pad:          %s" % (tuple(ctx.shape),))
print("\n  总耗时 %.2fs (Qwen %.2fs + Adapter %.2fs)" % (t_q + t_a, t_q, t_a))

ctx.numpy().astype(np.float16).tofile(H + "/npt/e2e2/o_ctx512.bin")
print("  已存 o_ctx512.bin")
