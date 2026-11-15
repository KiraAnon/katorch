"""在 PC 上校验纯 Python tokenizer 与 ComfyUI 官方结果是否一致。"""
import sys
sys.path.insert(0, r"D:\comfyui\ComfyUI-master")
sys.path.insert(0, r"F:\KAtorch\pcw\app")

from tok import QwenTok, T5Tok

CM = r"D:\comfyui\ComfyUI-master\comfy\text_encoders"
qwen = QwenTok(CM + r"\qwen25_tokenizer\vocab.json", CM + r"\qwen25_tokenizer\merges.txt")
t5 = T5Tok(CM + r"\t5_tokenizer\tokenizer.json")

from transformers import Qwen2Tokenizer, T5TokenizerFast
import os
qwen_ref = Qwen2Tokenizer.from_pretrained(CM + r"\qwen25_tokenizer")
t5_ref = T5TokenizerFast.from_pretrained(CM + r"\t5_tokenizer")

TESTS = [
    "1girl, white dress, water",
    "masterpiece, best quality, very aesthetic, 1girl, solo, white dress, long white hair, standing in shallow water, soft lighting, full body, detailed eyes",
    "Hello, world! 测试中文 & punctuation.",
    "a cute cat, (best quality:1.2), [smile]",
]

ok_q = ok_t = 0
for t in TESTS:
    a = qwen.encode(t)
    b = qwen_ref.encode(t, add_special_tokens=False)
    same_q = (a == b)
    ok_q += same_q
    print("\n文本: %s" % t[:44])
    print("  Qwen 我 : %s" % a[:12])
    print("  Qwen 官 : %s" % b[:12])
    print("  Qwen 一致: %s" % ("是" if same_q else "否"))
    if not same_q:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print("    首个差异 idx=%d 我=%d 官=%d" % (i, x, y))
                break

    c = t5.encode(t, add_eos=False)
    d = t5_ref.encode(t, add_special_tokens=False)
    same_t = (c == d)
    ok_t += same_t
    print("  T5 我 : %s" % c[:12])
    print("  T5 官 : %s" % d[:12])
    print("  T5 一致: %s" % ("是" if same_t else "否"))

print("\n汇总: Qwen %d/%d, T5 %d/%d" % (ok_q, len(TESTS), ok_t, len(TESTS)))
