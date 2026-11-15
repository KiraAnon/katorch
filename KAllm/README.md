# KAllm

LLM 文本编码模块。

## 组成

```
文本 → Qwen3-0.6B → hidden states [1, L, 1024]
                    ↓
         T5 tokenizer → ids [1, L2]
                    ↓
     LLMAdapter（cross-attn 融合）→ context [1, 512, 1024]
```

## 文件

| 文件 | 作用 |
|---|---|
| `qwen3.py` | Qwen3-0.6B 加载（用 transformers） |
| `llm_adapter.py` | **LLMAdapter**：6 层 self-attn + cross-attn，134.66M 参数 |
| `text_encoder.py` | 完整链路 + T5 tokenizer |

## 关键参数（踩过的坑）

| 项 | 值 | 说明 |
|---|---|---|
| Qwen3 `rope_theta` | **1000000** | 默认 10000 是错的 |
| Qwen3 层数 | 28 | 权重格式是标准 HF |
| LLMAdapter 层数 | 6 | |
| LLMAdapter 词表 | 32128 | T5 v1.1；ComfyUI 自带的是 32100，够用 |
| context 长度 | 512 | 不足时 pad |

## 依赖

- `transformers`（加载 Qwen3）
- `tokenizers`（T5 + Qwen2 BPE）
- **必须 `LD_PRELOAD` libpython**，否则扩展模块符号解析失败

## 性能

- Qwen3 加载 15s，推理 1.2s
- LLMAdapter 0.28s
- **总计约 1.5s**（CPU，rel 很小，不需要上 NPU）
