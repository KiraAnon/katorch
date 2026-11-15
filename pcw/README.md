# pcw — NPU 生图实验工作区

> 2026-11-14 整理。这是 DiT / VAE 的**历史实验记录**：脚本里写死了作者本机的路径
> （`F:\KAtorch\...`、`D:\comfyui\...`、手机上的 `/data/data/com.termux/...`），
> 换机器用之前要自己改。现代码结构和结论见 [docs/performance.md](../docs/performance.md)。
>
> 以下目录只在作者机器上存在，**不随仓库发布**：`headers/`（厂商头文件）、
> `pymod/`、`ref/`（与 KAcustom/KAdit 重复的副本）、`ctx/`、`bin/`、`out/`（张量与产物）。

## 目录

| 目录 | 内容 |
|---|---|
| `scripts/` | 实验脚本（`.py` 是可复现部分；`_*.sh` 和调度壳不入库） |

## 最有用的三个文件

```
scripts/make_ctx2.py      正确的 ctx 生成：Qwen3 → LLMAdapter(6层) → 512 零填充
scripts/gen_image_cfg.py  带 CFG 的生图主脚本（KA_CFG 环境变量控制）
scripts/phone_verify_b00.py  NPU 块 vs PyTorch 对拍
```

## 关键结论

1. **PC 编译产物在真机上等价**：缓存推过去，出图与手机自编逐字节相同
2. **ctx 必须过 LLMAdapter**：漏掉这 6 层，喂什么都是噪声
3. **CFG 不需要重新编译**：ctx 是运行时输入，每步跑两遍即可
4. **NPU 块实现忠实**：对拍相对误差 0.26%
5. **画质上限由 token 数决定**：256 token 装不下波纹、倒影这类细节

## 环境注意

- WSL 里跑 qemu 必须加 `-R 0x10000000000`，否则 scudo 的 MAP_FIXED 全失败
- PC 编译用 `comfy` conda 环境（torch 2.10 + ROCm）
- 手机的 Python 在 `/data/data/com.termux/files/usr/bin/python`
- 模型权重：PC 在 `D:\comfyui\ComfyUI-master\models\`，手机在 `~/ComfyUI/models/`
- **VAE 要用 `Qwen/qwen_image_vae.safetensors`**，不是 `moss/flux2-vae`（后者 128 通道，会报错）

## 未完成

- 对比度偏低（x.std=0.634 vs 应 ~1.0），根因未定位
- PC 端 `pe` 编译触发 GraphBridging 崩溃
- 28 层整块：14 层已超 109 分钟
- LoRA 未合并
