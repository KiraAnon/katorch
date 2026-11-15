# KAtorch 模块化 ComfyUI 节点

手机 NPU（MediaTek MDLA）出图的**标准 ComfyUI 节点组**，替代原来那个把整条管线
包成一个黑盒的 `KAtorchNPUGenerate`。

## 设计：三层

```
资产层 (assets.py)     扫描 blob 目录 + 校准文件 → manifest.json
    ↓
运行时层              host.py（NPU 宿主会话） dit.py（DiT 前向）
                      samplers.py（采样器/调度器） enc.py（文本编码） vae_*.py（VAE）
    ↓
节点层 (nodes.py)     标准的 ComfyUI 节点，一个节点只做一件事
```

## 节点清单

| 节点 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `KAtorchUNETLoader` | 模型名 | `KAT_MODEL` | 查 manifest，绑定 blob 集 + 分辨率 |
| `KAtorchCLIPLoader` | 编码器名 | `KAT_CLIP` | Qwen3-0.6B + LLMAdapter 句柄 |
| `KAtorchCLIPTextEncode` | clip, 文本 | `KAT_COND` | prompt → ctx（512×1024 fp16） |
| `KAtorchVAELoader` | vae 名 | `KAT_VAE` | Wan2.1 VAE 句柄（CPU） |
| `KAtorchEmptyLatentImage` | 宽/高/批次 | `LATENT` | 标准 latent dict |
| `KAtorchKSampler` | model/正负条件/latent/参数 | `LATENT` | ★ 核心：NPU 逐块采样 |
| `KAtorchVAEDecode` | samples, vae | `IMAGE` | latent → 图（标准 IMAGE，能直连 SaveImage） |
| `KAtorchVAEEncode` | pixels, vae | `LATENT` | 图生图 |
| `KAtorchLoraLoader` | model, lora 名 | `KAT_MODEL` | 预合并 LoRA 组合（查 manifest） |
| `KAtorchModelInfo` | model | 文本 | 显示资产详情、显存/IOVA 提示 |
| `KAtorchFreeMemory` | — | — | 关掉 host 会话，回收 IOVA |
| `KAtorchNPUGenerate` | prompt 等 | `IMAGE` | 兼容旧工作流的黑盒节点 |

## 典型工作流（和 ComfyUI 原生长得一样）

```
KAtorchUNETLoader ──KAT_MODEL───────────────────────┐
KAtorchCLIPLoader ──KAT_CLIP──┐                     │
KAtorchCLIPTextEncode ──┐     │                     ▼
KAtorchCLIPTextEncode ──┼─────┴──────────────► KAtorchKSampler ──LATENT──► KAtorchVAEDecode ──► SaveImage
KAtorchEmptyLatentImage ┘                            ▲
KAtorchVAELoader ────────────────────────────────────┘
```

## 和「黑盒节点」的区别

- 可以只换编码器 / 换 VAE / 换采样器 / 换调度器
- 能拿中间 latent 做别的事（图生图、二次采样）
- 能插 LoRA（走 manifest 的预合并组合）
- 每个节点能在界面上单独调试、看耗时

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `KA_HOME` | `/data/data/com.termux/files/home` | Termux 家目录 |
| `KA_WORK` | `$KA_HOME/npt` | 工作目录（临时文件、manifest 缓存） |
| `KA_TMP` | `$KA_WORK/gen_app` | 中间张量落盘目录（有 tmpfs 时建议指过去） |
| `KA_BLOB_ROOTS` | `/data/local/tmp:$KA_WORK` | 扫描 blob 的根目录（`:` 分隔） |
| `KA_HOST` | `$KA_WORK/host_new` | NPU 宿主可执行 |
| `KA_MERGE_LORA` | 空 | 编译期合并进权重用的 LoRA（离线） |
| `KA_LOG` | `$KA_WORK/katorch_nodes.log` | 日志 |

## 部署

```powershell
powershell -ExecutionPolicy Bypass -File F:\KAtorch\katorch_nodes\deploy.ps1
```
推送整个包到手机 `~/ComfyUI/custom_nodes/katorch_npu/` 并重启 ComfyUI。

两个必须知道的坑（都踩过）：

1. **备份目录不能留在 `custom_nodes/` 里**。ComfyUI 会把该目录下每个含 `__init__.py`
   的子目录都当插件加载，备份留在原地会重复注册节点，旧备份还会覆盖新节点。
   脚本现在把备份放在 `~/npt/node_backups/`。
2. **不能在 adb shell 里直接 `pkill -f "main.py --cpu"`**。那条命令行自己就含这个串，
   `pkill` 会把执行者自己杀掉，看起来像"重启了但日志没变"。脚本走设备上的
   `~/npt/ka_restart.sh`（cmdline 里不带该模式）。

## 实测（MT6897 / Dimensity 8300）

| 项目 | 数值 |
|---|---|
| DiT 单步（14 块 map/run/unmap） | **11.0 s** |
| 8 步采样 | **90 s** |
| VAE 解码（512×512，CPU） | **36 s** |
| 文本编码（Qwen3-0.6B + 6 层 adapter） | 8～12 s（首次含加载约 30 s） |
| ComfyUI 端到端（含加载） | **151 s** |
| NPU 宿主 bload（16 块 3.9GB 常驻） | 1 s（OS 缓存命中时） |

## 等价性验证（重要）

模块化拆开之后数值没走样，逐位验证过：

| 环节 | 对比对象 | 结果 |
|---|---|---|
| DiT 单步前向 | `npt/app/gen.py::Pipeline._stack` | `max|Δ| = 0`，逐位相同 |
| 8 步采样（euler+beta） | `Pipeline.generate` | latent `max|Δ| = 0`，逐位相同 |
| VAE 解码 | `Pipeline.decode` | 像素差 0，逐位相同 |

（er_sde 内部有 `torch.randn_like`，但随机序列与旧管线同序，同 seed 同样可复现。）

## 端到端验证记录（2026-11-15）

| 工作流 | 结果 | 耗时 |
|---|---|---|
| `modular_api.json` 模块化文生图（8 步 er_sde+beta） | ✅ 出图正常 | 151 s |
| `cfg_api.json` cfg=1.6 带负向提示词（6 步） | ✅ 出图正常 | 197 s |
| `blackbox_api.json` 旧黑盒节点 | ✅ 出图正常 | 156 s |
| `roundtrip_api.json` VAE 编码→解码往返 | ✅ PSNR **46.15 dB** | 65 s |
| `img2img_api.json` 图生图（denoise 0.55） | ✅ 保留构图、按提示词改 | 106 s |

### 图生图的坑：latent 归一化

ComfyUI 的 Wan VAE 走视频路径，`VAE.encode()` 返回的 latent：

1. **带一维 T**（`[1,16,1,64,64]`），要压掉；
2. **没做 mean/std 归一化**（归一化在 sampling 侧、由 latent_format 处理）。

我们 DiT 工作在归一化空间，所以 `KAtorchVAEEncode` 里自己做了
`(z - LAT_MEAN) / LAT_STD`。不做这一步，图会过曝+过饱和（实测 PSNR 只有 16.9 dB）。

## 图生图怎么搭

```
LoadImage ──IMAGE──► KAtorchVAEEncode ──LATENT──► KAtorchKSampler(denoise=0.4~0.7) ──► KAtorchVAEDecode ──► SaveImage
```

`denoise` 决定改动幅度：0.3 是微调，0.55 保留构图换细节，0.8 以上基本重画。
注意图生图时采样起点的噪声幅度会按 `sigmas[0]` 缩放（和纯文生图不同），这是对的。

## 换底模

NPU 是静态图，**权重被编进 blob 里**，所以换底模 = 重新生成图 + 重新编译，不是换个文件就行。

```sh
# 1. 模型推到 ~/ComfyUI/models/diffusion_models/
# 2. 生成 16 张图（实测 97 秒）
KA_ANIMA=<新模型路径> KA_MERGE_LORA=<turbo.safetensors> KA_LORA_STR=0.6 \
KA_DST=$HOME/npt/e2eper512miao KA_SCRATCH=/data/local/tmp/gscratchmiao \
python ~/npt/g512_param.py
# 3. 编译 blob（实测 49 秒）
KA_DST=$HOME/npt/e2eper512miao KA_BLOBDIR=/data/local/tmp/blobs512miao \
python ~/npt/e512_param.py
# 4. 在 ~/npt/assets/overrides.json 登记为 verified 资产，重启 ComfyUI
```

三个坑：

1. **不同 checkpoint 的 key 前缀不一样**（`net.blocks.*` vs `blocks.*`）。`anima_block.py` 和
   `st.py` 都做了兼容。
2. **微调底模连文本编码器都改了**。`KAtorchCLIPTextEncode` 连上 `model` 就自动用该 checkpoint
   的 `llm_adapter`；不连会用 `anima_v1` 的，条件空间会错位。
3. **LoRA 强度要跟官方工作流对齐**：miaomiaoHarem 官方 8 步是 `turbo @ 0.6`，不是 1.0。

当前并存的两个资产：

| id | 底模 | turbo LoRA |
|---|---|---|
| `blobs512m` | anima_v1 | 1.0 |
| `blobs512miao` | miaomiaoHarem_anima12 | 0.6 |

## 常见问题

**"没有 anima_v1 @ 768x768 的编译产物"**
NPU 是静态图，分辨率在编译时被固化进 blob。想出新分辨率：
1. 用 `编译器/` 那套流程（`lowering.py` + `host.c`）按新分辨率编出 16 块
2. 放到 `/data/local/tmp/blobs<边长>/` 之类的新目录
3. 在 `~/npt/assets/overrides.json` 里登记 `{"blobs768": {"width": 768, "height": 768, "calib": "...", "verified": true}}`
4. 重启 ComfyUI，新的分辨率就会出现在下拉框里

单块编译约 2-3 分钟 × 16 块 ≈ 40 分钟，可以后台跑。**注意内存**：800×1424 会加载
7.9GB 模型 + 大激活，实测把手机拖死过。

**下拉框里只有 `blobs512m`，其他几份去哪了**
默认只显示 `overrides.json` 里标了 `"verified": true` 的资产。手机上有好几份历史编译产物
（旧的 `blobs512`、量化的 `blobs512q`、残件 `blobs512mix`），选错画质直接崩，而界面上看不出区别。
想看全部：`export KA_SHOW_ALL_ASSETS=1` 再重启。

**画质偏平 / 塑料感**
检查 `scheduler` 是不是 `beta`（PC 上的原版工作流就是 beta）。旧的 `linear` 是早期默认值，
当初画风对不上就是因为这个。

**cfg > 1 很慢**
正常。cfg>1 要正负两个分支各跑一遍 DiT，耗时翻倍（8 步 ≈ 3 分钟）。Anima + turbo LoRA 在
cfg=1 下就能出图，除非确实需要负向提示词。

**LoRA 没效果**
NPU 是静态图，LoRA 必须在**编译前**合并进权重再编 blob。`KAtorchLoraLoader` 只能选
"已经并进去的那份组合"，选了没编进去的 LoRA 会提示但不会有变化。

## 自检

不想启动 ComfyUI 也能查环境（需要先停掉 ComfyUI，NPU 宿主会冲突）：

```sh
export LD_PRELOAD=/data/data/com.termux/files/usr/lib/libpython3.14.so
python -c "import sys; sys.path.insert(0,'/data/data/com.termux/files/home/ComfyUI/custom_nodes'); \
           from katorch_npu import selftest; selftest.main(full=True)"
```

## 工作流

`workflows/` 里三份：

| 文件 | 用途 |
|---|---|
| `modular_ui.json` | 界面上打开就是完整模块化连线（9 个节点） |
| `modular_api.json` | API 格式，脚本提交用（er_sde + beta，8 步） |
| `cfg_api.json` | 带负向提示词的 cfg=1.6 示例（耗时翻倍） |
| `blackbox_api.json` | 旧黑盒节点 `KAtorchNPUGenerate`，验证向后兼容 |

提交测试：
```powershell
python F:\KAtorch\tools\comfy_submit.py workflows\modular_api.json --url http://127.0.0.1:18188
```
（先 `adb forward tcp:18188 tcp:8188`）

