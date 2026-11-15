"""LoRA 合并模块：给 anima_block.load() 用。

设计:
  · 默认不合并（KA_MERGE_LORA 未设置时行为与原来完全一致）
  · 设置 KA_MERGE_LORA=<safetensors 路径> 后，load(name) 会返回 W + (B @ A)
  · 权重命名映射: 模型 `net.blocks.0.xxx.weight`
                  LoRA `diffusion_model.blocks.0.xxx.lora_A.weight` / `.lora_B.weight`
"""
import json
import struct

import numpy as np
import torch

_LORA = None          # 惰性加载的 LoRA 字典
_LORA_PATH = None


def _read_st(path):
    """读 safetensors -> {name: torch.float32 tensor}"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, e in hdr.items():
            if k == "__metadata__":
                continue
            o1, o2 = e["data_offsets"]
            dt = e.get("dtype", "F32")
            f.seek(base + o1)
            raw = f.read(o2 - o1)
            if dt == "BF16":
                arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                continue
            out[k] = torch.from_numpy(np.ascontiguousarray(arr.reshape(e["shape"])))
    return out


def lora_for(model_name):
    """模型权重名 -> (A, B) 或 None。"""
    global _LORA, _LORA_PATH
    import os
    path = os.environ.get("KA_MERGE_LORA", "")
    if not path:
        return None
    if _LORA is None or _LORA_PATH != path:
        _LORA = _read_st(path)
        _LORA_PATH = path
    if not model_name.endswith(".weight"):
        return None
    stem = model_name[:-len(".weight")]
    if stem.startswith("net."):
        stem = "diffusion_model." + stem[4:]
    else:
        stem = "diffusion_model." + stem
    a = _LORA.get(stem + ".lora_A.weight")
    b = _LORA.get(stem + ".lora_B.weight")
    if a is None or b is None:
        return None
    return a, b


def apply_lora(w, model_name):
    """w: [out,in] float32；返回合并后的权重（若存在对应 LoRA）。"""
    ab = lora_for(model_name)
    if ab is None:
        return w
    import os
    a, b = ab
    # a: [r,in]  b: [out,r]  -> delta = b @ a
    str_ = float(os.environ.get("KA_LORA_STR", "1.0"))
    delta = (b.float() @ a.float()) * str_
    if delta.shape != w.shape:
        return w
    return w + delta
