"""最小 safetensors 读取器：只读单个张量，支持 BF16/F16/F32，可选 LoRA 合并。

不依赖 safetensors 库（手机上的版本行为未验证），直接解析头部。
"""
import json
import struct

import numpy as np
import torch

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}


class SafeTensorFile(object):
    """打开一个 safetensors 文件，按需读取名字对应的张量（float32）。"""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n).decode("utf-8"))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def keys(self):
        return list(self.header.keys())

    def has(self, name):
        return self._resolve(name) is not None

    def _resolve(self, name):
        """兼容不同 checkpoint 的 key 前缀。

        anima_v1 用 `net.blocks.0...`，miaomiaoHarem_anima12 是裸 `blocks.0...`，
        张量数量和形状完全一致，只是前缀不同。
        """
        e = self.header.get(name)
        if e is None:
            if name.startswith("net."):
                e = self.header.get(name[4:])
            elif ("net." + name) in self.header:
                e = self.header["net." + name]
        return e

    def get(self, name, dtype=torch.float32):
        e = self._resolve(name)
        if e is None:
            raise KeyError("%s 里没有张量 %s" % (self.path, name))
        o1, o2 = e["data_offsets"]
        dt = e.get("dtype", "F32")
        with open(self.path, "rb") as f:
            f.seek(self.base + o1)
            raw = f.read(o2 - o1)
        if dt == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif dt == "I64":
            arr = np.frombuffer(raw, dtype=np.int64)
        elif dt == "I32":
            arr = np.frombuffer(raw, dtype=np.int32)
        else:
            raise TypeError("不支持的 dtype %s（%s）" % (dt, name))
        t = torch.from_numpy(np.ascontiguousarray(arr.reshape(e["shape"])))
        return t.to(dtype) if dtype is not None and t.is_floating_point() else t


class LoraSet(object):
    """一个 LoRA 文件：把模型权重名映射到 (A, B)。"""

    def __init__(self, path):
        self.path = path
        self.st = SafeTensorFile(path)
        self._cache = {}

    @staticmethod
    def _stem(model_name):
        if not model_name.endswith(".weight"):
            return None
        s = model_name[:-len(".weight")]
        if s.startswith("net."):
            s = "diffusion_model." + s[4:]
        elif not s.startswith("diffusion_model."):
            s = "diffusion_model." + s
        return s

    def delta(self, model_name, strength=1.0):
        stem = self._stem(model_name)
        if stem is None:
            return None
        key = (stem, float(strength))
        if key in self._cache:
            return self._cache[key]
        a = stem + ".lora_A.weight"
        b = stem + ".lora_B.weight"
        d = None
        if self.st.has(a) and self.st.has(b):
            d = self.st.get(b).float() @ self.st.get(a).float()
            if strength != 1.0:
                d = d * float(strength)
        self._cache[key] = d
        return d
