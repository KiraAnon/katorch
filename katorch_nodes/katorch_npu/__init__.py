"""KAtorch NPU 节点包（模块化版）。

和旧版（单文件 comfy_npu_node.py，整条管线一个黑盒节点）的区别：
拆成 UNETLoader / CLIPLoader / CLIPTextEncode / EmptyLatent / KSampler / VAEDecode，
每个都能单独换、单独调、单独复用中间产物。
"""
import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

VERSION = "2.0.0"

try:
    from . import runtime as _rt
    _rt.log("KAtorch 节点 v%s 已加载（%d 个节点）" % (VERSION, len(NODE_CLASS_MAPPINGS)))
    try:
        _mf = _rt.manifest()
        _names = [a.get("id") for a in _mf.get("assets", [])]
        _rt.log("可用资产：%s" % (", ".join(_names) if _names else "（无）"))
    except Exception as _e:
        _rt.log("资产扫描失败：%s" % _e)
except Exception as _e:  # noqa: BLE001
    print("[katorch] 初始化警告：%s" % _e, flush=True)
