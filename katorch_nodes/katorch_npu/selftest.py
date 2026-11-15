"""KAtorch 节点包自检：不启动 ComfyUI，直接在 Termux 里跑。

用法（手机 Termux，需要 root + 先停掉 ComfyUI，否则 NPU 宿主会冲突）：
    export LD_PRELOAD=/data/data/com.termux/files/usr/lib/libpython3.14.so
    python -c "import sys; sys.path.insert(0, '/data/data/com.termux/files/home/ComfyUI/custom_nodes'); \
               from katorch_npu import selftest; selftest.main()"

    # 加上 DiT 单步（约 15 秒，需要 4GB 空闲内存）
    ... selftest.main(full=True)
"""
import os
import sys
import time


def _ok(msg):
    print("  [OK]   " + msg, flush=True)


def _warn(msg):
    print("  [WARN] " + msg, flush=True)


def _fail(msg):
    print("  [FAIL] " + msg, flush=True)


def check_assets():
    from . import assets
    print("\n== 资产层", flush=True)
    mf = assets.get_manifest(force=True)
    ok = True
    if not mf["assets"]:
        _fail("没有找到任何 blob 集（扫描根：%s）" % " / ".join(assets.BLOB_ROOTS))
        return mf, False
    for a in mf["assets"]:
        tag = "[已验证]" if a.get("verified") else "[未验证]"
        line = "%s %-12s %sx%s %d块 %.2fGB → %s" % (
            tag, a["id"], a.get("width"), a.get("height"), a.get("blocks", 0),
            (a.get("bytes") or 0) / 1e9, a.get("calib"))
        if not a.get("complete", True):
            _warn(line + "（不完整，已忽略）")
        elif a.get("verified"):
            _ok(line)
        else:
            print("         " + line, flush=True)
    if not any(a.get("verified") for a in mf["assets"]):
        _warn("没有任何资产被标记为已验证：编辑 %s 把确认可用的一份设成 \"verified\": true" % assets.OVERRIDES)
    print("  清单缓存：%s" % assets.MANIFEST, flush=True)
    return mf, ok


def check_weights():
    from . import runtime
    print("\n== 权重", flush=True)
    ok = True
    try:
        clip = runtime.load_clip()
        _ok("文本编码器：%s" % clip.qwen_path)
        _ok("LLMAdapter 权重：%s" % clip.anima_path)
        _ok("tokenizer 目录：%s" % clip.tok_dir)
    except Exception as e:
        _fail("文本编码器：%s" % e)
        ok = False
    try:
        vae = runtime.load_vae()
        _ok("VAE：%s" % vae.path)
    except Exception as e:
        _fail("VAE：%s" % e)
        ok = False
    from .host import DEFAULT_HOST
    if os.path.isfile(DEFAULT_HOST):
        _ok("NPU 宿主：%s" % DEFAULT_HOST)
    else:
        _fail("NPU 宿主不存在：%s" % DEFAULT_HOST)
        ok = False
    return ok


def check_text_encode():
    from . import runtime
    print("\n== 文本编码", flush=True)
    try:
        clip = runtime.load_clip()
        t0 = time.time()
        cond = runtime.encode_text(clip, "1girl, solo, blue eyes")
        arr = cond.ctx
        _ok("%.1fs  ctx=%s dtype=%s mean=%.5f" % (time.time() - t0, arr.shape, arr.dtype, float(arr.mean())))
        return True
    except Exception as e:
        _fail("编码失败：%s" % e)
        return False


def check_dit(seed=1234):
    from . import assets, runtime
    import torch
    print("\n== DiT 单步（会占用 NPU 和约 4GB 内存）", flush=True)
    mf = assets.get_manifest()
    target = None
    for a in mf["assets"]:
        if a.get("verified"):
            target = a
            break
    if target is None:
        target = next((a for a in mf["assets"] if a.get("complete", True)), None)
    if target is None:
        _fail("没有可用资产")
        return False
    try:
        model = runtime.KAtorchModel(target)
        rt = runtime.get_dit(model)
        torch.manual_seed(seed)
        x = torch.randn(1, 16, rt.LH, rt.LW)
        ctx = os.path.join(rt.tmp, "i_ctx_selftest.bin")
        import numpy as np
        np.zeros((512, 1024), np.float16).tofile(ctx)
        t0 = time.time()
        v = rt.forward(x, ctx, 0.9428)
        dt = time.time() - t0
        _ok("单步 %.1fs  v.mean=%.4f v.std=%.4f（NaN=%s）"
            % (dt, float(v.mean()), float(v.std()), bool(torch.isnan(v).any())))
        _ok("预计 8 步 %d 秒" % int(dt * 8))
        return not bool(torch.isnan(v).any())
    except Exception as e:
        _fail("DiT 前向失败：%s" % e)
        return False


def main(full=False):
    print("KAtorch 自检  %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("  KA_HOME=%s" % os.environ.get("KA_HOME", "/data/data/com.termux/files/home"), flush=True)
    results = []
    mf, ok1 = check_assets()
    ok2 = check_weights()
    ok3 = check_text_encode()
    results = [("资产", ok1), ("权重", ok2), ("文本编码", ok3)]
    if full:
        results.append(("DiT 单步", check_dit()))
    print("\n== 结论", flush=True)
    for name, ok in results:
        print("   %s: %s" % (name, "通过" if ok else "失败"), flush=True)
    return all(ok for _, ok in results)


if __name__ == "__main__":
    sys.exit(0 if main("--full" in sys.argv) else 1)
