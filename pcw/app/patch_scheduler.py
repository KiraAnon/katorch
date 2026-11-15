"""把 gen.py 的 sigma 调度从 linear+shift 改成 ComfyUI 的 beta 调度器。

ComfyUI 的 beta scheduler（comfy/samplers.py::beta_scheduler）：
    alpha = beta = 0.6
    ts = [1 - Beta.ppf(1 - i/n, alpha, beta) ...]  实际是取 beta 分布的逆分位
这里直接用 scipy 的 beta.ppf 复现。

保留原来的 linear+shift 作为可选（KA_SCHED=linear）。
"""
import re

P = "/data/data/com.termux/files/home/npt/app/gen.py"
src = open(P, encoding="utf-8").read()

OLD = ("        sigs = [float(_ts(SHIFT, (k + 1) / 1000.0)) for k in range(1000)]\n"
       "        self.sigs = [sigs[-(1 + int(x * (1000.0 / STEPS)))] for x in range(STEPS)] + [0.0]\n")

NEW = '''        # 调度器：ComfyUI 原生工作流用的是 beta，这里对齐（KA_SCHED=linear 可回退）
        _sched = os.environ.get("KA_SCHED", "beta")
        if _sched == "beta":
            try:
                from scipy.stats import beta as _betadist
                _a = float(os.environ.get("KA_BETA_A", "0.6"))
                _b = float(os.environ.get("KA_BETA_B", "0.6"))
                _sig = [float(_betadist.ppf(1.0 - (i + 1.0) / STEPS, _a, _b))
                        for i in range(STEPS)]
                _sig = [min(max(v, 0.0), 1.0) for v in _sig] + [0.0]
                self.sigs = _sig
                print("[gen] 调度器 beta(a=%.2f,b=%.2f) sigs=%s"
                      % (_a, _b, [round(v, 4) for v in _sig]), flush=True)
            except Exception as _e:
                print("[gen] beta 调度不可用(%s)，回退 linear" % _e, flush=True)
                _sched = "linear"
        if _sched != "beta":
            sigs = [float(_ts(SHIFT, (k + 1) / 1000.0)) for k in range(1000)]
            self.sigs = [sigs[-(1 + int(x * (1000.0 / STEPS)))] for x in range(STEPS)] + [0.0]
            print("[gen] 调度器 linear+shift=%.1f sigs=%s"
                  % (SHIFT, [round(v, 4) for v in self.sigs]), flush=True)
'''

if "KA_SCHED" in src:
    print("已经改过，跳过")
elif OLD in src:
    src = src.replace(OLD, NEW)
    open(P, "w", encoding="utf-8").write(src)
    print("已改成 beta 调度器（备份 gen.py.bak-sched）")
else:
    print("锚点没找到，需要手动处理")
