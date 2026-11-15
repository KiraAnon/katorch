p = "/data/data/com.termux/files/home/npt/bal_norm.py"
src = open(p).read()
src = src.replace("        return torch.maximum(v, torch.tensor(1e-5))",
                  "        return v")
open(p, "w").write(src)
print("_clamp_v 已改成直通（等于没加保护），恢复 1.577% 那一版的行为")
import re
for ln in src.splitlines():
    if "_clamp_v" in ln or "return v" == ln.strip():
        print("  ", ln)
