"""测 patchify / unpatchify 到底占多少时间，以及张量化版本能快多少。"""
import sys, time
sys.path.insert(0, "/data/data/com.termux/files/home/ComfyUI/custom_nodes")

import torch
import numpy as np


class Old(object):
    S_H = S_W = 32
    S = S_H * S_W
    LH = LW = 64

    def patchify(self, x):
        ll = x[0].permute(1, 2, 0)
        pp = torch.zeros(self.S, 68)
        for i in range(self.S_H):
            for j in range(self.S_W):
                pp[i * self.S_W + j, :64] = ll[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] \
                    .permute(2, 0, 1).reshape(-1)
        return pp

    def unpatchify(self, tok):
        v = torch.zeros(1, 16, self.LH, self.LW)
        for i in range(self.S_H):
            for j in range(self.S_W):
                v[0, :, i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = \
                    tok[i * self.S_W + j].reshape(2, 2, 16).permute(2, 0, 1)
        return v


class New(object):
    S_H = S_W = 32
    S = S_H * S_W
    LH = LW = 64

    def patchify(self, x):
        # [1,16,64,64] -> [1,16,32,2,32,2] -> [32,32,16,2,2] -> [1024,64]
        ll = x[0]                                    # [16,64,64]
        a = ll.reshape(16, 32, 2, 32, 2)             # [16,32,2,32,2]
        a = a.permute(1, 3, 0, 2, 4)                 # [32,32,16,2,2]
        return a.reshape(self.S, 64)

    def unpatchify(self, tok):
        # [1024,64] -> [32,32,16,2,2] -> [16,32,2,32,2] -> [1,16,64,64]
        a = tok.reshape(32, 32, 16, 2, 2)
        a = a.permute(2, 0, 3, 1, 4)                 # [16,32,2,32,2]
        return a.reshape(1, 16, self.LH, self.LW)


x = torch.randn(1, 16, 64, 64)
tok = torch.randn(1024, 64)

o, n = Old(), New()

# 正确性
po, pn = o.patchify(x), n.patchify(x)
print("patchify 一致:", torch.allclose(po[:, :64], pn, atol=1e-6))

uo, un = o.unpatchify(tok), n.unpatchify(tok)
print("unpatchify 一致:", torch.allclose(uo, un, atol=1e-6))
print()

REP = 20
t0 = time.time()
for _ in range(REP):
    o.patchify(x)
t_old_p = (time.time() - t0) / REP * 1000

t0 = time.time()
for _ in range(REP):
    o.unpatchify(tok)
t_old_u = (time.time() - t0) / REP * 1000

t0 = time.time()
for _ in range(REP):
    n.patchify(x)
t_new_p = (time.time() - t0) / REP * 1000

t0 = time.time()
for _ in range(REP):
    n.unpatchify(tok)
t_new_u = (time.time() - t0) / REP * 1000

print("旧版 patchify   : %7.2f ms" % t_old_p)
print("旧版 unpatchify : %7.2f ms" % t_old_u)
print("旧版合计/步     : %7.2f ms" % (t_old_p + t_old_u))
print()
print("新版 patchify   : %7.3f ms  (%.0fx)" % (t_new_p, t_old_p / max(t_new_p, 1e-6)))
print("新版 unpatchify : %7.3f ms  (%.0fx)" % (t_new_u, t_old_u / max(t_new_u, 1e-6)))
print("新版合计/步     : %7.3f ms" % (t_new_p + t_new_u))
print()
print("8 步能省: %.2f 秒" % ((t_old_p + t_old_u - t_new_p - t_new_u) * 8 / 1000))
