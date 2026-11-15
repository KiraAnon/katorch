"""只跑 VAE 解码，测试 latent 缩放对对比度的影响。不重新采样。"""
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")

import sys, os
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
from vae_decoder import WanVaeDecoder
from safetensors.torch import load_file

H = _KA_HOME
D = _KA_WORK
OUT = D + "/gen5"
LH = LW = 32

LAT_MEAN = torch.tensor([-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,
                         0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]).view(16,1,1)
LAT_STD = torch.tensor([2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,
                        3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]).view(16,1,1)

x = torch.from_numpy(np.fromfile(OUT + "/final_latent.bin", dtype=np.float16).astype(np.float32)).reshape(16, LH, LW)
print("latent std=%.4f  min=%.3f max=%.3f" % (x.std(), x.min(), x.max()), flush=True)

print("加载 VAE ...", flush=True)
vae_sd = load_file(H + "/ComfyUI/models/vae/qwen_image_vae.safetensors", device="cpu")
dec = WanVaeDecoder(vae_sd, H0=LH, W0=LW).eval()

for k in (1.0, 1.25, 1.5):
    z = (x * k) * LAT_STD + LAT_MEAN
    with torch.no_grad():
        img = dec(z.permute(1, 2, 0).unsqueeze(0)).squeeze(0)
    img8 = torch.clamp((img + 1) * 127.5, 0, 255).to(torch.uint8).numpy()
    p = OUT + "/vae_k%.2f.ppm" % k
    f = open(p, "wb")
    f.write(("P6\n%d %d\n255\n" % (img8.shape[1], img8.shape[0])).encode())
    f.write(img8.tobytes()); f.close()
    print("  k=%.2f  z.std=%.3f  img std=%.3f  range=[%.2f,%.2f]  -> %s"
          % (k, z.std(), img.std(), img.min(), img.max(), p), flush=True)
