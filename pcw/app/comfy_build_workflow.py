"""生成 ComfyUI 的 UI 格式工作流（可拖到界面里直接打开）。"""
import json, os

W, H = 512, 512
STEPS, CFG = 8, 1.0
POS = "masterpiece, best quality, very aesthetic, 1girl, silver hair, blue eyes, white dress, standing in water, moonlight"
NEG = "lowres, bad anatomy, bad hands, worst quality, low quality, jpeg artifacts, watermark, signature"


def node(nid, ntype, pos, size, inputs, outputs, widgets, title=None):
    n = {"id": nid, "type": ntype, "pos": pos, "size": size,
         "flags": {}, "order": nid, "mode": 0,
         "inputs": [{"name": a, "type": b, "link": None} for a, b in inputs],
         "outputs": [{"name": a, "type": b, "links": [], "slot_index": i}
                     for i, (a, b) in enumerate(outputs)],
         "properties": {"Node name for S&R": ntype},
         "widgets_values": widgets}
    if title:
        n["title"] = title
    return n


nodes = [
    node(1, "UNETLoader", [60, 120], [320, 82], [], [("MODEL", "MODEL")],
         ["anima_v1.safetensors", "default"], "UNet加载器"),
    node(2, "CLIPLoader", [60, 260], [320, 106], [], [("CLIP", "CLIP")],
         ["qwen3.safetensors", "qwen_image"], "加载CLIP"),
    node(3, "VAELoader", [60, 420], [320, 58], [], [("VAE", "VAE")],
         ["flux2-vae.safetensors"], "VAE加载器"),
    node(4, "CLIPTextEncode", [430, 100], [400, 160], [("clip", "CLIP")],
         [("CONDITIONING", "CONDITIONING")], [POS], "正向提示词"),
    node(5, "CLIPTextEncode", [430, 300], [400, 120], [("clip", "CLIP")],
         [("CONDITIONING", "CONDITIONING")], [NEG], "负向提示词"),
    node(6, "EmptyLatentImage", [430, 470], [320, 106], [], [("LATENT", "LATENT")],
         [W, H, 1], "空Latent"),
    node(7, "KSampler", [880, 200], [320, 262],
         [("model", "MODEL"), ("positive", "CONDITIONING"),
          ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
         [("LATENT", "LATENT")],
         [42, "randomize", STEPS, CFG, "euler", "simple", 1.0], "采样器"),
    node(8, "VAEDecode", [1240, 200], [210, 46],
         [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [], "VAE解码"),
    node(9, "SaveImage", [1240, 300], [400, 450], [("images", "IMAGE")], [],
         ["katorch"], "保存图像"),
]

# links: [link_id, 源节点, 源slot, 目标节点, 目标slot, 类型]
defs = [
    (4, 0, 4, 0, "CLIP"),    # CLIPLoader -> 正向
    (4, 0, 5, 0, "CLIP"),    # CLIPLoader -> 负向
    (1, 0, 7, 0, "MODEL"),   # UNET -> KSampler
    (4, 0, 7, 1, "CONDITIONING"),
    (5, 0, 7, 2, "CONDITIONING"),
    (6, 0, 7, 3, "LATENT"),
    (7, 0, 8, 0, "LATENT"),
    (3, 0, 8, 1, "VAE"),
    (8, 0, 9, 0, "IMAGE"),
]
links = []
for i, (src, sslot, dst, dslot, typ) in enumerate(defs, start=1):
    links.append([i, src, sslot, dst, dslot, typ])
    for nd in nodes:
        if nd["id"] == src:
            nd["outputs"][sslot]["links"].append(i)
        if nd["id"] == dst:
            nd["inputs"][dslot]["link"] = i

wf = {
    "last_node_id": 9, "last_link_id": len(links),
    "nodes": nodes, "links": links,
    "groups": [], "config": {}, "extra": {"ds": {"scale": 0.85, "offset": [40, 30]}},
    "version": 0.4,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katorch_anima.json")
json.dump(wf, open(out, "w"), ensure_ascii=False, indent=1)
print("已生成 %s (%d 字节, %d 节点, %d 连线)"
      % (out, os.path.getsize(out), len(nodes), len(links)))
