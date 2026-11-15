"""按用户原工作流（工作流-Anima-带画师串-8步.json）重建手机版。

原流用到的插件手机上缺：XB_ToolBox（KSampler/VAEDecode/SageAttention/ImageParamsMaster）、
rgthree（Seed/Context）、Custom-Scripts（CR Text）。这里换成原生节点，
保留下来的核心是：AnimaV4RandomPrompt（画师串生成）+ 两级 StringConcatenate 拼接 + 8步 er_sde/beta。
"""
import json, os

# ---- 画师串（从原 Note 节点里取的常用那批，可自由增删） ----
ARTISTS = ("@chigusa minori, @yoneyama mai, @mochizuki kei, @hiten, "
           "@wlop, @takeuchi takashi, @sos adult, @kobinbin")

QUALITY = ("masterpiece, best quality, score_9, score_8, highres, absurdres, "
           "anime screenshot, official art")

NEG = ("worst quality, low quality, score_1, score_2, score_3, artist name, "
       "simple background, lowres, (bad), text, error, fewer, extra, missing, "
       "jpeg artifacts, watermark, unfinished, displeasing, oldest, early, "
       "signature, artistic error, username, scan, [abstract], english text, "
       "shiny hair, extra toes")

NOTE = """画师串用法：
在「画师串」节点的 string_a 里写 @画师名，用逗号分隔。常用：
@chigusa minori  @yoneyama mai  @mochizuki kei  @hiten  @wlop  @takeuchi takashi
@sos adult  @kobinbin  @choujiroo  @sincos  @ratatatat74  @satou kuuki  @endou okito
@dishwasher1910  @akkgsyk  @vanripper  @ke-ta  @as109  @karory  @ebifurya  @yueko
@ciloranko  @wanke  @horuhara  @keihh  @mikozin  @frengartist  @chen bin
@tamanoi peromekuri  @xilmo  @remsrar  @okazu  @ningen mame

拼接顺序：画师串 + AnimaV4RandomPrompt 生成的正文 → 官方质量词 → 正向条件
"""

W, H = 800, 1424


def N(nid, ntype, pos, size, inputs, outputs, widgets, title=None):
    n = {"id": nid, "type": ntype, "pos": pos, "size": size, "flags": {},
         "order": nid, "mode": 0,
         "inputs": [{"name": a, "type": b, "link": None} for a, b in inputs],
         "outputs": [{"name": a, "type": b, "links": [], "slot_index": i}
                     for i, (a, b) in enumerate(outputs)],
         "properties": {"Node name for S&R": ntype},
         "widgets_values": widgets}
    if title:
        n["title"] = title
    return n


nodes = [
    N(1, "UNETLoader", [60, 60], [330, 82], [], [("MODEL", "MODEL")],
      ["anima_v1.safetensors", "default"], "UNet加载器"),
    N(2, "LoraLoader", [60, 190], [330, 130],
      [("model", "MODEL"), ("clip", "CLIP")], [("MODEL", "MODEL"), ("CLIP", "CLIP")],
      ["turbo.safetensors", 0.6, 0.6], "LoRA 加载 · turbo"),
    N(3, "CLIPLoader", [60, 370], [330, 106], [], [("CLIP", "CLIP")],
      ["qwen3.safetensors", "qwen_image"], "全局共享 CLIP"),
    N(4, "VAELoader", [60, 520], [330, 58], [], [("VAE", "VAE")],
      ["qwen_image_vae.safetensors"], "全局共享 VAE"),

    N(5, "AnimaV4RandomPrompt", [450, 60], [420, 560], [],
      [("text", "STRING"), ("report", "STRING"), ("level", "STRING"), ("cn_summary", "STRING")],
      # 注意：seed 后面要跟一个 control_after_generate 占位，一共 22 个值
      ["sfw", -1, "randomize", "自动", 0.65, "", "", "", "", "", True, "", False,
       "", "不指定", "随机", False, True, "角色标签发型", "角色标签发型",
       "角色标签发型", "角色标签发型"], "随机提示词（画师串正文）"),

    N(6, "StringConcatenate", [900, 60], [420, 130],
      [("string_a", "STRING"), ("string_b", "STRING"), ("delimiter", "STRING")],
      [("STRING", "STRING")], [ARTISTS, "", ", "], "① 画师串 + 随机正文"),
    N(7, "StringConcatenate", [900, 230], [420, 130],
      [("string_a", "STRING"), ("string_b", "STRING"), ("delimiter", "STRING")],
      [("STRING", "STRING")], [QUALITY, "", ", "], "② 官方质量词 + 上式"),

    N(8, "CLIPTextEncode", [900, 400], [420, 160], [("clip", "CLIP"), ("text", "STRING")],
      [("CONDITIONING", "CONDITIONING")], [""], "正向条件（自动拼接）"),
    N(9, "CLIPTextEncode", [900, 600], [420, 200], [("clip", "CLIP"), ("text", "STRING")],
      [("CONDITIONING", "CONDITIONING")], [NEG], "负向词（官方）"),

    N(10, "EmptyLatentImage", [60, 620], [330, 106], [], [("LATENT", "LATENT")],
      [W, H, 1], "空Latent 800x1424"),

    N(11, "KSampler", [1380, 60], [330, 262],
      [("model", "MODEL"), ("positive", "CONDITIONING"),
       ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
      [("LATENT", "LATENT")],
      [42, "randomize", 8, 1, "er_sde", "beta", 1], "采样器 8步 er_sde/beta"),
    N(12, "VAEDecode", [1380, 360], [330, 46],
      [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [], "VAE解码"),
    N(13, "SaveImage", [1380, 440], [420, 460], [("images", "IMAGE")], [],
      ["katorch_anima"], "保存图像"),

    N(14, "Note", [60, 760], [420, 420], [], [], [NOTE], "画师串参考"),
]

# (源节点, 源slot, 目标节点, 目标slot, 类型)
defs = [
    (3, 0, 2, 1, "CLIP"),       # CLIPLoader -> LoraLoader.clip
    (2, 1, 8, 0, "CLIP"),       # LoraLoader.CLIP -> 正向
    (2, 1, 9, 0, "CLIP"),       # LoraLoader.CLIP -> 负向
    (2, 0, 11, 0, "MODEL"),     # LoraLoader.MODEL -> KSampler
    (5, 0, 6, 1, "STRING"),     # 随机正文 -> 拼接①的 string_b
    (6, 0, 7, 1, "STRING"),     # 拼接① -> 拼接②的 string_b
    (7, 0, 8, 1, "STRING"),     # 拼接② -> 正向 text
    (8, 0, 11, 1, "CONDITIONING"),
    (9, 0, 11, 2, "CONDITIONING"),
    (10, 0, 11, 3, "LATENT"),
    (11, 0, 12, 0, "LATENT"),
    (4, 0, 12, 1, "VAE"),
    (12, 0, 13, 0, "IMAGE"),
]
links = []
for i, (src, sslot, dst, dslot, typ) in enumerate(defs, start=1):
    links.append([i, src, sslot, dst, dslot, typ])
    for nd in nodes:
        if nd["id"] == src:
            nd["outputs"][sslot]["links"].append(i)
        if nd["id"] == dst:
            nd["inputs"][dslot]["link"] = i

wf = {"last_node_id": 14, "last_link_id": len(links), "nodes": nodes, "links": links,
      "groups": [], "config": {},
      "extra": {"ds": {"scale": 0.62, "offset": [30, 20]}}, "version": 0.4}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katorch_anima_artist.json")
json.dump(wf, open(out, "w"), ensure_ascii=False, indent=1)
print("已生成 %s" % out)
print("  %d 节点, %d 连线, %.0f 字节" % (len(nodes), len(links), os.path.getsize(out)))
