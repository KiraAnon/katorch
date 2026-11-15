"""画师串 + NPU 加速的工作流（保留用户的拼接结构，采样走 NPU）。"""
import json, os

ARTISTS = ("@chigusa minori, @yoneyama mai, @mochizuki kei, @hiten, "
           "@wlop, @takeuchi takashi, @sos adult, @kobinbin")
QUALITY = ("masterpiece, best quality, score_9, score_8, highres, absurdres, "
           "anime screenshot, official art")

NOTE = """画师串用法：
在「① 画师串 + 随机正文」节点的 string_a 里写 @画师名，逗号分隔。常用：
@chigusa minori  @yoneyama mai  @mochizuki kei  @hiten  @wlop  @takeuchi takashi
@sos adult  @kobinbin  @choujiroo  @sincos  @ratatatat74  @satou kuuki  @endou okito
@dishwasher1910  @akkgsyk  @vanripper  @ke-ta  @as109  @karory  @ebifurya  @yueko
@ciloranko  @wanke  @horuhara  @keihh  @mikozin  @frengartist  @chen bin
@tamanoi peromekuri  @xilmo  @remsrar  @okazu  @ningen mame

流程：画师串 + AnimaV4RandomPrompt 正文 → 官方质量词 → NPU 节点生成（512x512，约 2 分钟）
如果想完全自由写提示词，把 NPU 节点的 prompt 文本框填上，并断开 prompt_in 的连线。
"""


def N(nid, ntype, pos, size, inputs, outputs, widgets, title=None):
    n = {"id": nid, "type": ntype, "pos": pos, "size": size, "flags": {},
         "order": nid, "mode": 0,
         "inputs": [{"name": a, "type": b, "link": None, "shape": 7} for a, b in inputs],
         "outputs": [{"name": a, "type": b, "links": [], "slot_index": i}
                     for i, (a, b) in enumerate(outputs)],
         "properties": {"Node name for S&R": ntype},
         "widgets_values": widgets}
    if title:
        n["title"] = title
    return n


nodes = [
    N(1, "AnimaV4RandomPrompt", [60, 80], [420, 560], [],
      [("text", "STRING"), ("report", "STRING"), ("level", "STRING"), ("cn_summary", "STRING")],
      ["sfw", -1, "randomize", "自动", 0.65, "", "", "", "", "", True, "", False,
       "", "不指定", "随机", False, True, "角色标签发型", "角色标签发型",
       "角色标签发型", "角色标签发型"], "随机提示词（正文）"),

    N(2, "StringConcatenate", [520, 80], [420, 130],
      [("string_a", "STRING"), ("string_b", "STRING"), ("delimiter", "STRING")],
      [("STRING", "STRING")], [ARTISTS, "", ", "], "① 画师串 + 随机正文"),
    N(3, "StringConcatenate", [520, 250], [420, 130],
      [("string_a", "STRING"), ("string_b", "STRING"), ("delimiter", "STRING")],
      [("STRING", "STRING")], [QUALITY, "", ", "], "② 官方质量词 + 上式"),

    N(4, "KAtorchNPUGenerate", [520, 420], [420, 250],
      [("prompt_in", "STRING")],
      [("image", "IMAGE")],
      ["", "", 42, 512, 512], "③ KAtorch NPU 生图（约2分钟）"),

    N(5, "SaveImage", [980, 80], [420, 460], [("images", "IMAGE")], [],
      ["katorch_artist"], "保存图像"),
    N(6, "Note", [60, 680], [420, 420], [], [], [NOTE], "画师串参考"),
]

defs = [
    (1, 0, 2, 1, "STRING"),
    (2, 0, 3, 1, "STRING"),
    (3, 0, 4, 0, "STRING"),
    (4, 0, 5, 0, "IMAGE"),
]
links = []
for i, (src, sslot, dst, dslot, typ) in enumerate(defs, start=1):
    links.append([i, src, sslot, dst, dslot, typ])
    for nd in nodes:
        if nd["id"] == src:
            nd["outputs"][sslot]["links"].append(i)
        if nd["id"] == dst:
            nd["inputs"][dslot]["link"] = i

wf = {"last_node_id": 6, "last_link_id": len(links), "nodes": nodes, "links": links,
      "groups": [], "config": {},
      "extra": {"ds": {"scale": 0.75, "offset": [30, 20]}}, "version": 0.4}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katorch_artist_npu.json")
json.dump(wf, open(out, "w"), ensure_ascii=False, indent=1)
print("已生成 %s（%d 节点, %d 连线）" % (out, len(nodes), len(links)))
