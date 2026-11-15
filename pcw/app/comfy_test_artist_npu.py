import json, urllib.request

API = "http://127.0.0.1:8188"

# 先确认新节点带着 prompt_in
oi = json.load(urllib.request.urlopen(API + "/object_info", timeout=60))
print("总节点数:", len(oi))
k = oi.get("KAtorchNPUGenerate")
if k:
    req = list(k.get("input", {}).get("required", {}).keys())
    opt = list(k.get("input", {}).get("optional", {}).keys())
    print("KAtorchNPUGenerate  required:", req)
    print("                     optional:", opt)
else:
    print("节点缺失")
    raise SystemExit

ARTISTS = ("@chigusa minori, @yoneyama mai, @mochizuki kei, @hiten, "
           "@wlop, @takeuchi takashi, @sos adult, @kobinbin")
QUALITY = ("masterpiece, best quality, score_9, score_8, highres, absurdres, "
           "anime screenshot, official art")

prompt = {
    "1": {"class_type": "AnimaV4RandomPrompt",
          "inputs": {"nsfw_level": "sfw", "seed": -1, "tag_cap": "自动",
                     "base_prob": 0.65, "disabled_sections": "", "lock_fields": "",
                     "force_role": "", "weights_json": "", "extra_body": "",
                     "appearance_follow": True, "selected_roles": "",
                     "big_breasts": False, "role_works": "", "breast_size": "不指定",
                     "outfit_style": "随机", "role_fav_only": False,
                     "hair_follow": True, "hair_len_pick": "角色标签发型",
                     "hair_tie_pick": "角色标签发型", "hair_curl_pick": "角色标签发型",
                     "hair_acc_pick": "角色标签发型"}},
    "2": {"class_type": "StringConcatenate",
          "inputs": {"string_a": ARTISTS, "string_b": ["1", 0], "delimiter": ", "}},
    "3": {"class_type": "StringConcatenate",
          "inputs": {"string_a": QUALITY, "string_b": ["2", 0], "delimiter": ", "}},
    "4": {"class_type": "KAtorchNPUGenerate",
          "inputs": {"prompt": "", "negative": "", "seed": 777,
                     "width": 512, "height": 512, "prompt_in": ["3", 0]}},
    "5": {"class_type": "SaveImage",
          "inputs": {"images": ["4", 0], "filename_prefix": "artist_npu"}},
}

req = urllib.request.Request(API + "/prompt", data=json.dumps({"prompt": prompt}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=40)
    j = json.loads(r.read().decode())
    print("提交成功:", j.get("prompt_id"))
    if j.get("node_errors"):
        print("节点错误:", json.dumps(j["node_errors"], ensure_ascii=False)[:600])
except urllib.error.HTTPError as e:
    print("提交失败:", e.code)
    print(e.read().decode()[:1200])
