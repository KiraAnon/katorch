import json, urllib.request

r = urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=40)
oi = json.load(r)

want = ["UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
        "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"]
for name in want:
    d = oi.get(name)
    if not d:
        print("%s: 不存在" % name)
        continue
    inp = d.get("input", {})
    req = inp.get("required", {})
    print("=== %s ===" % name)
    for k, v in req.items():
        t = v[0] if isinstance(v, list) and v else v
        if isinstance(t, list):
            print("   %-14s 可选值: %s" % (k, t[:8]))
        else:
            print("   %-14s %s" % (k, t))
    print("   输出:", d.get("output_name") or d.get("output"))
    print()
