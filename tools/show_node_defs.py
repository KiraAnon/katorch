import json, urllib.request

oi = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=60))
want = ["KAtorchUNETLoader", "KAtorchCLIPLoader", "KAtorchVAELoader",
        "KAtorchCLIPTextEncode", "KAtorchEmptyLatentImage", "KAtorchKSampler",
        "KAtorchVAEDecode", "KAtorchClearCondCache"]
for n in want:
    d = oi.get(n)
    if not d:
        print("%-24s 缺失" % n)
        continue
    req = d.get("input", {}).get("required", {})
    opt = d.get("input", {}).get("optional", {})
    print("%-24s 必填=%s 可选=%s" % (n, list(req.keys()), list(opt.keys())))
    for k, v in req.items():
        t = v[0] if isinstance(v, list) and v else v
        if isinstance(t, list):
            print("      %-16s 选项: %s" % (k, t[:6]))
        elif isinstance(t, str):
            extra = v[1] if isinstance(v, list) and len(v) > 1 else {}
            dflt = extra.get("default") if isinstance(extra, dict) else None
            print("      %-16s %-12s default=%s" % (k, t, dflt))
