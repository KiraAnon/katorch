import json, urllib.request

oi = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=60))
for name in ("AnimaV4RandomPrompt", "StringConcatenate"):
    d = oi.get(name)
    if not d:
        print(name, "缺失")
        continue
    print("=" * 60)
    print(name, "| 分类:", d.get("category"))
    req = d.get("input", {}).get("required", {})
    opt = d.get("input", {}).get("optional", {})
    print("  必填:")
    for k, v in req.items():
        t = v[0] if isinstance(v, list) and v else v
        if isinstance(t, list):
            print("    %-22s 选项(%d): %s" % (k, len(t), t[:6]))
        elif isinstance(t, str):
            extra = v[1] if isinstance(v, list) and len(v) > 1 else {}
            dflt = extra.get("default") if isinstance(extra, dict) else None
            print("    %-22s %-10s default=%s" % (k, t, dflt))
        else:
            print("    %-22s %s" % (k, t))
    if opt:
        print("  可选:", list(opt.keys()))
    print("  输出:", d.get("output_name"), d.get("output"))
