import json, os
src = "/data/local/tmp/q16i/meta.json"
dst = "/data/local/tmp/blobs512mix/meta.json"
m = json.load(open(src))
json.dump({"b00": m}, open(dst, "w"))
print("meta 已加 blob 名: %s" % list(m))
print("  输入 %d 个, 输出 %d 个" % (len(m["inputs"]), len(m["outputs"])))
