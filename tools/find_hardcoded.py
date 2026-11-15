"""扫描开源仓库里所有硬编码的本地路径和机器特定值。"""
import os
import re

DST = r"F:\comfyui-android"

# 要查的模式
PATTERNS = [
    ("Windows盘符路径", r"[A-Za-z]:\\[^\s'\"`<>|]+"),
    ("Linux绝对路径",   r"/(?:data/data/com\.termux|data/local/tmp|vendor|system)[^\s'\"`<>|]*"),
    ("AndroidSDK路径",  r"(?:D:\\AndroidSdk|C:\\Program Files\\Java)[^\s'\"`]*"),
    ("手机序列号",      r"\b(?:TNK4PCN|PA7M)[A-Z0-9]+\b"),
    ("具体版本号硬编码", r"build-tools\\\\?\d+\.\d+\.\d+|android-\d{2}"),
]

TEXT_EXT = {".md", ".py", ".sh", ".ps1", ".java", ".xml", ".json", ".txt", ".yml", ".yaml"}

# 这些是「说明用途」的，可以保留（文档里教别人路径长什么样）
ALLOW_CONTEXT = [
    "例如", "比如", "如：", "示例", "路径形如", "通常位于", "一般在",
    "for example", "e.g.", "such as",
]

hits = []
for root, dirs, files in os.walk(DST):
    dirs[:] = [d for d in dirs if d not in {"build", "out", "__pycache__", ".git"}]
    for f in files:
        if os.path.splitext(f)[1].lower() not in TEXT_EXT:
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, DST)
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if any(a in line for a in ALLOW_CONTEXT):
                continue
            for name, pat in PATTERNS:
                for m in re.finditer(pat, line):
                    hits.append((rel, i, name, m.group(0)[:80], line.strip()[:110]))

print("找到 %d 处硬编码：" % len(hits))
print()
cur = None
for rel, ln, name, match, line in hits:
    if rel != cur:
        print("── %s" % rel)
        cur = rel
    print("   L%-4d [%s] %s" % (ln, name, match))
print()
