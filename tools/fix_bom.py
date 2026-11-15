"""给仓库里所有 .ps1 补 UTF-8 BOM（PS 5.1 不加 BOM 会按 GBK 读坏中文）。

用法： python fix_bom.py [目录]
"""
import os
import sys

BOM = b"\xef\xbb\xbf"


def fix(path):
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(BOM):
        return "已有 BOM"
    with open(path, "wb") as f:
        f.write(BOM + data)
    return "已加 BOM"


def verify(path):
    """用 PowerShell 的 parser 做语法检查。"""
    import subprocess
    ps = (
        "$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile('%s',[ref]$null,[ref]$e)|Out-Null;"
        "if($e -and $e.Count -gt 0){$e|ForEach-Object{$_.Message}}else{'OK'}"
    ) % path.replace("'", "''")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    return "OK" if out == "OK" else out[:200]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"F:\comfyui-android"
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"build", "out", "__pycache__", ".git"}]
        for f in filenames:
            if f.lower().endswith(".ps1"):
                found.append(os.path.join(dirpath, f))

    if not found:
        print("没找到 .ps1 文件")
        return

    for p in found:
        rel = os.path.relpath(p, root)
        status = fix(p)
        chk = verify(p)
        flag = "[OK] " if chk == "OK" else "[ERR]"
        print("%s %-30s %s  |  %s" % (flag, rel, status, chk))


if __name__ == "__main__":
    main()
