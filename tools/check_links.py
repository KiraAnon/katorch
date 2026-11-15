#!/usr/bin/env python3
"""Check that every relative markdown link in the tracked .md files resolves.

Used by CI, but runs anywhere:

    python tools/check_links.py

Exits non-zero if any link points at a file that does not exist. Anchors are ignored
(nothing here verifies that the heading actually exists), as are http(s)/mailto targets.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "data:")


def tracked_markdown():
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "*.md"],
                             capture_output=True, text=True, encoding="utf-8").stdout
    except FileNotFoundError:
        return []
    return [line for line in out.splitlines() if line.strip()]


def main():
    files = tracked_markdown()
    if not files:
        print("no markdown files found (not a git checkout?)")
        return 0

    broken = 0
    for rel in files:
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            print("unreadable: %s (%s)" % (rel, exc))
            broken += 1
            continue
        for match in LINK.finditer(text):
            target = match.group(1).strip()
            if target.startswith(SKIP_PREFIX):
                continue
            target = target.split("#")[0]
            if not target:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if not os.path.exists(resolved):
                print("broken: %s -> %s" % (rel, match.group(1)))
                broken += 1

    print("checked %d markdown files, %d broken link(s)" % (len(files), broken))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
