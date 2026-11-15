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
    """Tracked .md paths.

    `-z` and `core.quotepath=false` are both deliberate. With git's default
    core.quotepath=true, non-ASCII paths (this repo has several Chinese directory
    names) come back escaped as octal `\\NNN` inside double quotes, which is not a
    usable filename. That only breaks in a fresh clone -- on a machine where
    quotepath was turned off locally, the bug is invisible. It is the reason this
    script passed locally and failed in CI.
    """
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "-c", "core.quotepath=false", "ls-files", "-z", "*.md"],
            capture_output=True).stdout
    except FileNotFoundError:
        return []
    return [p for p in out.decode("utf-8", "replace").split("\0") if p.strip()]


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
