"""Fail when a local Markdown link points to a missing repository file."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    missing: list[str] = []
    checked = 0
    for markdown in sorted(root.rglob("*.md")):
        if any(part.startswith(".") or part in {"build", "dist", "venv_lm_adapt"} for part in markdown.relative_to(root).parts):
            continue
        text = markdown.read_text(encoding="utf-8")
        for raw_target in LINK.findall(text):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            local = unquote(target.split("#", 1)[0])
            if not local:
                continue
            checked += 1
            if not (markdown.parent / local).exists():
                missing.append(f"{markdown.relative_to(root)} -> {target}")
    if missing:
        print("Missing local Markdown links:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        raise SystemExit(1)
    print(f"valid local Markdown links: {checked}")


if __name__ == "__main__":
    main()
