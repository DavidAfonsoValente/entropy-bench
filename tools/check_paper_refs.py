"""Fail when the built paper has an unresolved cross-reference or citation.

`make` exits 0 on an undefined `\ref`: LaTeX prints "Figure ??" and warns, but does not fail. That
is how the 11-page cut shipped with the pipeline figure deleted and two references to it left
behind -- the build was green and the PDF was wrong. This gate reads the build log instead of
trusting the exit code.

Usage:
    python tools/check_paper_refs.py [paper_sota.log]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = [
    (re.compile(r"^LaTeX Warning: Reference `([^']+)' on page (\S+) undefined", re.M), "undefined reference"),
    (re.compile(r"^LaTeX Warning: Citation `([^']+)' on page (\S+) undefined", re.M), "undefined citation"),
    (re.compile(r"^LaTeX Warning: (There were undefined references)", re.M), "summary"),
]


def main() -> None:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_sota.log")
    if not log.exists():
        print(f"paper refs: {log} absent -- run `make` first", file=sys.stderr)
        raise SystemExit(1)
    text = log.read_text(encoding="utf-8", errors="ignore")

    problems = []
    for pattern, kind in PATTERNS:
        if kind == "summary":
            continue
        for match in pattern.finditer(text):
            problems.append(f"{kind}: `{match.group(1)}` on page {match.group(2)}")

    for problem in sorted(set(problems)):
        print(f"PROBLEM: {problem}", file=sys.stderr)
    print(f"paper refs: PROBLEMS {len(set(problems))}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
