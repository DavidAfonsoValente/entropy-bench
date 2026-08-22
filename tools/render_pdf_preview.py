#!/usr/bin/env python3
"""Render PDF pages to PNG previews with Ghostscript for visual QA."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_pattern", help="Ghostscript pattern, e.g. /tmp/page-%02d.png")
    parser.add_argument("--dpi", type=int, default=144)
    args = parser.parse_args()
    subprocess.run(
        [
            "gs",
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{args.dpi}",
            f"-sOutputFile={args.output_pattern}",
            str(args.pdf),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
