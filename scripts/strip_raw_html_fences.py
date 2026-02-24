#!/usr/bin/env python3

from pathlib import Path
import sys


def strip_raw_html_fences(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    in_raw_html = False

    for line in lines:
        if not in_raw_html and line.strip() == "```{=html}":
            in_raw_html = True
            continue

        if in_raw_html and line.strip() == "```":
            in_raw_html = False
            continue

        out.append(line)

    path.write_text("".join(out), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: strip_raw_html_fences.py <path>", file=sys.stderr)
        return 1

    strip_raw_html_fences(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
