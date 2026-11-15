#!/usr/bin/env python3
"""
Slide preview rendering tool.
Renders specific page(s) or all pages to PNG in .tmp/ for quick visual inspection.
Usage:
  python3 tools/render_preview.py 5        # renders page 5
  python3 tools/render_preview.py 1-10     # renders pages 1 to 10
  python3 tools/render_preview.py all      # renders all pages
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def render_preview(page_spec="1"):
    repo_root = Path(__file__).resolve().parent.parent
    main_file = repo_root / "main.typ"
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"[INFO] Rendering preview for page(s): {page_spec}")

    if page_spec == "all":
        out_pattern = str(tmp_dir / "slide-{0p}.png")
        cmd = ["typst", "compile", "--format", "png", "--ppi", "144", str(main_file), out_pattern]
    else:
        out_pattern = str(tmp_dir / "slide-{0p}.png")
        cmd = ["typst", "compile", "--format", "png", "--ppi", "144", "--pages", page_spec, str(main_file), out_pattern]

    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print("[ERROR] Compilation failed:")
        print(res.stderr)
        sys.exit(1)

    rendered_pngs = sorted(list(tmp_dir.glob("slide-*.png")))
    print(f"[SUCCESS] Rendered {len(rendered_pngs)} slide preview(s) in .tmp/:")
    for p in rendered_pngs:
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    page_arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    render_preview(page_arg)
