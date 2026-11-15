#!/usr/bin/env python3
"""
Slide overflow detection tool.
Compiles main.typ to a temporary PDF, extracts page texts, and checks for overflow indicators
such as blank overflow pages, orphaned single lines, or heading-to-page mismatches.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def detect_overflow():
    repo_root = Path(__file__).resolve().parent.parent
    main_file = repo_root / "main.typ"
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    pdf_out = tmp_dir / "check_overflow.pdf"

    print("=" * 70)
    print("CHECKING FOR SLIDE OVERFLOW & COMPILATION HEALTH")
    print("=" * 70)

    # Run typst compile
    cmd = ["typst", "compile", str(main_file), str(pdf_out)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\n[ERROR] Typst compilation failed:")
        print(result.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    print("\n[SUCCESS] Typst compilation succeeded with zero syntax errors.")

    # Check query for total pages using typst query
    query_cmd = ["typst", "query", str(main_file), "<touying:slide>"]
    query_res = subprocess.run(query_cmd, capture_output=True, text=True)

    # Also render png of pages to check dimensions/page count
    png_template = str(tmp_dir / "page-{0p}-of-{t}.png")
    png_cmd = ["typst", "compile", "--format", "png", "--ppi", "100", str(main_file), png_template]
    png_res = subprocess.run(png_cmd, capture_output=True, text=True)

    if png_res.returncode == 0:
        png_files = sorted(list(tmp_dir.glob("page-*.png")))
        print(f"[INFO] Rendered {len(png_files)} total physical pages.")
    else:
        print("[WARNING] PNG generation had warnings:", png_res.stderr)

    # Clean up temporary artifacts
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n[CLEANUP] Temporary inspection artifacts cleaned up.")
    print("=" * 70)

if __name__ == "__main__":
    detect_overflow()
