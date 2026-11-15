#!/usr/bin/env python3
"""
Slide counter and structure verification tool for the Typst presentation.
Parses main.typ and sections/*.typ to verify slide counts, sections, and speaker notes.
"""

import os
import re
import sys
from pathlib import Path

def count_slides():
    repo_root = Path(__file__).resolve().parent.parent
    sections_dir = repo_root / "sections"
    main_file = repo_root / "main.typ"

    if not sections_dir.exists():
        print(f"Error: {sections_dir} does not exist.")
        sys.exit(1)

    typ_files = sorted(list(sections_dir.glob("*.typ")))
    total_slides = 0
    total_speaker_notes = 0
    
    print("=" * 70)
    print("SLIDE DECK STRUCTURE & COUNT REPORT")
    print("=" * 70)

    section_stats = []

    for f in typ_files:
        content = f.read_text(encoding="utf-8")
        
        # Count section header slides
        section_headers = len(re.findall(r"#section-header-slide\[", content))
        # Count subheader slides
        subheaders = len(re.findall(r"#subheader-slide\[", content))
        # Count content slides (== Heading)
        content_headings = len(re.findall(r"^==\s+.+$", content, flags=re.MULTILINE))
        # Count explicit slide blocks #slide[
        explicit_slides = len(re.findall(r"#slide\[", content))
        # Count speaker notes
        notes = len(re.findall(r"#speaker-note\[", content))
        
        file_slides = section_headers + subheaders + content_headings + explicit_slides
        total_slides += file_slides
        total_speaker_notes += notes

        section_stats.append({
            "file": f.name,
            "section_headers": section_headers,
            "subheaders": subheaders,
            "content_slides": content_headings + explicit_slides,
            "total_slides": file_slides,
            "notes": notes,
        })
        
        print(f"\nFile: {f.name}")
        print(f"  Section Headers: {section_headers}")
        print(f"  Subheaders:      {subheaders}")
        print(f"  Content Slides:  {content_headings + explicit_slides}")
        print(f"  Total Slides:    {file_slides}")
        print(f"  Speaker Notes:   {notes}")

    print("\n" + "=" * 70)
    print(f"GRAND TOTAL SLIDES (Sections): {total_slides} (+ 1 Title Slide in main.typ)")
    print(f"TOTAL SPEAKER NOTES:           {total_speaker_notes}")
    print("=" * 70)

if __name__ == "__main__":
    count_slides()
