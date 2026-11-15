# Visual Polish, Design Critique, and Triage Protocol

Methodology for systematic evaluation, multi-pass critique, and prioritized refinement.

## 1. The Triage Hierarchy

When refining or fixing a slide deck or document, fix defects in strict priority order:

1. Structural Deficiencies and Blockers:
   - Broken layout, text overflowing slide bounds, missing content.
   - Narrative flow failures (failing the Ghost Deck Test).
2. Hierarchy and Comprehension Drift:
   - Weak visual hierarchy where headings do not stand out.
   - Ambiguous focal points (more than one element shouting for attention).
   - Missing or unclear action titles.
3. Design System and Style Inconsistencies:
   - Palette drift, unapproved font size jumps, erratic padding/margins.
   - Mismatched vector icon weights or non-standard bullet styles.
   - Non-concentric nested border radii.
4. Optical Alignment and Spacing Polish:
   - Micro-alignments, whitespace rhythm balancing, optical centering.
   - Contrast validation (verifying body text >= 4.5:1, headings >= 3.0:1).
5. Source Code and Markup Cleanup:
   - Removing dead rules, organizing functions, formatting source code.

## 2. Cognitive Load Assessment

Evaluate each slide against cognitive load limits:
- The 4-Item Rule: A single slide must not present more than 4 primary decision points or competing visual blocks at once.
- Information Chunking: Group related data points into distinct visual modules rather than a continuous wall of bullets.
- Scannability: An audience member glancing at the slide for 3 seconds should identify the core takeaway, the primary diagram, and the supporting conclusion.

## 3. Systematic 5-Dimension Audit and Severity Grading

When conducting an audit pass, evaluate the document across 5 dimensions:

- Accessibility (A11y): Contrast floors (>= 4.5:1 body, >= 3.0:1 display), reading order, legible font sizes (>= 16pt body, >= 11pt captions), and color-independent meaning.
- Layout and Responsiveness: Zero canvas overflow, balanced column heights, edge margin gutters (>= 16pt), and container padding floors.
- Theming and Color: Strict adherence to semantic token roles, 3-hue maximum, OKLCH perceptual lightness consistency, and dark mode elevation shifts.
- Typography and Scannability: Proportional hierarchy (>= 1.25x steps), measure (45-75ch), proportional leading, tabular figures for numbers, and action titles.
- Implementation Integrity: Zero anti-patterns (no gradient text, no non-concentric radii, no ghost cards, no nested containers, no AI cliché slop).

Severity Tagging for Defects:
- P0 (Blocking): Text overflow, broken layout, clipping, or unreadable contrast (< 3.0:1). Fix immediately.
- P1 (Major): WCAG AA contrast failure, ambiguous visual hierarchy, or missing action title. Fix before delivery.
- P2 (Minor): Mismatched corner radii, inconsistent spacing gap, or non-tabular numbers. Fix in polish pass.
- P3 (Polish): Micro-spacing tweak, optical alignment refinement, or subtle color tuning.

## 4. Pre-Delivery Polish Checklist

- [ ] Ghost Deck Test: Reading slide titles sequentially conveys the complete argument.
- [ ] Action Titles: Every content slide heading is an active conclusion assertion.
- [ ] Squint Test: The primary focal point is unambiguous on every slide.
- [ ] Skeleton Test: Layout conveys structural hierarchy even when copy is removed.
- [ ] Craft Floor: Contrast >= 4.5:1 for body text, >= 3.0:1 for large display titles.
- [ ] Zero Overflows: No text clips or exceeds slide margins.
- [ ] Anti-Pattern Free: No nested cards inside cards, no eyebrow kickers, no gradient text, no emoji bullets.
- [ ] Concentric Radii: Inner corner radii equal outer radius minus container padding.
- [ ] Color Restraint: Palette uses at most 3 semantic hues with dominant neutrals.
- [ ] Typographic Scale: Proportional hierarchy maintained across all slides with >= 1.25x step ratio.
- [ ] Native Diagrams: All architectures and flowcharts are drawn natively in Fletcher/Typst.
- [ ] Direct Data Labeling: Charts and graphs use direct series labels rather than detached legends.
- [ ] Tabular Figures: Numbers in tables and metric comparisons use aligned tabular figures.
- [ ] Speaker Notes: Every content slide contains a complete `#speaker-note[...]` block.
- [ ] Concluding Slide: Final slide summarizes contributions for Q&A (no empty "Thank You").
