# Slide Hardening and Edge Case Handling

Guidelines for ensuring slides and documents remain robust against extreme text lengths, complex math, and rendering constraints.

## 1. Text Hardening and Overflow Prevention

- Long Word and Equation Handling: Long compound words or extended inline formulas must not clip outside columns. Use line breaks or smaller inline math scopes.
- Multi-Line Heading Wrapping: Slide titles (action titles) must be tested against 2-line wraps to ensure they do not collide with header badges or slide numbering.
- Fixed 16:9 Canvas Constraints: Unlike responsive web pages, slides have a fixed, non-scrolling viewport. Content that exceeds vertical height is a hard defect.
- Text Trimming Priority: When text exceeds available height, cut words rather than reducing font size below the 16pt floor.
- No Body Text at Edges: Guarantee at least 16pt to 24pt margins between content containers and canvas boundaries.

## 2. Mathematical Notation Hardening

- Long Equation Multi-Line Alignment: Multi-step derivations must use aligned equations (`&`) across multiple lines rather than a single overflowing horizontal block.
- Variable Consistency: Ensure notation is identical between slide text, diagram labels, and mathematical formulas (for example, if hidden state is $h_t$, do not refer to it as $H_t$ or $s_t$ in the text).
- Notation Definition Integrity: Every subscript, superscript, and tensor dimension must be clearly identified on the slide or in accompanying speaker notes.

## 3. Font and Asset Resilience

- Font Fallback Safety: Always test typography without relying on rare, uninstalled local system fonts. Prefer standard accessible families (Linux Libertine, New Computer Modern, Inter, Roboto).
- Vector Asset Integrity: Never rely on external network URLs during compilation; all vector assets and images must be stored locally under `assets/`.
