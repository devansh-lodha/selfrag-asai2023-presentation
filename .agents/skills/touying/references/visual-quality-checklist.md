# Visual Quality & Review Checklist

Use this checklist before finalizing any Typst/Touying slide deck.

## 1. Content and Narrative Checks

- [ ] Ghost Deck Test: Reading slide titles sequentially conveys the entire argument from motivation to conclusion.
- [ ] Action Titles: Every content slide heading is a complete assertion, not a topic label.
- [ ] Word Count: All slides have <= 40-50 words of body text.
- [ ] Speaker Notes: Every content slide has a meaningful `#speaker-note[...]` block.
- [ ] Conclusions Slide: The final content slide summarizes key findings and contributions (no empty "Thank You" slide).

## 2. Layout and Typography Checks

- [ ] No Text Overflows: Content does not clip or overflow slide margins.
- [ ] Readable Font Sizes: Body text is >= 20pt, exhibit annotations >= 16pt, captions >= 14pt.
- [ ] Color Restraint: Palette uses at most 3 semantic colors (primary, accent, alert).
- [ ] White Space: Slides have sufficient margins and breathing room.
- [ ] High Resolution Exhibits: All architecture diagrams and charts are native Fletcher/Typst constructs, not low-res screenshots.

## 3. Technical Typst Checks

- [ ] Syntax: Single asterisks `*bold*` used for bolding, not `**bold**`.
- [ ] Math Mode: All mathematical symbols are written inside `$ ... $` blocks, not raw Unicode characters.
- [ ] Arrays and Dictionaries: Proper comma notation used for arrays `(a, b)` and singletons `(a,)`.
- [ ] Slide Breaks: Literal `---` used only for slide breaks, not in body sentences (use `--` or parentheses).
- [ ] References and Citations: All citations use `@key` syntax with `#magic.bibliography` or `#bibliography`.
- [ ] Compilation: Document compiles cleanly without warnings using `typst compile main.typ presentation.pdf`.
