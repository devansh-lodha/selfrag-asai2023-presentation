# Typst & Touying Slide Technical Standards

Universal technical constraints and syntax rules for authoring slides in Typst and Touying.

## 1. Typst vs. Markdown Syntax Reference

| Element | Typst | Markdown | Notes |
|---|---|---|---|
| Bold | `*bold*` | `**bold**` | Typst uses single asterisks |
| Italic | `_italic_` | `_italic_` or `*italic*` | Typst uses single underscores |
| Inline code | `` `code` `` | `` `code` `` | Fenced with single backticks |
| Code block | ` ```lang ... ``` ` | ` ```lang ... ``` ` | Fenced code block |
| Headings | `= H1`, `== H2`, `=== H3` | `# H1`, `## H2`, `### H3` | Typst uses `=` |
| Bullet list | `- item` | `- item` | Standard dash |
| Numbered list | `+ item` | `1. item` | Typst uses `+` |
| Term list | `/ Term: Description` | N/A | Typst description list |
| Images | `#image("file.png", width: 80%)` | `![alt](file.png)` | Function call |
| Label | `<my-label>` | N/A | Label for cross-referencing |
| Reference | `@my-label` | N/A | References a label |
| Math inline | `$x^2 + y^2$` | `$x^2 + y^2$` | Enclosed in dollar signs |
| Math block | `$ x^2 + y^2 $` (spaced) | `$$...$$` | Spaces inside dollar signs denote block math |
| Line break | `\` | `<br>` or double space | Trailing backslash |
| Slide break | `---` or `#pagebreak()` | `---` | `---` creates a slide break in Touying |
| Dash | `--` (en-dash), `---` (use with care) | `--`, `---` | In slide text, use `--` or `parentheses`; `---` triggers slide breaks |

## 2. Heading Hierarchy in Touying

- Level 1 (`=`): Creates a major section divider slide (e.g., `= 1. Sequence-to-Sequence Architecture`).
- Level 2 (`==`): Creates an individual content slide (e.g., `== Multi-layer LSTMs map sequences to fixed vectors`).
- Level 3 (`===`): Creates an in-slide subsection or card title (does NOT create a new slide).

Heading modifiers:
- `== Slide Title <touying:hidden>`: Renders the slide title on the slide but hides it from table of contents and outlines.
- `== <touying:hidden>`: Inserts a blank title slide and resets the previous heading context.
- `== Slide Title <touying:unnumbered>`: Excludes heading from section numbering.

## 3. P0 Hard Rules for Presentation Authoring

- Do not use `#` for headings: `#` starts a code expression in Typst. Headings must use `=`, `==`, `===`.
- Do not use `**bold**`: Use `*bold*` with single asterisks.
- Do not use Unicode math symbols directly in text: Write math in `$ ... $` blocks (e.g., write `$alpha$`, `$beta$`, `$theta$`, not `α, β, θ`).
- Arrays vs Content blocks: Arrays use parentheses `(item1, item2)`; content blocks use square brackets `[content]`.
- Single-item arrays: Require a trailing comma `(item,)`.
- Do not hardcode ad-hoc `set page(...)` rules in content slides: Use `config-page(...)` within the theme configuration.
- Speaker notes: Every content slide must have a `#speaker-note[...]` block.

## 4. Anti-Patterns to Avoid

- Wall of text: Avoid putting more than 40-50 words on a single slide.
- Topic-only slide titles: Never write `== Methodology` or `== Architecture`; always state the takeaway in an action title.
- Empty conclusion slides: Never create a slide that only says "Thank You" or "Questions?". End with key contributions.
- Shrinking font sizes: If content does not fit with body font >= 20pt, split into multiple subslides or trim content.
