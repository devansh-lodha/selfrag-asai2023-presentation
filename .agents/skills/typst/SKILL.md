---
name: typst
description: Generate idiomatic Typst (.typ) code, edit and troubleshoot Typst documents and projects, and answer Typst syntax/reference questions. Use when working with .typ files or when the user explicitly asks for Typst document creation, editing, debugging, compilation, formatting, template work, or package usage.
---

# Typst Skill

Guide for generating, modifying, and troubleshooting Typst documents and packages.

## Minimal document structure

```typst
#set document(title: "Document Title", author: "Author Name")
#set page(paper: "a4", margin: (x: 2cm, y: 2.5cm), numbering: "1")
#set text(font: "Linux Libertine", lang: "en", size: 11pt)
#set par(justify: true)

= Section Title

Typst markup paragraph text with *bold*, _emphasis_, and @reference syntax.

== Subsection Title

Numbered and unnumbered lists:
- First bullet item
- Second bullet item

$ sum_(i=1)^n i = frac(n(n+1), 2) $ <sum-formula>
```

## Core syntax and distinctions

Data structures:
- Arrays: `(item1, item2, item3)`. Empty is `()`. Single item is `(item,)`. Access with `arr.at(index)`.
- Dictionaries: `(key1: "val1", key2: "val2")`. Empty is `(:)`. Access with `dict.at("key")` or `dict.key`.
- Content blocks: `[content]`. Content represents structured document layout, not raw strings.
- Code blocks: `{ let x = 1; x + 2 }`.
- Typst has no separate tuple type; all parenthesized comma lists are arrays.

Modes and hash `#` usage:
- Markup mode: Text is content by default. Use `#` to start code expressions (such as `#let`, `#if`, `#for`, `#box()`, `#figure()`).
- Math mode: Enclosed in `$ ... $`. Code requires `#`. Shorthands and function calls use unquoted names.
- Code mode: Inside code blocks `{ ... }` or argument lists `(...)`. Do not prefix function calls or identifiers with `#` in code mode. Example: `grid(columns: 2, image("a.png"), image("b.png"))`.

Styling and transformation:
- Set rules: `#set <element>(<param>: <value>)` configure default properties within scope.
- Show-set rules: `#show <selector>: set <element>(<param>: <value>)` selectively configure properties for matching elements.
- Show transform rules: `#show <selector>: it => <transformed content>` reshape or replace matched elements.

Context and introspection:
- Use `context` expression to access dynamic document information (such as `counter`, `state`, `locate`, `measure`, and `query`).
- Contextual queries and measurements are resolved during layout iteration passes.

## Standard workflows

Creating a document:
- Start with a minimal `#set` configuration for document, page, text, and paragraph properties.
- Use native headings (`=`, `==`, `===`) for document hierarchy.
- Split multi-file documents using `#include "path/to/file.typ"`.
- Import reusable libraries and functions with `#import "path/to/module.typ": *`.

Modifying and editing documents:
- Locate target selectors, content blocks, or set rules.
- Confirm function parameters and signatures against reference documentation in `references/`.
- Validate syntax and types before compilation.

Quality and formatting checks:
- Compile using `typst compile <file.typ> <output.pdf>`.
- Format using `typstyle` if available in the environment (`command -v typstyle`).
- Run `typstyle --check <file>` to verify formatting.
- Inspect proposed changes with `typstyle --diff <file>` prior to applying `typstyle -i <file>`.

Probing uncertain behavior:
- For runtime behavior verification without creating scratch files, run `typst eval '<expression>'`.

## Common pitfalls

- Accessing arrays with square brackets (use `arr.at(i)` rather than `arr[i]`).
- Using `#` inside function argument lists (write `figure(image("x.png"))`, not `figure(#image("x.png"))`).
- Omitting `#` in markup blocks when invoking functions or variables (write `[Total: #total]`, not `[Total: total]`).
- Using LaTeX syntax in Typst source (do not use `\begin{...}`, `\section`, `\cite`, or `\frac`; use native markup).
- Creating singleton arrays without a trailing comma (write `(elem,)`, not `(elem)`).
- Calling missing functions (Typst uses `table` rather than `tabular`, `math.cases` for piecewise functions).

## Documentation references

The bundled documentation files in `references/` provide detailed references:

Tutorials:
- `references/tutorial/1-writing.typ`: Markup basics, functions, labels, citations, and inline math
- `references/tutorial/2-formatting.typ`: Set rules, page setup, text formatting, and headings
- `references/tutorial/3-advanced.typ`: Templates, show rules, and layout logic
- `references/tutorial/4-template.typ`: Packaging reusable templates

Language Reference:
- `references/reference/language/syntax.typ`: Markup, math, and code modes, comments, escape sequences, identifiers
- `references/reference/language/scripting.typ`: Variables, expressions, control flow, functions, loops, modules, types
- `references/reference/language/styling.typ`: Set rules, show rules, selectors, contextual styling
- `references/reference/language/context.typ`: Context expressions, state inspection, engine compilation cycles

Standard Library Reference:
- `references/reference/library/foundations.typ`: Core types, assertions, sys variables, eval, label, type conversions
- `references/reference/library/model.typ`: Document metadata, headings, figures, lists, outline, table, citations, bibliography
- `references/reference/library/layout.typ`: Page, block, box, grid, stack, columns, align, pad, measure, transformations
- `references/reference/library/text.typ`: Font families, text shaping, weights, sizes, raw text, hyphenation
- `references/reference/library/math.typ`: Math equations, alignments, accents, matrices, fractions, symbols
- `references/reference/library/visualize.typ`: Shapes (rect, circle, ellipse, line, polygon, path), color, gradient, pattern, stroke, image
- `references/reference/library/introspection.typ`: Counter, state, query, locate, metadata
- `references/reference/library/data-loading.typ`: File readers for CSV, JSON, TOML, YAML, XML, CBOR, and raw text
- `references/reference/library/symbols.typ`: Symbol definitions and unicode shorthand tables

Guides:
- `references/guides/page-setup.typ`: Margins, headers, footers, pagination, multiple columns
- `references/guides/tables.typ`: Grid and table formatting, stroke styling, semantic alignment
- `references/guides/accessibility.typ`: Semantic tagging, alt text, language metadata, PDF/UA compliance
- `references/guides/for-latex-users.typ`: Direct mapping from LaTeX paradigms to Typst idioms

Export Formats:
- `references/reference/export/pdf.typ`: PDF generation options, standards, embedded metadata
- `references/reference/export/html.typ`: HTML generation features and output control
- `references/reference/export/svg.typ`: SVG export options
- `references/reference/export/png.typ`: PNG raster export options
- `references/reference/export/bundle.typ`: Multi-file bundles and package assets

Changelog:
- `references/changelog/index.typ`: Complete version-by-version changes and migration notes
