---
name: touying
description: Author, refactor, and troubleshoot Typst slide decks using the Touying presentation engine. Use when creating or modifying presentations, slides, animations, themes, speaker notes, and dynamic content in Typst.
---

# Touying Skill

Guide for authoring Typst presentations with Touying.

## Quick start

Import Touying and a theme, then initialize the theme with `#show: <theme>.with(...)`.

```typst
#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [Simple slides],
)

= First Section

== First Slide

Slide content goes here.
```

Working templates are available under `assets/examples/`:
- `assets/examples/simple.typ`: Minimal template
- `assets/examples/default.typ`: Bare unstyled template
- `assets/examples/metropolis.typ`: Metropolis theme template
- `assets/examples/stargazer.typ`: Stargazer theme template
- `assets/examples/university.typ`: University theme template
- `assets/examples/dewdrop.typ`: Dewdrop theme template
- `assets/examples/aqua.typ`: Aqua theme template
- `assets/examples/example.typ`: Comprehensive feature showcase

## Architecture and code organization

Use a single entry file (such as `main.typ`) that applies `#show` and configuration calls.

For multi-file slide decks, use the separation pattern:
- `globals.typ` or `config.typ`: Shared configuration and imports
- `main.typ`: Entry file applying theme and show rules
- `sections/`: Individual content files included via `#include`

Single-file structure:
```
├── globals.typ
├── main.typ
└── content.typ
```

Multi-file structure:
```
├── globals.typ
├── main.typ
└── sections/
    ├── section1.typ
    └── section2.typ
```

Prefer Touying configuration functions over direct low-level commands:
- Use `config-page` instead of direct `set page(...)`
- Use `config-common`, `config-info`, `config-colors`, `config-methods`, and `config-store` for global options
- Use headings for slide boundaries; use `config-common(slide-level: n)` to set which heading level creates slides
- Wrap custom slide functions with `touying-slide-wrapper`

## Core API

Slides:
- `touying-slides`: Top-level slide processor
- `slide`: Base slide generator
- `touying-slide`: Custom slide helper
- `touying-slide-wrapper`: Wrapper function to register slide components
- `empty-slide`: Creates slides without header or footer

Dynamics and animations:
- `pause`: Simple sequential reveal marker
- `meanwhile`: Concurrent reveal marker
- `uncover`: Reserves layout space while hidden
- `only`: Does not reserve layout space while hidden
- `effect`: Custom dynamic transformer
- `alternatives`, `alternatives-match`, `alternatives-fn`, `alternatives-cases`: Switch between content variants per subslide

Configuration:
- `config-common`: General options (slide-level, handout, zero-margin-header, datetime-format, show-notes-on-second-screen)
- `config-page`: Page parameters (margin, header, footer, background, fill)
- `config-info`: Metadata (title, subtitle, author, date, institution)
- `config-colors`: Theme color palette (primary, secondary, neutral, etc.)
- `config-methods`: Functional hooks (init, alert, cover, transform)
- `config-store`: Arbitrary dictionary storage for custom themes
- `appendix`: Freezes total slide counter for backup slides

Utilities:
- `utils.fit-to-height`, `utils.fit-to-width`: Fit content to dimension bounds
- `utils.semi-transparent-cover`: Cover hidden content with transparency
- `utils.methods(self)`: Unpacks dynamic methods (`uncover`, `only`, `alternatives`) in callback slides
- `utils.display-current-short-heading`: Displays heading text for breadcrumbs and navigation

Components:
- `components.side-by-side`: Multi-column slide layout
- `components.adaptive-columns`: Dynamic multi-column outline
- `components.progressive-outline`: Outline with current section highlighted
- `components.custom-progressive-outline`: Parameterized progressive outline

Integrations:
- `touying-reducer`: State reducer for complex integrations
- `touying-equation`: Math equation animation helper
- `touying-mitex`: LaTeX math support
- `speaker-note`: Presenter notes collector
- `pdfpc.*`: PDF Presenter Console export utilities

## Slide control and navigation

- Use `<touying:hidden>` on a heading to create a slide while hiding the heading from outlines and navigation.
- Use `<touying:unnumbered>`, `<touying:unoutlined>`, `<touying:unbookmarked>`, or `<touying:skip>` for fine-grained heading control.
- Use `== <touying:hidden>` to insert a blank title slide and reset the previous heading context.
- Use `---` or `#pagebreak()` to split slides without creating headings.
- Set `config-common(handout: true)` to compile a handout version with only the final state of each slide.
- Apply `#show: appendix` before supplementary slides so slide totals remain fixed.

## Dynamic slides and animations

Simple syntax reveals:
```typst
== Simple Reveal

First item

#pause

Second item appears next

#meanwhile

Third item appears simultaneously
```

Callback syntax (preferred for complex logic, nested components, or when mark warnings occur):
```typst
#slide(
  repeat: 3,
  self => [
    #let (uncover, only, alternatives) = utils.methods(self)

    #uncover("1-")[Shown on all subslides]

    #uncover("2-")[Shown on subslides 2 and 3]

    #only("3")[Shown only on subslide 3]

    #alternatives[Variant A][Variant B][Variant C]
  ],
)
```

Equation animations:
- Place `#pause` directly inside `$ ... $` blocks for multi-step math derivations.

## Theming and layout

Configure themes via named arguments in the theme constructor:

```typst
#show: default-theme.with(
  aspect-ratio: "16-9",
  config-common(
    slide-level: 2,
    zero-margin-header: false,
  ),
  config-colors(
    primary: rgb("#1d3557"),
    secondary: rgb("#457b9d"),
  ),
  config-info(
    title: [Presentation Title],
    author: [Author Name],
    date: datetime.today(),
  ),
)
```

Disable automatic section divider slides:
```typst
#show: metropolis-theme.with(
  config-common(new-section-slide-fn: none),
)
```

## Speaker notes and export

- Attach presenter notes using `#speaker-note[...]` inside any slide.
- Display notes for dual-monitor setups with `config-common(show-notes-on-second-screen: right)`.
- Export pdfpc files with `enable-pdfpc: true` and extract using `typst query --field value --one "<pdfpc-file>"`.

## Documentation references

Consult the reference guides in `references/` for detailed options:
- Academic presentation principles: `references/academic-presentation-guide.md`
- Slide layout patterns: `references/slide-layout-patterns.md`
- Technical standards and syntax: `references/slide-technical-standards.md`
- Presentation packages (theorems, pseudocode, numbering): `references/presentation-packages.md`
- Visual review and quality checklist: `references/visual-quality-checklist.md`
- Basics and start: `references/start.md`, `references/intro.md`, `references/faq.md`
- Multi-file projects: `references/tutorials/multi-file.md`
- Layout and headings: `references/tutorials/layout.md`, `references/tutorials/sections.md`, `references/tutorials/code-styles.md`
- Dynamic features: `references/tutorials/dynamic/simple.md`, `references/tutorials/dynamic/complex.md`, `references/tutorials/dynamic/handout.md`, `references/tutorials/dynamic/equation.md`, `references/tutorials/dynamic/waypoints.md`, `references/tutorials/dynamic/cover.md`
- Progress bars and navigation: `references/tutorials/progress/sections.md`, `references/tutorials/progress/navigation.md`, `references/tutorials/progress/counters.md`
- Custom theme authoring: `references/tutorials/build-your-own-theme.md`, `references/tutorials/settings.md`
- Built-in themes: `references/themes/default.md`, `references/themes/metropolis.md`, `references/themes/stargazer.md`, `references/themes/university.md`, `references/themes/dewdrop.md`, `references/themes/aqua.md`, `references/themes/simple.md`, `references/themes/custom.md`
- Third-party integrations: `references/integration/fletcher.md`, `references/integration/cetz.md`, `references/integration/codly.md`, `references/integration/pinit.md`, `references/integration/mitex.md`, `references/integration/theorion.md`
- External tools: `references/external/pdfpc.md`, `references/external/pympress.md`, `references/external/typst-preview.md`, `references/external/touying-exporter.md`
- Changelog: `references/changelog.md`
