# Color Strategy, OKLCH Space, and Contrast Standards

Color encodes hierarchy, mathematical distinction, state, and visual priority.

## 1. Perceptual Color Spaces (OKLCH)

Traditional sRGB and HSL color models are not perceptually uniform: yellow at 50% HSL lightness appears dramatically brighter to human vision than blue at 50% lightness.

Use OKLCH (Lightness, Chroma, Hue) for color generation and palette derivation:
- Lightness (L): Represents perceived brightness (0.0 to 1.0 or 0% to 100%). Equal lightness values across different hues yield equal perceptual brightness.
- Chroma (C): Represents color purity/saturation (0.0 for pure grayscale to ~0.37 for intense gamuts). Keep UI surface chroma low (< 0.04) and accent chroma focused (< 0.20) to prevent visual fatigue.
- Hue (H): The color angle (0 to 360 degrees). Provides predictable complementary (H + 180 deg) and triadic (H + 120 deg) relationships.

## 2. The Semantic Role Palette

Build colors strictly by semantic role rather than collecting arbitrary swatches:

- Canvas Ground: Main background surface (e.g. clean white `#ffffff` or crisp off-white `#f8f9fa` in light mode; deep slate `#0f172a` or `#090d16` in dark mode).
- Elevated Surface: Card, callout, or container fill. In light mode, slightly darker or tinted neutral (e.g. `#f1f5f9`). In dark mode, slightly lighter than canvas ground (e.g. `#1e293b`).
- Primary Text / Ink: High-contrast ink for body text and headings (e.g. `#0f172a` or `#1e293b` on light; `#f8fafc` on dark).
- Secondary Text / Ink: Muted ink for metadata, citations, captions (e.g. `#64748b` on light; `#94a3b8` on dark).
- Primary Accent: Dominant brand/focus color used for primary highlights, active nodes, and key derivations (e.g. academic navy `#1e40af`, cobalt `#2563eb`, or deep slate).
- Secondary Accent: Supporting hue for comparative models, secondary curves, or ablation series (e.g. teal `#0f766e` or amber `#d97706`).
- Alert / Critical: Reserved strictly for errors, key bottlenecks, or negative baselines (e.g. crimson `#b91c1c`).

## 3. The Three-Color Maximum Rule

Every technical presentation palette should use at most 3 chromatic hues:
1. One dominant brand / primary accent (focus nodes, main equations, primary result series).
2. One secondary comparison accent (baseline models, secondary data series, ablations).
3. One optional alert / highlight color (bottlenecks, errors, architectural loss).

Neutrals (grays, off-whites, dark slates) provide the surface structure without counting toward the 3 chromatic colors.

## 4. Contrast Floors and Tinted Neutrals

Maintain strict contrast ratios across all elements:
- Body text against background: Minimum 4.5:1 (WCAG AA standard).
- Large display headings and titles (>= 20pt bold or >= 24pt regular): Minimum 3.0:1.
- Interactive elements, diagrams, borders, and data lines: Minimum 3.0:1.

Tinted Neutrals Rule:
- On colored surfaces, derive secondary text and borders from the foreground or background hue rather than flat, washed-out neutral gray.
- Never use flat gray text on colored backgrounds.

## 5. Dark Mode Elevation Lightness Shifts

In dark interface modes:
- Canvas (ground): Lightness ~12% to 15% (e.g. `#0f172a`).
- Level 1 card / container: Lightness ~18% to 22% (e.g. `#1e293b`).
- Level 2 popover / modal / active node: Lightness ~25% to 30% (e.g. `#334155`).
- Never invert elevation in dark mode by making raised surfaces darker than the canvas.

## 6. Prohibited Color Habits

- Do not use multi-color gradient text on headings or metric figures.
- Do not use colored zero-offset glow halos or blurred colored shadows on dark backgrounds.
- Do not drop floating radial spotlight hazes behind sections.
- Do not default to warm cream or beige as an unexamined reflex.
- Avoid purple-and-cyan neon pairings associated with generic AI templates.

## 7. Curated Color Palettes for Technical Presentations

- Deep Academic Navy:
  - Canvas: `#ffffff`
  - Text: `#1e293b`
  - Primary Accent: `#1e40af` (Deep Blue)
  - Secondary Accent: `#0284c7` (Sky Blue)
  - Alert: `#b91c1c` (Crimson)
- Slate and Amber:
  - Canvas: `#fcfcfc`
  - Text: `#0f172a`
  - Primary Accent: `#334155` (Slate Gray)
  - Secondary Accent: `#d97706` (Amber Gold)
  - Alert: `#dc2626` (Red)
- Forest and Sage:
  - Canvas: `#fbfcfb`
  - Text: `#14291f`
  - Primary Accent: `#166534` (Forest Green)
  - Secondary Accent: `#15803d` (Sage / Emerald)
  - Alert: `#b91c1c` (Wine Red)
- Monochromatic Cobalt:
  - Canvas: `#ffffff`
  - Text: `#0f172a`
  - Primary Accent: `#2563eb` (Cobalt)
  - Secondary Accent: `#64748b` (Slate Neutral)
  - Alert: `#ef4444` (Coral Red)
