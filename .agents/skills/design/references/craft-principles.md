# Design Craft Principles, Mechanical Floors, and Anti-Pattern Catalog

Comprehensive rules for visual craft, aesthetic intentionality, and eliminating the signatures of generic automated generation.

## 1. The Principle of Intentionality

Every visual element (container, border, color tint, shadow, icon, divider) must earn its place:
- If a border, background tint, divider line, or shadow does not clarify hierarchy or improve legibility, remove it.
- Restraint in chrome; precision in content, typography, and diagramming.
- Avoid aesthetic defaults that signal unexamined automated generation.

## 2. The Comprehensive Anti-Pattern Catalog

Refuse the following automated design habits:

### Structural and Container Slop
- Nested Containers: Do not place cards inside cards, or boxes inside shaded containers. Group related items with spacing and proximity rather than stacked borders.
- The Universal AI Card Template: Avoid the reflexive layout of a small rounded-square icon tile stacked directly above a heading. Place icons in natural text flow or use varied compositions.
- Non-Concentric Corner Radii: When nesting a rounded element inside a padded container, never use the same border radius. Set inner radius to outer radius minus container padding: R_inner = max(0, R_outer - padding).
- Monotonous Spacing: Do not use the exact same gap between all elements. Create rhythm with tight internal groupings (4-8pt) and generous section separations (24-48pt).
- Column Imbalance: In multi-column slides, do not let one column stretch far taller than its sibling, leaving the other floating in dead space. Balance column heights or flow long text below.
- Cramped Container Padding: Never place text flush against container borders. Always provide at least 8pt to 12pt of padding inside bordered or colored boxes.
- Body Text at Screen Edges: Never let body text touch the slide or viewport edge without margin insets (minimum 16pt).
- Edge-Flush Cards: Do not span cards or containers fully from edge to edge without lateral gutters unless designing a full-bleed hero banner.

### Typographic Slop
- Eyebrow Chips and Kickers: Avoid placing tiny uppercase letter-spaced labels or pill badges directly above headings. Let the heading state the complete takeaway.
- Flat Type Hierarchy: Avoid font sizes that are too close together. Ensure at least a 1.25x ratio between adjacent hierarchical levels.
- Oversized Long Headlines: Do not set full-sentence headlines at massive display sizes that overwhelm the slide. If a title is long, scale it to title size (22-26pt) rather than display size (36pt+).
- Overused Typefaces: Avoid converging on the same handful of generic fonts (Inter, Roboto, Geist, Space Grotesk) by reflex. Choose a face that reinforces technical precision or academic rigor.
- Italic Serif Display: Do not use oversized italic serif display headlines by reflex as a substitute for visual structure.
- Gratuitous Numbering: Do not label sections or bullets with "01", "02", "03" unless the sequence itself conveys necessary chronological or algorithmic meaning.
- All-Caps Body Passages: Long sentences in uppercase remove ascenders and descenders, severely impairing reading speed. Reserve uppercase strictly for short badges or acronyms.
- Crushed or Wide Tracking: Do not crush letter-spacing destructively (< -0.04em) or over-track body text (> +0.05em).
- Justified Text without Hyphenation: Avoid fully justified text without hyphenation, which produces irregular word spacing and rivers of white.

### Color and Surface Slop
- Gradient Text: Do not use multi-color gradient fills on text or metric numbers. Use solid colors and typographic weight for emphasis.
- AI Cliché Palettes: Avoid generic dark mode with purple/violet gradients and cyan neon accents.
- The Reflexive Cream/Beige Background: Avoid defaulting to warm cream or beige as an unexamined "safe" surface. Choose backgrounds from an intentional palette.
- Glowing Shadow Accents and Halos: Do not use zero-offset colored glow halos or chromatic blurred shadows on dark surfaces. Use soft, offset neutral depth shadows.
- Radial Spotlight Haze: Do not place soft colored radial gradients or spotlight hazes behind headings or sections as decorative atmosphere.
- Dark Gradient Ground: Do not build dark background surfaces from steep vertical linear gradients instead of clean solid surface tokens.
- Gray Text on Colored Backgrounds: Never place neutral gray text on a colored background (which looks washed out). Use a darker/lighter shade of the surface color or high-contrast white.
- Side-Stripe Callout Overuse: Avoid putting thick vertical colored borders on every text block or card. Use subtle background tints or whitespace instead.
- Zero-Blur Hard Offset Shadows: Do not use hard-edged block shadows (such as 4px 4px 0) outside intentionally neobrutalist designs.

### Motion and Decorative Slop
- Bounce or Elastic Easing: Do not use bouncy or elastic animation curves. Real objects decelerate smoothly; use exponential ease-out curves.
- Decorative Pulsing Dots: Do not add pulsing status dot animations unless tied directly to live streaming data.
- Fake Blinking Carets: Do not animate a blinking text cursor in static headers where no real text input exists.
- Shape-Assembled Illustrations: Avoid large vector illustrations built from piles of primitive shapes that resemble amateur clip art. Use clean vector diagrams, charts, or professional assets.
- Auto-Scrolling Marquees: Do not use continuously scrolling tickers that force users to wait or chase text.

### Copy and Cadence Slop
- Em-Dash Overuse: Avoid saturating prose with em-dashes. Prefer standard commas, colons, periods, or parentheses.
- Marketing Buzzwords: Eliminate filler verbs and empty praise (streamline, empower, supercharge, cutting-edge, next-generation, seamless). State directly what the model or algorithm does.
- Aphoristic Cadence: Avoid repetitive short rebuttal slogans across sections ("X. No Y.", "Not a feature. A platform.").

## 3. Geometric Rules and Elevation Mechanics

- Nested Border Radii Formula: R_inner = max(0, R_outer - padding). For example, a card with border-radius 12pt and padding 8pt requires an inner child element radius of 4pt (12 - 8 = 4).
- Declare Elevation Once (No Ghost Cards): Declare depth using either a subtle border or a soft elevation shadow, never both fighting each other on the same card. A distinct border combined with a wide soft shadow produces visual mud.
- Natural Elevation Physics: Real light sources originate from above. Elevation requires a positive vertical (Y) offset, soft blur radius, and neutral alpha transparency. Prohibit zero-offset colored glow rings.
- Dark Mode Elevation: In dark interfaces, elevated surfaces become slightly lighter in lightness value, never darker, reflecting proximity to the ambient light source.
- Theme Un-drawn UI Elements: Elements not drawn by default (focus rings, text selection color, custom scrollbars, underline offset, tabular numeral alignment) must be themed from the palette rather than left to inconsistent system defaults.

## 4. The Mechanical Craft Floor

Every built slide or document must satisfy these objective quality checks:
- Contrast ratio: Body text against surface >= 4.5:1; large display titles >= 3.0:1; interactive borders >= 3.0:1.
- Depth integrity: Shadows carry both a positive Y-offset and a soft blur; zero-offset colored halos are prohibited.
- Spacing rhythm: Generous outer margins with tight internal groupings. More vertical space above headings than below them.
- Measure and line length: Prose lines between 45 and 75 characters.
- Text overflow zero-tolerance: Run real content at the target aspect ratio; never allow text to clip or overflow slide bounds.

## 5. The Skeleton Test

Strip all text and copy out of the planned slide or document layout:
- Does the bare structural skeleton still convey the hierarchy, focal element, and reading sequence?
- If the layout only makes sense once words are filled in, the design relies solely on font size rather than visual structure.
- Re-architect the skeleton until spatial relationships, diagram placements, and container groupings speak for themselves.

## 6. The Squint Test

Evaluate the rendered layout with detail blurred:
- Is the primary focal point immediately obvious?
- Does the secondary element support the primary element logically?
- Are distinct conceptual groups visibly separated from one another?
- Does the eye naturally follow the intended reading path?
