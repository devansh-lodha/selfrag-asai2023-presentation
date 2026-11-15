# Data and Chart Visualization Standards

Guidelines for presenting experimental benchmarks, neural network architectures, loss curves, and comparative data tables.

## 1. Direct Labeling over Detached Legends

- Direct Curve and Series Labeling: Place text labels directly adjacent to lines, bars, or scatter points. Never force readers to cross-reference a distant floating legend box.
- Interactive and Animated Series: In progressive slide reveals, highlight the active series label in tandem with the visual curve.
- Proximity Rule: Keep annotations and callouts physically connected to the specific data point or inflection point using clean lines or minimal arrows.

## 2. Table and Matrix Design

- Alignment Rules:
  - Left-align textual columns and model names.
  - Right-align numeric data, BLEU scores, perplexity metrics, and parameter counts.
  - Center short status badges, checks, or cross markers.
- Tabular Lining Figures: Ensure numeric data uses monospaced or tabular figures so decimal points align vertically across rows.
- Minimal Table Chrome:
  - Eliminate heavy vertical gridlines.
  - Use subtle horizontal rules (0.5pt to 1pt) only below the table header and at the table bottom.
  - Use subtle background row zebra-striping or whitespace padding rather than dense border grids.
- Highlight the Breakthrough: Use a subtle background tint or primary accent color on the key winning row or cell (such as the ensemble Seq2Seq BLEU score of 34.81).

## 3. Chart and Plot Ergonomics

- Gridline Softness: Keep gridlines faint (neutral gray with low opacity, 10% to 20%) and subordinate to data marks.
- Axis Legibility: Explicitly label both X and Y axes with units (such as "Sentence Length (words)" and "BLEU Score"). Never leave unlabelled coordinate axes.
- Zero-Baseline Rule: Bar charts must always start at zero to avoid distorting relative magnitudes. Line charts showing incremental deltas must clearly indicate non-zero baselines on the Y-axis.
- Semantic Chart Palette: Use high-contrast, distinguishable colors for data series. Reserve the primary accent hue for the proposed model and neutral/slate tones for baseline comparisons.

## 4. Architectural and Flow Diagrams

- Fletcher and Typst Integration: Draw neural network pipelines, encoder-decoder unrollings, and attention matrices natively using vector primitives.
- Consistent Node Shapes: Use rectangular nodes with consistent corner radii for functional operations (LSTM cell, Embedding lookup) and circle/diamond nodes for mathematical operators or decision points.
- Flow Direction: Maintain a single consistent primary flow direction (left-to-right for temporal sequences, top-to-bottom for feedforward layer stacks).
- Arrow Precision: Ensure connection arrows terminate exactly at node boundaries using precise anchor points rather than imprecise visual offsets.
