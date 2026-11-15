# Third-Party Presentation Packages for Typst

Essential packages for technical, mathematical, and algorithmic presentations in Typst.

## 1. Pseudocode & Algorithms (`lovelace`)

Package: `@preview/lovelace:0.3.1`

Typesets algorithms and pseudocode cleanly for slides:

```typst
#import "@preview/lovelace:0.3.1": *

#pseudocode-list(booktabs: true, numbered-title: smallcaps[Greedy Decoding])[
  + *Input*: Context vector $v$, maximum length $T'$
  + Initialize state $s_0 = v$, token $y_0 = "<BOS>"$
  + *for* $t = 1$ *to* $T'$ *do*
    + Compute output distribution $P(y_t | s_(t-1), y_(t-1))$
    + Select $y_t = argmax_w P(w | s_(t-1), y_(t-1))$
    + *if* $y_t = "<EOS>"$ *then*
      + *break*
    + *end*
    + Update state $s_t = "LSTM"(s_(t-1), y_t)$
  + *end*
  + *return* Sequence $(y_1, ..., y_t)$
]
```

## 2. Theorems, Definitions & Lemmas (`theorion`)

Package: `@preview/theorion:0.6.0`

Provides slide-friendly boxes for formal definitions, theorems, and lemmas:

```typst
#import "@preview/theorion:0.6.0": *
#import cosmos.clouds: *
#show: show-theorion

#definition[Conditional Sequence Likelihood][
  Given input sequence $(x_1, ..., x_T)$ and target sequence $(y_1, ..., y_(T'))$, the goal is to maximize:
  $ 1/|S| sum_((X, Y) in S) log P(Y | X) = 1/|S| sum_((X, Y) in S) sum_(t=1)^(T') log P(y_t | v, y_1, ..., y_(t-1)) $
] <def:seq-prob>

#theorem[Reversal Advantage][
  Reversing source words introduces many short-term dependencies, vastly simplifying LSTM optimization.
]
```

Available styles:
- `cosmos.simple`: Minimalist LaTeX-like layout
- `cosmos.clouds`: Rounded card background
- `cosmos.rainbow`: Left-accent colored stripe

## 3. Section and Slide Numbering (`numbly`)

Package: `@preview/numbly:0.1.0`

Formats hierarchical section numbering automatically:

```typst
#import "@preview/numbly:0.1.0": numbly

#set heading(numbering: numbly(
  "Part {1:I}.",   // Level 1: Part I., Part II., ...
  "{1:I}.{2}.",    // Level 2: I.1., I.2., ...
))
```

## 4. Charts and Plots (`lilaq` or native `plot`)

Simple statistical bar and line plots without external raster images:

```typst
// Native table-based comparison chart
#table(
  columns: (2fr, 1fr, 1fr, 1fr),
  stroke: none,
  table.hline(stroke: 1.5pt),
  [*Model Architecture*], [*BLEU (Clean)*], [*BLEU (Reversed)*], [*Gain*],
  table.hline(stroke: 0.8pt),
  [1-Layer Shallow LSTM], [26.1], [29.8], [+3.7],
  [4-Layer Deep LSTM (1000 cells)], [28.4], [34.8], [+6.4],
  [Ensemble of 5 Deep LSTMs], [30.2], [36.5], [+6.3],
  table.hline(stroke: 1.5pt),
)
```
