# Academic Presentation Design Guide

Standards for structuring and authoring academic research presentations using Touying.

## 1. Action Titles

Every content slide title must be a complete sentence stating the core takeaway rather than a generic topic label.

Topic labels to avoid:
- `== Methodology`
- `== Results`
- `== Architecture`
- `== Conclusion`

Action titles to use:
- `== Multi-layer LSTMs map variable-length sequences to fixed-dimensional vectors`
- `== Reversing the source sentence significantly improves long-sequence translation`
- `== Four-layer deep architectures outperform shallow LSTM baselines`
- `== Beam search decoding consistently improves BLEU score over greedy search`

### The Ghost Deck Test

Read the slide titles sequentially from first to last. The titles alone must form a complete, self-contained, and logically coherent argument.

## 2. Narrative Structure

Structure the overall slide deck using a recognized narrative spine:

- Situation-Complication-Resolution (SCR):
  - Situation: Established background and conventional approaches
  - Complication: Fundamental limitations, bottlenecks, or open challenges
  - Resolution: Proposed model, theoretical formulation, empirical validation, and takeaways
- Funnel: Broad domain context -> Specific bottleneck -> Model architecture -> Experimental results -> Implications
- Answer First: Core result / contribution -> Supporting mathematical evidence -> Ablations and analysis

## 3. Exhibit Discipline

- One primary exhibit per results slide: Each results slide should feature exactly one central diagram, table, or plot supporting the action title.
- Direct interpretation: Every exhibit must be accompanied by explicit takeaway callouts highlighting the crucial data point or trend.
- Native resolution: Avoid low-resolution paper screenshots. Construct diagrams natively using Fletcher and format tables natively in Typst.

## 4. Text and Layout Density

- Word count budget: Aim for ~40 words per slide.
- Bullet points: Keep bullets concise and telegraphic (3-5 bullets maximum per slide).
- Typography scale: Ensure body text remains >= 20pt and labels remain >= 16pt for readability.
- White space: Treat white space as analytical clarity rather than empty area to fill.

## 5. Deck Architecture and Closing Slide

Standard academic sequence:
- Title slide: Title, authors, affiliations, date
- Motivation & Problem Statement: 1-2 slides
- Core Method / Architecture: 2-4 slides
- Experimental Setup & Datasets: 1 slide
- Main Results & Findings: 2-4 slides (one finding per slide)
- Analysis & Ablations: 1-2 slides
- Key Contributions & Implications: Final content slide (must remain on screen during Q&A)
- Optional Appendix: Backup slides for detailed questions

Critical rule: Do not use an empty "Thank You" slide. The final slide must summarize key contributions so it provides context throughout the Q&A period.

## 6. Speaker Notes

Every content slide must include a `#speaker-note[...]` block:
- State the action title assertion
- Explain the key exhibit or data point
- Provide the transition sentence leading to the subsequent slide

```typst
#speaker-note[
  Here we observe that reversing the input sequence reduces minimal time lag between source and target words.
  This simple transformation yields an improvement of over 4 BLEU points on long sentences.
]
```

## 7. Mathematical Formulations and Algorithms

- Define every symbol and variable immediately after equations.
- Use `$ ... $` blocks with clear alignment for multi-step derivations.
- Use pseudocode (`lovelace`) or theorem environments (`theorion`) when presenting formal algorithms or theoretical properties.
