# Paper Audit: Issues in the Self-RAG Paper

This file lists the issues found in "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (arXiv 2310.11511v1, the only arXiv version; the ICLR 2024 camera-ready was not checked). Line numbers refer to the markdown copy `reference/paper/2310.11511.md` ("md line"). Items marked "verified" were checked against the arXiv LaTeX source (ar5iv rendering), which separates real errors in the paper from artifacts of the markdown conversion. Values read from figures are approximate. Issues in the released code are in `study/codebase_audit.md`.

The study module files keep the paper's own statements and add notes where these issues matter; nothing in the paper has been silently "corrected".

---

## 1. Artifacts of the Markdown Conversion (Not Errors in the Paper)

- Equation 3 is truncated at md line 117. The original reads $f(y_t, d, \text{Critique}) = p(y_t \mid x, d, y_{<t}) + \mathcal{S}(\text{Critique})$ (verified; the source also has a stray closing parenthesis, see Section 2).
- The `IsUse` score formula at md line 328 is garbled. The original is $s(\text{IsUse}) = \sum_{i=1}^{5} w_i \, p(\text{IsUse} = i) / S$ with $S = \sum_{t=1}^{5} p(\text{IsUse} = t)$ (verified).
- Table 1 (md lines 51-61) and Table 2 (md lines 152-176) are scrambled; Table 2's values, as reproduced in Module 11 Part 2, were checked cell by cell against the scrambled layout.
- Algorithms 1, 2, and 3 (md lines 67, 277, 314) are flattened into single paragraphs.
- The appendix table of contents (md lines 238-250) and the Table 5 instructions (md lines 394-408) are split across table cells.
- Reflection-token names appear as fragments such as "I SREL", "IREL", or "ISSUP" because the paper typesets them in boxed small caps; accented characters appear as stray marks (for example "´").
- `_page_8_Diagram_2.jpeg` and `_page_8_Figure_1.jpeg` are two copies of Figure 3.

---

## 2. Typos and Editing Errors in the Paper

- Appendix A.3 (md line 324): the `IsSup` score formula is labeled $s(\text{IsRel})$ instead of $s(\text{IsSup})$ (verified).
- Appendix A.3 (md line 334): the adaptive-retrieval denominator is printed as $p(\text{Retrieve}=\text{Yes}) + p(p(\text{Retrieve}=\text{No}))$, with a doubled $p$ (verified).
- Equation 3: the source has an extra closing parenthesis after $y_{<t}$ (verified).
- Appendix A.1 (md line 263): the Retrieve count is printed as "1,2594" (meaning $12,594$; verified).
- Appendix A.3 (md line 326): "We assigns weighted scores" (verified).
- Section 1 (md line 27): "citation previsions" for "citation precision" (verified).
- Section 1 (md line 23): "a propriety LM" for "a proprietary LM".
- Section 1 (md line 15): "never second visits the generation quality".
- Section 2 (md line 33): "SELF-RAG trains to an arbitrary LM".
- Section 3.2.1 (md line 87): "followed by few-shot demonstrations I the original task input x and output y" is garbled.
- Section 5.2 (md lines 206-208): "dramatically change on both datasets. as δ varies"; "We also evaluate Figures 4a, 4b and 4c shows the models' performance" (garbled sentence); "we do not observed".
- Section 5.2 (md line 206): "the token probability of reward tokens" refers to the retrieval tokens.
- Appendix A.1 (md line 260): "We evaluate attributions in three scale".
- Appendix A.2 (md line 271): "FLAN subsets subsets"; Table 3 (md line 291): "OpenBoookQA".
- Appendix A.2 (md line 304): the FLAN-3B comparison is said to "investigate the effectiveness of different data sizes", but it compares two base models trained on the same data.
- Appendix B.2 (md line 344): "effectivess of such appraoches"; (md line 346): Llama2-chat "tend to generate significantly lower outputs", where the example shows longer outputs (about $100$ vs $40$ tokens).
- Table 4 (md line 374): the first example ends with "[ISUSE:5]" instead of "[ISUSE=5]".
- Table 6 (md line 446): "The the Reference [1]".
- Figure 1 (image): "11 of 50 states names are come from persons".
- Figure 2 (image): the right example's "Output:" line reads "1 of 50 states names come from persons", while its augmented output reads "11 of 50 states' names come from person".
- Figure 4a (image): the y-axis label reads "Perfomance".

---

## 3. Internal Inconsistencies

Definitions and algorithms:
- Retrieval threshold: Section 3.3 (md line 113) normalizes the Retrieve=Yes probability "over all output tokens in Retrieve" (three tokens, verified), while Appendix A.3 (md line 334) normalizes over Yes and No only.
- Critique scores: below Equation 4 (md line 122), each group score $s_t^G$ is the normalized probability of the single most desirable token; Appendix A.3 instead gives `IsSup` partial credit ($0.5$ for Partially) and defines `IsUse` as an expected weight in $[-1, 1]$.
- Utility timing: Algorithm 1 line 6 (md line 67) predicts `IsUse` for every passage at every segment; Appendix A.1 (md line 256) says utility is given only at the output level, and the training examples (Table 4) place it once at the end.
- Algorithm 1 lines 9-10 use an undefined $\mathcal{M}_{gen}$ and condition on $x$ alone, omitting the preceding segments $y_{<t}$.
- Algorithm 3 line 9 (md line 314): "C predicts IsUse for each d", whereas the Appendix A.2 text and the released code predict utility once per example from $(x, y)$.
- Equation 1 (md line 92) conditions the Critic on $(x, y)$ only, although `IsRel` and `IsSup` inputs include the passage $d$ (a notational shortcut).

Numbers and labels:
- Section 3.2.1 (md line 87) says "4k-20k supervised training data for each type", but the utility set has $3,831$ examples (Appendix A.1).
- Section 3.2.1 (md line 94) says the Critic achieves "a higher than 90% agreement ... on most reflection token categories"; the table shows $93.8$ (Retrieve), $93.5$ (`IsSup`), $80.2$ (`IsRel`), and $73.5$ (`IsUse`), so two of four exceed $90\%$, and FLAN-3B beats Llama2-7B on `IsRel` ($82.0$ vs $80.2$).
- The same sentence cites "Appendix Table 5", but the appendix labels the Critic table "Figure 5" (verified), and "Table 5" is a different table (the evaluation instructions).
- Section 5.2 (md line 196) calls the ablation table "Table 3a"; it is Figure 3a.
- Appendix D (md line 370) describes Table 10 as "three-way output tokens for IsRel" (`IsRel` has two values) and Table 11 as an `IsRel` table, while Table 11's caption says `IsSup` (verified). Table 10's caption says "given the input only", although its prompt includes evidence and optional preceding sentences.
- Appendix C.1 (md line 354) lists the evaluated models as Self-RAG 7B, Alpaca 7B, Alpaca 13B, and Llama2-Chat-13B, then reports $20\%$ for "Alpaca 30B" and $15\%$ for "Alpaca" with no size (verified).
- Table 3 is captioned "The generator LM M training data statistics" ($145,619$ instances), but the paper does not say whether these counts are before or after the generator data was filtered (the released code drops and downsamples examples).
- Section 5.1 (md line 182) says CoVe "iteratively prompts Llama2 65B" (verified; Section 4.2 writes only "CoVE 65B"). Llama 2 has no 65B model; CoVe used the original LLaMA 65B.

Analysis figures versus Table 2:
- ASQA citation precision for the default 7B model is about $70.5$ in Figure 3b ($w_{\text{sup}} = 1$) and about $70.1$ for the $150\text{k}$ point of Figure 4c, versus $66.9$ in Table 2; the default point's MAUVE in Figure 3b (about $86.5$) also differs from Table 2's $74.3$. PubHealth at $150\text{k}$ is about $72.8$ in Figure 4b versus $72.4$ in Table 2. The paper does not say which evaluation set the analysis figures use (only the ablation table states "sampled 150 instances" for ASQA).
- Figure 3c's "normalized accuracy" is not defined (the curves start at $1.0$ for $\delta = 0$), and the PubHealth accuracy axis shows "0.99" twice because its ticks ($0.985$ to $1.000$) are rounded to two decimals.
- The $50\text{k}$ ablation model scores $73.5$ on PubHealth, higher than the full $150\text{k}$ model's $72.4$.
- Figure 3b's $w_{\text{sup}}$ values are unlabeled except the ticks $1$ and $2$; the points sit at about $0.1$, $0.5$, $1$, and $2$.

---

## 4. Claims Stronger Than the Reported Numbers

- Abstract (md line 11): Self-RAG "outperforms ChatGPT ... on Open-domain QA, reasoning and fact verification tasks". ChatGPT is higher on TriviaQA ($74.3$ vs $66.4$ and $69.3$) and on ARC-Challenge ($75.3$ vs $67.3$ and $73.1$); the claim holds for PopQA and PubHealth.
- Section 1 (md line 27): Self-RAG "outperforms retrieval-augmented ChatGPT on four tasks". Against Ret-ChatGPT it is higher on PopQA, TriviaQA, PubHealth, and ASQA citation precision, and lower on ARC-Challenge and on ASQA str-em, ROUGE, MAUVE, and citation recall (Ret-ChatGPT has no Bio number). "Four tasks" holds only if ASQA citation precision counts as a task.
- Section 1 (md line 27): Self-RAG outperforms "Llama2-chat ... and Alpaca ... on all tasks". On ASQA str-em, Ret-Llama2-chat ($32.8$) and Alpaca 13B with retrieval ($34.8$) beat both Self-RAG models ($30.0$, $31.7$), and Alpaca 13B with retrieval also beats Self-RAG 7B on ROUGE ($36.7$ vs $35.7$).
- Section 5.1 (md line 184): "the best performance among non-proprietary LM-based models on all tasks". Not on ASQA str-em (Alpaca 13B with retrieval, $34.8$), and not for the 7B model on ASQA ROUGE (Llama2-FT $35.8$, Alpaca 13B with retrieval $36.7$, vs $35.7$).
- Section 5.1 (md line 182): Self-RAG outperforms ChatGPT on "ASQA (Rouge and MAUVE)". For ROUGE this holds only for 13B ($37.0$ vs $36.2$); 7B scores $35.7$.
- Section 5.1 (md line 190): Llama2-FT "lags behind" Self-RAG. It is marginally higher on ASQA str-em ($31.0$ vs $30.0$) and ROUGE ($35.8$ vs $35.7$).
- Section 5.2 (md line 196): "all components play important roles". On PubHealth, the four inference ablations change accuracy by at most $0.9$ points (No retrieval: $-0.5$); the large effects are on PopQA and ASQA.
- Section 5.2 (md line 198): raising the `IsSup` weight "leads to positive effects on the models' citation precision". The gain is about $0.8$ points, while MAUVE falls by about $10$ points.
- Section 5.2 (md line 208): Llama2-FT shows no comparable gain from $50\text{k}$ to $150\text{k}$ examples; no numbers are shown.
- Section 3.2.1 (md line 87): "Manual assessment reveals that GPT-4 reflection token predictions show high agreement with human evaluations". The assessment (Appendix A.1) is the authors' own check of $20$ instances per aspect.
- Throughout: "significantly outperforms" is used without significance tests, variance, or multiple runs.

---

## 5. Errors in Showcased Examples

- Table 7, Rory Tapner (md lines 462-466): the second segment says he was "born on 30 September 1959 in London, England" and is labeled Fully Supported, but Reference [2] gives no birthplace. The paper discusses only the first segment's correct Contradictory label.
- Table 7, G. Venugopal (md lines 470-474): the output says "over 300 songs in Malayalam, Tamil, and Hindi movies", while Reference [1] says "more than 300 films" and mentions only Malayalam. Appendix C.3 (md line 366) attributes the Partially Supported label to "the statement listing the name of the songs", which the output does not contain.
- Table 6, "Who is the author of The Lie?" (md lines 422-430): the model labels a passage about a different book (*Lying*) as Relevant and Fully Supported; the paper acknowledges both labels as incorrect, which makes it a useful failure example.
- Table 4, poem example (md line 382): two sentences appear twice in a row; whether the training data contains these duplicates is not discussed.
- Table 4, Wizard of Wikipedia example (md line 390): the output starts mid-word ("hey also are easily trained...") and has no retrieval token before its first sentence, so the example appears truncated.

---

## 6. Missing Details and Limitations (Useful for a Critique Slide)

- No variance, seeds, or significance tests are reported; every number is a single run.
- Human evaluation: the paper "sampled 50 samples from PopQA and Bio results" without saying whether that is $50$ per task, who the annotators were, how many there were, or how much they agreed; `S&P` excludes instances where the model predicted Irrelevant or No support.
- GPT-4 label quality rests on the authors checking $20$ instances per aspect.
- The `IsRel` Critic has the lowest agreement among the passage-level tokens ($80.2\%$), yet its labels decide which passages enter the generator's training data.
- Inference cost is not measured: each retrieval step runs $K$ passage-conditioned generations and long-form decoding adds a segment-level beam search, but no latency or compute comparison with single-pass RAG is reported.
- The analysis figures (3b, 3c, 4) do not state their evaluation sets, and "normalized accuracy" is undefined.
- Retrieval setups differ by task (2018 Wikipedia; a December 2020 dump for PopQA; five extra Google Programmable Search documents for PopQA, TriviaQA, and biographies; GTR-XXL passages for ASQA). Results marked with an asterisk (Toolformer, SAIL, CoVe) are copied from other papers, and Perplexity.ai uses its own retrieval.
- Instructions were adjusted per model to avoid format violations (Appendix B.2), so prompts are not identical across baselines; the ChatGPT version is not stated.
- PopQA and TriviaQA use answer inclusion (a lenient metric) rather than exact match.
- Data-construction heuristics that shape the training data (rebalancing of `[Relevant]` labels, keeping $40\%$ of no-retrieval examples, dataset-specific passage overrides and relabeling) appear only in the released code (`study/codebase_audit.md`, items C6 and C15-C19).
- Self-RAG 7B beats 13B on FactScore and MAUVE; the paper explains only the precision side.
- Segments are always sentences (spaCy); the claim that the framework applies to other segment units is not tested.

---

## 7. Paper vs Released Code (Summary)

Details and line numbers are in `study/codebase_audit.md`.
- 13B maximum sequence length: $1,524$ in the paper, $1536$ in the training script.
- Training setup: the paper states batch $128$, $3\%$ warmup, linear decay, and $4$ A100 80GB GPUs for all models; the README's Critic command uses an effective batch of $16$, $1\%$ warmup, and a cosine schedule, and the README describes $8$ A100 40GB GPUs for the 7B generator.
- FactScore generation budget: $200$ tokens per segment in the paper, $300$ in the README command.
- Short-form utility weight: $0.5$ in the paper, default $1.0$ in `run_short_form.py`.
- Long-form scoring: the released script zeroes the utility score, ignores the sequence score, hardcodes the weights, reuses one fixed passage set at every depth, and leaks sibling text into deeper expansions.
- Adaptive retrieval: the short-form script computes the threshold ratio on raw log-probabilities (inverting the decision), and the long-form script requires the greedy output to contain `[Retrieval]` before applying the threshold.
- Generator training: the released masking code silently removes the loss on intermediate sentences of multi-passage examples.
- Evaluation instructions: the released `TASK_INST` omits the formatting sentences that Table 5 shows for PubHealth and ARC-Challenge.
