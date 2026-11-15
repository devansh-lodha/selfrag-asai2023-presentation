# Study Memory & Scratchpad

## Status and Conventions
- All $34$ part files in `01_*` through `12_*` are written and were reviewed line by line against `reference/paper/` and `reference/code/` (numeric examples recomputed, Table 2 checked cell by cell, figure values measured from the images).
- Reference appendices: `paper_audit.md` (issues in the paper, with conversion artifacts separated from real paper errors by checking the arXiv v1 LaTeX source) and `codebase_audit.md` (issues in the released code; replaces the earlier `CODEBASE_AUDIT.md` and `codebase_audit_and_bugs.md`).
- Part files keep the paper's statements as written and flag errors next to them; never silently "correct" what the paper says.
- Code line numbers inside part files can drift from `reference/code/` by up to about $90$ lines; `codebase_audit.md` has re-verified numbers. Search by identifier when a cited line does not match.
- Figure readings (Figures 3 and 4) are approximate values measured from the images; the ASQA analysis figures do not match Table 2 (about $70$ vs $66.9$ citation precision), so they likely use a different evaluation set.

## Mathematical & Conceptual Nuances
- Segment decomposition: $y = [y_1, \dots, y_T]$ where each $y_t$ is a sentence (via spaCy `en_core_web_sm`).
- Four reflection token groups (13 reflection tokens + 2 context markup tokens = 15 additional special tokens):
  - `Retrieve`: `[Retrieval]`, `[No Retrieval]`, `[Continue to Use Evidence]`
  - `IsRel`: `[Relevant]`, `[Irrelevant]`
  - `IsSup`: `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`
  - `IsUse`: `[Utility:1]`, `[Utility:2]`, `[Utility:3]`, `[Utility:4]`, `[Utility:5]`
  - Context markups: `<paragraph>`, `</paragraph>`
- Token ids: the $15$ tokens get ids $32000$-$32014$ in the order of the `additional_special_tokens` list, and the pad token gets $32015$ in both training scripts ($32016$ tokens in total).
- Paper Equation 3 vs Codebase Segment Scoring:
  - Paper Eq 3 writes $f(y_t, d, \text{Critique}) = p(y_t \mid x, d, y_{<t}) + \mathcal{S}(\text{Critique})$ (the arXiv source has a stray closing parenthesis; the markdown copy truncates the equation).
  - Code (`run_short_form.py` and `run_long_form_static.py`) computes `seq_score = pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1)` and adds `np.exp(seq_score)` (the geometric mean token probability) only when `--use_seqscore` is enabled; the long-form script never receives the flag.
  - Critique scores in code (following Appendix A.3, not the simpler definition below Eq 4):
    - $s_t(\text{IsRel}) = \frac{p(\text{[Relevant]})}{p(\text{[Relevant]}) + p(\text{[Irrelevant]})}$ at `pred_log_probs[0]`.
    - $s_t(\text{IsSup}) = \frac{p(\text{[Fully supported]}) + 0.5 \, p(\text{[Partially supported]})}{\sum_{r \in \text{IsSup}} p(r)}$ at the first token index where any `grd_tokens` ID appears in `pred_token_ids`.
    - $s_t(\text{IsUse}) = \sum_{i=1}^5 w_i \frac{p(\text{[Utility:}i\text{]})}{\sum_{j=1}^5 p(\text{[Utility:}j\text{]})}$ with $w = [-1, -0.5, 0, 0.5, 1]$ at the first token index where any `ut_tokens` ID appears.
- Adaptive retrieval: Appendix A.3 normalizes over Yes and No, Section 3.3 says "over all output tokens in Retrieve". The short-form code divides raw log-probabilities (inverted decision); the long-form code retrieves only if the greedy text contains `[Retrieval]` and the probability test passes.

## Most Presentation-Relevant Findings
- Long-form inference as released ranks segments by exactly $s(\text{IsRel}) + s(\text{IsSup})$ (utility zeroed, sequence score unused, weights hardcoded), reuses one fixed passage set at every depth, and leaks sibling text into deeper expansions.
- Generator training code masks everything from the first `<paragraph>` to the last `</paragraph>` (no `break`), silently removing the loss on intermediate sentences of multi-passage examples.
- Headline claims to quote carefully: Self-RAG does not beat ChatGPT on TriviaQA or ARC-Challenge; it is not best on ASQA str-em among open models; Llama2-FT is marginally higher on ASQA str-em and ROUGE.
- The Table 7 Rory Tapner example contains an unflagged self-critique error ("born in London" labeled Fully Supported).
