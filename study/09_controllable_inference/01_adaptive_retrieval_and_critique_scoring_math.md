# Adaptive Retrieval and Critique Scoring Math

## 1. Plain-English Intuition: Turning Token Probabilities into Test-Time Dials and Scores

In Module 08, we trained the Self-RAG Generator $\mathcal{M}$ over the expanded vocabulary $\mathcal{V}_{\text{exp}}$ ($32016$ tokens). Because the $15$ special tokens are part of $\mathcal{M}$'s vocabulary, at every generation step $\mathcal{M}$ outputs a softmax probability for every regular word and every special reflection token.

Reading the probabilities of those reflection tokens at test time gives us two powerful controls without ever retraining the model:

1. A dial $\delta \in [0, 1]$ to control how often the model retrieves:
   Suppose that at the start of a response, greedy decoding ($\arg\max$) picks a regular word (or `[No Retrieval]`) because $p(\text{[No Retrieval]}) = 0.06$ and $p(\text{[Retrieval]}) = 0.04$. Even though `[Retrieval]` was not the top token, the model assigned `[Retrieval]` a relative probability of $\frac{0.04}{0.04 + 0.06} = 0.40$ ($40\%$) compared to `[No Retrieval]`.
   By setting a threshold $\delta$ (for example, $\delta = 0.20$), we can tell the system: *"Whenever the normalized probability of `[Retrieval]` exceeds $\delta = 0.20$, trigger retrieval anyway!"* Lowering $\delta$ toward $0$ makes the model retrieve more often; raising $\delta$ toward $1$ makes it retrieve less often.

2. Preference weights $(w_{\text{rel}}, w_{\text{sup}}, w_{\text{use}})$ to score and rank candidate passages:
   When retrieval is triggered and $K$ candidate passages $d_1, \dots, d_K$ are processed in parallel, greedy decoding inside each branch $k$ outputs a discrete reflection token (such as `[Relevant]` and `[Fully supported]`).
   Suppose both Branch $1$ and Branch $2$ output `[Relevant]` and `[Fully supported]`. How do we break the tie and know which branch is more strongly supported?
   Instead of looking only at the discrete token string, we read the continuous softmax probabilities of `[Relevant]`, `[Fully supported]`, `[Partially supported]`, and `[Utility:1..5]` from `pred_log_probs`, normalize them within each token group into continuous critique scores $s_t(\text{IsRel}), s_t(\text{IsSup}), s_t(\text{IsUse})$, and combine them with user-chosen weights $(w_{\text{rel}}, w_{\text{sup}}, w_{\text{use}})$.

---

## 2. Step-by-Step Derivation: Adaptive Retrieval with Threshold $\delta$ (Section 3.3 & Appendix A.3)

### 2.1 Initial Adaptive Retrieval Condition ($t = 1$)

Given prompt $x$, the model runs a forward pass at the first response token position ($j = 1$, `pred_log_probs[0]`). Let $p_0(\text{[Retrieval]})$ and $p_0(\text{[No Retrieval]})$ denote the probabilities assigned by $\mathcal{M}$ to `[Retrieval]` (token ID $32001$) and `[No Retrieval]` (token ID $32000$) at position $0$.

As defined in Appendix A.3, soft adaptive retrieval triggers the retriever $\mathcal{R}$ whenever the probability of `[Retrieval]` normalized over $\{\text{[Retrieval]}, \text{[No Retrieval]}\}$ exceeds the threshold $\delta \in [0, 1]$:

$$\frac{p_0(\text{[Retrieval]})}{p_0(\text{[Retrieval]}) + p_0(\text{[No Retrieval]})} > \delta$$

where:
- $p_0(\text{[Retrieval]}) = \exp(\text{logprob}_0[32001]) \in (0, 1)$ is the model probability for `[Retrieval]`.
- $p_0(\text{[No Retrieval]}) = \exp(\text{logprob}_0[32000]) \in (0, 1)$ is the model probability for `[No Retrieval]`.
- $\delta \in [0, 1]$ is the user-specified retrieval threshold (`--threshold`, set to $\delta = 0.2$ by default for most tasks and $\delta = 0.0$ for ALCE-ASQA).

Two caveats about this formula. First, Section 3.3 of the paper describes the normalization as being "over all output tokens in Retrieve" (which would include `[Continue to Use Evidence]`), while Appendix A.3 normalizes over Yes and No only, as above; the appendix also prints the denominator with a doubled `p(p(...))`. Second, the released inference scripts implement this decision differently from the formula: Part 2 shows the short-form version, and Module 10 Part 2 shows the long-form version.

### 2.2 Mid-Generation Adaptive Retrieval Condition ($t > 1$)

During multi-sentence long-form generation (`run_long_form_static.py` lines 124-125), after generating a sentence $y_t$, the model may output `[No Retrieval]` at token index `idx` before continuing to sentence $y_{t+1}$. At that mid-generation boundary, all three `Retrieve` tokens (`[Retrieval]`, `[Continue to Use Evidence]`, and `[No Retrieval]`) are active candidates, and `run_long_form_static.py` triggers retrieval (remapping `[No Retrieval]` to `[Retrieval]`) if:

$$\frac{p_{\text{idx}}(\text{[Retrieval]}) + p_{\text{idx}}(\text{[Continue to Use Evidence]})}{p_{\text{idx}}(\text{[Retrieval]}) + p_{\text{idx}}(\text{[No Retrieval]})} > \delta$$

This mid-generation rule exists only in the code; the paper defines only the Yes/No ratio from Section 2.1. Note that the numerator includes `[Continue to Use Evidence]` while the denominator does not, so this ratio is not a normalized probability and can exceed $1$.

---

## 3. Step-by-Step Derivation: The Three Normalized Critique Scores (Appendix A.3)

When retrieval is triggered at segment step $t$, the generator $\mathcal{M}$ processes $K$ prompts in parallel (`Prompt + [Retrieval]<paragraph>d_k</paragraph>` for $k \in \{1, \dots, K\}$). For each passage $d_k$, $\mathcal{M}$ generates a sequence of token IDs `pred_token_ids` and a list of per-position log-probability dictionaries `pred_log_probs`.

From `pred_token_ids` and `pred_log_probs`, Self-RAG computes three normalized critique scores (Appendix A.3, implemented in `run_short_form.py` lines 106-146 and `run_long_form_static.py` lines 41-84; the long-form copy contains a bug that zeroes the utility score, covered in Module 10 Part 2). Note that the main text defines each group score as the normalized probability of the single most desirable token, $s_t^G = p_t(\hat{r}) / \sum_{i=1}^{N^G} p_t(r_i)$; Appendix A.3 refines this for `IsSup` (partial credit) and `IsUse` (expected weight), and the code follows the appendix:

### 3.1 Passage Relevance Score $s_t(\text{IsRel}) \in [0, 1]$

Because the input prompt ends with `</paragraph>`, the very first token generated by $\mathcal{M}$ (at index $0$ of `pred_log_probs`) is the `IsRel` critique token (`[Relevant]` vs `[Irrelevant]`).

Let $p_0(\text{[Relevant]}) = \exp(\text{pred\_log\_probs}[0][32004])$ and $p_0(\text{[Irrelevant]}) = \exp(\text{pred\_log\_probs}[0][32003])$. The relevance score $s_t(\text{IsRel})$ is the probability of `[Relevant]` normalized over the two `IsRel` tokens:

$$s_t(\text{IsRel}) = \frac{p_0(\text{[Relevant]})}{p_0(\text{[Relevant]}) + p_0(\text{[Irrelevant]})} \in [0, 1]$$

- When $p_0(\text{[Relevant]}) \gg p_0(\text{[Irrelevant]})$, $s_t(\text{IsRel}) \to 1.0$.
- When $p_0(\text{[Irrelevant]}) \gg p_0(\text{[Relevant]})$, $s_t(\text{IsRel}) \to 0.0$.

### 3.2 Factual Support / Groundedness Score $s_t(\text{IsSup}) \in [0, 1]$

After emitting `[Relevant]` at index $0$, $\mathcal{M}$ generates the natural-language tokens of the candidate sentence $y_t^{(k)}$, and immediately after the sentence emits an `IsSup` token (`[Fully supported]`, `[Partially supported]`, or `[No support / Contradictory]`).

Because the sentence $y_t^{(k)}$ can have any length, the inference script scans `pred_token_ids` to find the first token index $j_{\text{sup}}$ where any of the three `IsSup` token IDs $\{32012, 32013, 32014\}$ appears.

At position $j_{\text{sup}}$, let $S_{\text{sup}}$ denote the sum of the probabilities of all three `IsSup` tokens:

$$S_{\text{sup}} = p_{j_{\text{sup}}}(\text{[Fully supported]}) + p_{j_{\text{sup}}}(\text{[Partially supported]}) + p_{j_{\text{sup}}}(\text{[No support / Contradictory]})$$

Instead of only rewarding `[Fully supported]`, Appendix A.3 assigns weight $1.0$ to `[Fully supported]`, weight $0.5$ to `[Partially supported]`, and weight $0.0$ to `[No support / Contradictory]`:

$$s_t(\text{IsSup}) = \frac{1.0 \cdot p_{j_{\text{sup}}}(\text{[Fully supported]}) + 0.5 \cdot p_{j_{\text{sup}}}(\text{[Partially supported]})}{S_{\text{sup}}} \in [0, 1]$$

(If no `IsSup` token appears in `pred_token_ids`, for example after `[Irrelevant]`, $s_t(\text{IsSup})$ is set to $0.0$.)

### 3.3 Perceived Utility Score $s_t(\text{IsUse}) \in [-1, 1]$

Next, the script scans `pred_token_ids` to find the first token index $j_{\text{use}}$ where any of the five `IsUse` token IDs $\{32007, 32008, 32009, 32010, 32011\}$ (`[Utility:1]` through `[Utility:5]`) appears.

At position $j_{\text{use}}$, let $S_{\text{use}}$ denote the sum of the probabilities of all five `IsUse` tokens:

$$S_{\text{use}} = \sum_{i=1}^{5} p_{j_{\text{use}}}\left(\text{[Utility:}i\text{]}\right)$$

Appendix A.3 assigns evenly spaced weights from $-1.0$ to $+1.0$ across the 5-point utility scale:

$$w_{\text{util}} = (-1.0,\; -0.5,\; 0.0,\; 0.5,\; 1.0) \quad \text{for } i \in \{1, 2, 3, 4, 5\}$$

The utility score $s_t(\text{IsUse})$ is the expected utility weight under the normalized `IsUse` distribution:

$$s_t(\text{IsUse}) = \sum_{i=1}^{5} w_{\text{util}, i} \, \frac{p_{j_{\text{use}}}\left(\text{[Utility:}i\text{]}\right)}{S_{\text{use}}} \in [-1, 1]$$

Look at why $w_{\text{util}}$ ranges from $-1.0$ to $+1.0$:
- High utility (`[Utility:5]` or `[Utility:4]`) produces a positive score up to $+1.0$.
- Mediocre utility (`[Utility:3]`) produces $0.0$.
- Low utility (`[Utility:2]` or `[Utility:1]`) produces a negative score down to $-1.0$, actively penalizing unhelpful responses.

---

## 4. Combined Segment Score (Equations 3 and 4)

Given user-selected preference weights $w_{\text{rel}} \ge 0$, $w_{\text{sup}} \ge 0$, and $w_{\text{use}} \ge 0$ (which default to $w_{\text{rel}} = 1.0, w_{\text{sup}} = 1.0, w_{\text{use}} = 0.5$ in Section 4.3), the overall critique score $\mathcal{S}(\text{Critique})$ (Equation 4) is the weighted linear sum:

$$\mathcal{S}(\text{Critique}) = w_{\text{rel}} \, s_t(\text{IsRel}) + w_{\text{sup}} \, s_t(\text{IsSup}) + w_{\text{use}} \, s_t(\text{IsUse})$$

Equation 3 of the paper adds the segment's generation probability to the critique score, $f(y_t, d, \text{Critique}) = p(y_t \mid x, d, y_{<t}) + \mathcal{S}(\text{Critique})$ (the arXiv source prints it with a stray closing parenthesis). The code includes this term only when `--use_seqscore` is enabled, and it replaces the raw sequence probability with the length-normalized $\exp(\text{seq\_score})$ (the geometric mean token probability from Module 01 Part 4). The README's short-form commands pass `--use_seqscore`; in long-form inference the flag never reaches the scoring function (Module 10 Part 2). With the flag, the code's final segment score (`final_score`) is:

$$f(y_t, d_k, \text{Critique}) = \exp\left(\frac{1}{\max(\vert y_t \vert, 1)} \sum_{j=1}^{\vert y_t \vert} \log p_{\mathcal{M}}(y_{t,j} \mid x, d_k, y_{<t}, y_{t,<j})\right) + w_{\text{rel}} \, s_t(\text{IsRel}) + w_{\text{sup}} \, s_t(\text{IsSup}) + w_{\text{use}} \, s_t(\text{IsUse})$$

where:
- $\exp(\text{seq\_score}) \in (0, 1]$ is the geometric mean per-token probability of the generated segment.
- $w_{\text{rel}} s_t(\text{IsRel}) \in [0, w_{\text{rel}}]$ rewards passage relevance.
- $w_{\text{sup}} s_t(\text{IsSup}) \in [0, w_{\text{sup}}]$ rewards factual entailment by passage $d_k$.
- $w_{\text{use}} s_t(\text{IsUse}) \in [-w_{\text{use}}, w_{\text{use}}]$ rewards overall response utility.

---

## 5. Complete Worked Numerical Example: Scoring 3 Candidate Passages by Hand

Let us work through a complete numerical example for $K = 3$ candidate passages using the default weights $w_{\text{rel}} = 1.0$, $w_{\text{sup}} = 1.0$, $w_{\text{use}} = 0.5$, and `--use_seqscore` enabled.

Suppose for our three candidate passages ($d_1, d_2, d_3$), the model generates candidate segments with the following token probabilities (computed via $\exp(\text{logprob})$):

| Quantity | Branch $1$ (Passage $d_1$: Relevant & Fully Supported) | Branch $2$ (Passage $d_2$: Irrelevant Distractor) | Branch $3$ (Passage $d_3$: Relevant & Partially Supported) |
|---|---|---|---|
| `seq_score` (avg logprob per token) | $-0.2231$ ($\exp = 0.800$) | $-0.3567$ ($\exp = 0.700$) | $-0.2231$ ($\exp = 0.800$) |
| $p_0(\text{[Relevant]})$, $p_0(\text{[Irrelevant]})$ | $0.90,\; 0.10$ | $0.05,\; 0.95$ | $0.80,\; 0.20$ |
| $p_{j_{\text{sup}}}(\text{[Fully]}),\; p_{j_{\text{sup}}}(\text{[Part.]}),\; p_{j_{\text{sup}}}(\text{[No]})$ | $0.85,\; 0.10,\; 0.05$ | $0.05,\; 0.15,\; 0.80$ | $0.20,\; 0.70,\; 0.10$ |
| $p_{j_{\text{use}}}(\text{[U:1]} \dots \text{[U:5]})$ | $0.0, 0.0, 0.0, 0.20, 0.80$ | $0.50, 0.30, 0.20, 0.0, 0.0$ | $0.0, 0.0, 0.10, 0.40, 0.50$ |

Let us compute every score step by step:

1. Relevance score $s_t(\text{IsRel})$:
   - Branch $1$: $\frac{0.90}{0.90 + 0.10} = 0.900$
   - Branch $2$: $\frac{0.05}{0.05 + 0.95} = 0.050$
   - Branch $3$: $\frac{0.80}{0.80 + 0.20} = 0.800$

2. Support score $s_t(\text{IsSup})$:
   - Branch $1$: $\frac{1.0(0.85) + 0.5(0.10)}{0.85 + 0.10 + 0.05} = \frac{0.85 + 0.05}{1.00} = 0.900$
   - Branch $2$: $\frac{1.0(0.05) + 0.5(0.15)}{0.05 + 0.15 + 0.80} = \frac{0.05 + 0.075}{1.00} = 0.125$
   - Branch $3$: $\frac{1.0(0.20) + 0.5(0.70)}{0.20 + 0.70 + 0.10} = \frac{0.20 + 0.35}{1.00} = 0.550$

3. Utility score $s_t(\text{IsUse})$ (with weights $[-1.0, -0.5, 0.0, +0.5, +1.0]$):
   - Branch $1$: $0.5(0.20) + 1.0(0.80) = 0.10 + 0.80 = +0.900$
   - Branch $2$: $-1.0(0.50) + (-0.5)(0.30) + 0.0(0.20) = -0.50 - 0.15 = -0.650$
   - Branch $3$: $0.0(0.10) + 0.5(0.40) + 1.0(0.50) = 0.20 + 0.50 = +0.700$

4. Total `final_score` $= \exp(\text{seq\_score}) + 1.0 \cdot s_t(\text{IsRel}) + 1.0 \cdot s_t(\text{IsSup}) + 0.5 \cdot s_t(\text{IsUse})$:
   - Branch $1$: $0.800 + 1.0(0.900) + 1.0(0.900) + 0.5(0.900) = 0.800 + 0.900 + 0.900 + 0.450 = 3.050$ ($1$st place)
   - Branch $3$: $0.800 + 1.0(0.800) + 1.0(0.550) + 0.5(0.700) = 0.800 + 0.800 + 0.550 + 0.350 = 2.500$ ($2$nd place)
   - Branch $2$: $0.700 + 1.0(0.050) + 1.0(0.125) + 0.5(-0.650) = 0.700 + 0.050 + 0.125 - 0.325 = 0.550$ ($3$rd place)

Even though Branch $1$ and Branch $3$ had the exact same token fluency ($\exp(\text{seq\_score}) = 0.800$), Branch $1$'s higher support score ($0.900$ vs $0.550$) and relevance score ($0.900$ vs $0.800$) give it a clear lead ($3.050$ vs $2.500$), while the irrelevant Branch $2$ is pushed down to $0.550$.
