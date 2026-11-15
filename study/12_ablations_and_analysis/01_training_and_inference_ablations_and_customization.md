# Training and Inference Ablations and Customization

## 1. Plain-English Intuition: Testing Every Moving Part of Self-RAG

In Module 11, we saw that the full Self-RAG model outperforms conventional RAG and instruction-tuned baselines across all six benchmarks.

However, Self-RAG combines several distinct mechanisms across training and inference:
- During training: It augments the training corpus with retrieved passages (via Retriever $\mathcal{R}$) and reflection tokens (via Critic $\mathcal{C}$).
- During inference: It triggers retrieval using a soft probability threshold $\delta$, processes $K$ retrieved passages in parallel instead of just the top-$1$ passage, and ranks candidate segments using the support score $s_t(\text{IsSup})$ and adjustable weights $(w_{\text{rel}}, w_{\text{sup}}, w_{\text{use}})$.

To verify that each of these components is genuinely necessary and to show how the model can be customized at test time, Section 5.2 of the paper conducts four controlled analyses (shown in Figure 3 and Figure 4):
1. Training and inference ablations (Figure 3a): Removing one component at a time on a $50\text{k}$-instance training subset.
2. Test-time preference customization (Figure 3b): Varying the `IsSup` weight $w_{\text{sup}}$ to trade off citation precision against fluency (`MAUVE`) without retraining.
3. Retrieval frequency vs accuracy trade-off (Figure 3c): Varying the retrieval threshold $\delta$ on PubHealth and PopQA.
4. Training data scaling (Figures 4a-4c): Training Self-RAG 7B on $5\text{k}, 10\text{k}, 20\text{k}, 50\text{k},$ and $150\text{k}$ examples.

---

## 2. Step-by-Step Walkthrough of the Ablation Study (Figure 3a / Table 3a)

To run multiple training ablations efficiently, the authors randomly sampled $50,000$ instances from the $150\text{k}$ Generator training set and evaluated `Self-RAG 7B (50k)` against two training variants and four test-time variants on PopQA (`PQA (acc)`), PubHealth (`Med (acc)`), and a $150$-example subset of ASQA (`AS (em)`).

The table below reproduces Figure 3a from Section 5.2 of the paper:

| Ablation Group | Model / Inference Variant | PopQA `PQA (acc)` | PubHealth `Med (acc)` | ASQA `AS (em)` | Change vs `SELF-RAG (50k)` |
|---|---|---|---|---|---|
| Full Reference ($50\text{k}$) | `SELF-RAG (50k)` | $45.5$ | $73.5$ | $32.1$ | Baseline reference |
| Training Ablation | `No Retriever R` | $43.6$ | $67.8$ | $31.0$ | $-1.9$ PQA, $-5.7$ Med, $-1.1$ AS |
| Training Ablation | `No Critic C` | $42.6$ | $72.0$ | $18.1$ | $-2.9$ PQA, $-1.5$ Med, $-14.0$ AS |
| Inference Ablation | `No retrieval` | $24.7$ | $73.0$ | – | $-20.8$ PQA, $-0.5$ Med |
| Inference Ablation | `Hard constraints` | $28.3$ | $72.6$ | – | $-17.2$ PQA, $-0.9$ Med |
| Inference Ablation | `Retrieve top1` | $41.8$ | $73.1$ | $28.6$ | $-3.7$ PQA, $-0.4$ Med, $-3.5$ AS |
| Inference Ablation | `Remove IsSup` | $44.1$ | $73.2$ | $30.6$ | $-1.4$ PQA, $-0.3$ Med, $-1.5$ AS |

Let us analyze what each of these six ablations proves:

### 2.1 Training Ablations (`No Retriever R` and `No Critic C`)

1. `No Retriever R` ($43.6$ PQA, $67.8$ Med, $31.0$ AS):
   This variant trains Llama-2 7B on the plain $50\text{k}$ instruction-output pairs $(x, y)$ without any retrieved passages or reflection tokens during training, and then augments the model with retrieved passages at test time.
   Without exposure to interleaved passages during training, PubHealth accuracy drops by $-5.7\%$ ($73.5\% \to 67.8\%$), showing that standard instruction tuning does not teach a model how to filter and reason over noisy medical evidence at test time.

2. `No Critic C` ($42.6$ PQA, $72.0$ Med, $18.1$ AS):
   This variant trains Llama-2 7B on the same $50\text{k}$ instruction-output pairs, but always prepends the top-1 retrieved passage to every training example without any reflection tokens from the Critic $\mathcal{C}$ (replicating the SAIL training paradigm on the $50\text{k}$ Self-RAG dataset).
   Look at what happens on ASQA: `AS (em)` collapses from $32.1\%$ down to $18.1\%$ ($-14.0\%$), and PopQA drops to $42.6\%$ (scoring even lower than `No Retriever R` at $43.6\%$).
   Why does training with the top-1 passage on every example without Critic reflection tokens hurt performance so severely?
   The paper does not analyze this beyond noting that both trained components matter. A plausible explanation is that in a diverse instruction-tuning corpus, the top-1 passage returned by Contriever is often irrelevant or only partially relevant, so training the model to always condition on an unverified top-1 passage, without `[No Retrieval]`, `[Irrelevant]`, or `[Partially supported]` tokens, teaches it either to over-rely on single passages or to ignore context.

### 2.2 Inference Ablations (`No retrieval`, `Hard constraints`, `Retrieve top1`, `Remove IsSup`)

All four inference ablations use the exact same trained `SELF-RAG (50k)` checkpoint and modify only the test-time decoding algorithm:

1. `No retrieval` ($24.7$ PQA, $73.0$ Med):
   Disables retrieval completely at test time (`--mode no_retrieval`). PopQA accuracy plunges by $-20.8\%$ ($45.5\% \to 24.7\%$) because rare long-tail entities cannot be recalled from parametric memory alone, whereas PubHealth drops by only $-0.5\%$ ($73.5\% \to 73.0\%$).

2. `Hard constraints` ($28.3$ PQA, $72.6$ Med):
   Instead of triggering retrieval when the normalized probability of `[Retrieval]` exceeds the soft threshold $\delta = 0.2$, this variant only retrieves when `[Retrieval]` is the greedy $\arg\max$ token (`"[Retrieval]" in pred`).
   On PopQA, accuracy drops by $-17.2\%$ ($45.5\% \to 28.3\%$). Why? Because at the start of a question, the model often assigns a higher raw probability to starting the answer sentence directly (or outputting `[No Retrieval]`) than to `[Retrieval]`, even when the relative probability $\frac{p(\text{[Retrieval]})}{p(\text{[Retrieval]}) + p(\text{[No Retrieval]})}$ is $0.30 > 0.20$. Using the soft threshold $\delta = 0.2$ is crucial to ensure frequent retrieval on knowledge-intensive tasks.

3. `Retrieve top1` ($41.8$ PQA, $73.1$ Med, $28.6$ AS):
   Always retrieves, but uses only the single top-$1$ passage (`ndocs = 1`) from the retriever rather than evaluating $K$ passages in parallel and ranking them with critique tokens.
   PopQA drops by $-3.7\%$ ($45.5\% \to 41.8\%$) and ASQA drops by $-3.5\%$ ($32.1\% \to 28.6\%$). Whenever the retriever's #1 passage is an off-topic keyword match (while passage #3 or #4 contains the true answer), `Retrieve top1` fails, whereas parallel $K$-passage critique evaluation recovers the relevant passage.

4. `Remove IsSup` ($44.1$ PQA, $73.2$ Med, $30.6$ AS):
   Evaluates all $K$ passages in parallel during beam search, but removes the groundedness score $s_t(\text{IsSup})$ from the segment scoring function (setting $w_{\text{sup}} = 0$ so ranking depends only on `IsRel` and `IsUse`).
   Both PopQA ($-1.4\%$) and ASQA ($-1.5\%$) drop, confirming that scoring whether the generated claim is entailed by the passage (`IsSup`) adds value beyond merely scoring whether the passage topic is relevant (`IsRel`).

---

## 3. Test-Time Preference Customization: Tuning $w_{\text{sup}}$ on ASQA (Figure 3b)

One of Self-RAG's core advantages over RLHF is that preference weights $(w_{\text{rel}}, w_{\text{sup}}, w_{\text{use}})$ in Equation 4 can be adjusted at test time without retraining.

In Figure 3b, the authors take the trained `Self-RAG 7B` model on ASQA and vary the weight $w_{\text{sup}}$ assigned to the `IsSup` support score (the other weights are presumably left at their defaults, $w_{\text{rel}} = 1.0$ and $w_{\text{use}} = 0.5$; the paper does not say). The plot labels only the ticks $1$ and $2$; the four points sit at about $w_{\text{sup}} \approx 0.1, 0.5, 1.0, 2.0$. Values below are read from the plot:

| `IsSup` Weight $w_{\text{sup}}$ | ASQA Citation Precision (`Precision`) | ASQA Fluency (`Mauve`) | Interpretation (Not Reported in the Paper) |
|---|---|---|---|
| $\approx 0.1$ (Low emphasis on support) | $\sim 69.7\%$ | $\sim 96.4$ | Selects longer, more fluent sentences that are not always strictly entailed by the cited passage. |
| $0.5$ (Moderate emphasis) | $\sim 70.0\%$ | $\sim 87.8$ | Balances sentence fluency with passage entailment. |
| $1.0$ (Default emphasis) | $\sim 70.5\%$ | $\sim 86.5$ | Strongly favors `[Fully supported]` sentences closely aligned with the retrieved text. |
| $2.0$ (High emphasis on support) | $\sim 70.5\%$ | $\sim 86.1$ | Maximizes strict factual support from the passage at the cost of shorter, more extractive phrasing. |

Three cautions when presenting this figure. The precision gain is small (about $0.8$ points) while MAUVE falls by about $10$ points. The default point ($w_{\text{sup}} = 1.0$: precision $\approx 70.5$, MAUVE $\approx 86.5$) does not match Table 2's 7B ASQA numbers ($66.9$ and $74.3$), so this analysis evidently uses a different evaluation set or configuration, which the paper does not specify. And the released long-form script hardcodes the critique weights, so it cannot reproduce this sweep without editing (Module 10 Part 2).

Why do `Citation Precision` and `MAUVE` move in opposite directions as $w_{\text{sup}}$ increases?
- Recall from Module 06 Part 1 (Table 11) that `[Fully supported]` is defined as: *"All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical."*
- Therefore, increasing $w_{\text{sup}}$ pushes the beam search to select shorter, tightly extractive sentences that mirror the passage wording (raising NLI `Citation Precision`), whereas lowering $w_{\text{sup}}$ allows the beam search to pick more expansive, free-flowing prose (raising `MAUVE` while occasionally introducing unsupported connective claims).

---

## 4. Retrieval Threshold $\delta$ Trade-Off (Figure 3c) and Training Data Scaling (Figures 4a-4c)

### 4.1 Retrieval Frequency vs Normalized Accuracy Across $\delta$ (Figure 3c)

In Figure 3c, the authors vary the adaptive retrieval threshold $\delta \in [0.0, 0.1, 0.5, 0.7]$ on `PubHealth` and `PopQA` and plot both the retrieval frequency (the fraction of test queries that trigger retrieval) and the "normalized" accuracy. The paper does not define the normalization; the curves start at $1.0$ for $\delta = 0$, which is consistent with dividing by the accuracy at $\delta = 0$. Values below are read from the plot (the PubHealth accuracy axis is very narrow: its tick labels are rounded to two decimals, so two ticks both read $0.99$):

| Threshold $\delta$ | PubHealth Retrieval Frequency | PubHealth Normalized Accuracy | PopQA Retrieval Frequency | PopQA Normalized Accuracy |
|---|---|---|---|---|
| $0.0$ | $1.00$ ($100\%$) | $1.00$ ($100\%$) | $1.00$ ($100\%$) | $1.00$ ($100\%$) |
| $0.1$ | $1.00$ ($100\%$) | $1.00$ ($100\%$) | $1.00$ ($100\%$) | $\sim 0.89$ ($89\%$) |
| $0.5$ | $\sim 0.16$ ($16\%$) | $\sim 0.985$ ($98.5\%$) | $\sim 0.60$ ($60\%$) | $\sim 0.75$ ($75\%$) |
| $0.7$ | $0.00$ ($0\%$) | $\sim 0.99$ ($99\%$) | $\sim 0.20$ ($20\%$) | $\sim 0.45$ ($45\%$) |

Comparing the two plots in Figure 3c shows how task characteristics determine sensitivity to $\delta$:
- On `PubHealth`, raising $\delta$ from $0.1$ to $0.7$ cuts retrieval calls from $100\%$ to $0\%$ (eliminating retrieval latency entirely) while normalized accuracy changes by less than $2\%$. The paper reports only that the deterioration from retrieving less is smaller on PubHealth than on PopQA; one plausible reason is that the true/false verdicts rely more on reasoning than on looking up rare facts.
- On `PopQA`, where every query asks for a rare long-tail entity ($<100$ monthly Wikipedia views), cutting retrieval frequency from $100\%$ to about $20\%$ causes normalized accuracy to drop by $55\%$ ($1.00 \to 0.45$).

### 4.2 Training Data Scaling Laws Across $5\text{k} \to 150\text{k}$ Instances (Figures 4a, 4b, 4c)

In Figures 4a, 4b, and 4c, the authors randomly sample subsets of $5\text{k}, 10\text{k}, 20\text{k},$ and $50\text{k}$ instances from $\mathcal{D}_{\text{gen}}$, fine-tune four `Self-RAG 7B` models on those subsets, and compare them with the final `Self-RAG 7B` trained on all $150\text{k}$ instances:

| Training Set Size | PopQA Accuracy (Figure 4a) | PubHealth Accuracy (Figure 4b) | ASQA Citation Precision (Figure 4c) |
|---|---|---|---|
| $5\text{k}$ instances | $\sim 33.9\%$ | $\sim 71.8\%$ | $\sim 41.2\%$ |
| $10\text{k}$ instances | $\sim 39.3\%$ | $\sim 70.9\%$ | $\sim 51.6\%$ |
| $20\text{k}$ instances | $\sim 40.8\%$ | $\sim 71.8\%$ | $\sim 57.4\%$ |
| $50\text{k}$ instances | $45.5\%$ | $73.5\%$ | $\sim 58.9\%$ |
| $150\text{k}$ instances (Full) | $\sim 54.9\%$ (Table 2: $54.9\%$) | $\sim 72.8\%$ (Table 2: $72.4\%$) | $\sim 70.1\%$ (Table 2: $66.9\%$) |

The $150\text{k}$ points in Figures 4b and 4c do not match Table 2 (the ASQA gap is about $3$ points), so, like Figure 3b, these ASQA analysis numbers likely come from a different evaluation set; the paper does not say. Notice also that the $50\text{k}$ ablation model scores higher on PubHealth ($73.5\%$) than the full $150\text{k}$ model.

Notice two key takeaways from Figures 4a-4c:
1. Both `PopQA` accuracy (about $34\% \to 55\%$) and `ASQA` citation precision (about $41\% \to 70\%$ in the plot) show steep, non-saturating upward trajectories all the way to $150\text{k}$ instances, while PubHealth stays within about $3$ points across all sizes.
2. As noted in Section 5.2, when the baseline `Llama2-FT 7B` (trained without reflection tokens or passages) is scaled from $50\text{k}$ to $150\text{k}$ instances, it does not exhibit such large gains; the paper states this without showing the numbers. The authors read this as a sign that further expanding Self-RAG's training data may bring further improvements.
