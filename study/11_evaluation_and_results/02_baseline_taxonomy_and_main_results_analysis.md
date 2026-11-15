# Baseline Taxonomy and Main Results Analysis

## 1. Plain-English Intuition: How to Read Table 2 as a Set of Controlled Experiments

In Part 1, we learned what each of the six benchmarks and ten evaluation columns in Table 2 measures. Now let us look at the $19$ model rows in Table 2 (Section 4.2 and Section 5.1 of the paper).

Rather than viewing Table 2 as just a leaderboard, think of the baselines as controlled experiments designed to answer three core questions:
1. Does Self-RAG win because of its $150\text{k}$ training dataset, or because of its reflection tokens and critique-guided inference?
   To answer this, the authors trained a special controlled baseline called `Llama2-FT 7B` on the exact same $150\text{k}$ instruction-output pairs as `Self-RAG 7B`, but without reflection tokens or interleaved passages during training, and then augmented `Llama2-FT 7B` with the exact same retrieved passages at test time. Comparing `Self-RAG 7B` against `Llama2-FT 7B` isolates the exact contribution of the Self-RAG framework.
2. When does conventional test-time RAG help, and when does it fail?
   Comparing each model without retrieval (`Llama2`, `Alpaca`, `Llama2-c`, `ChatGPT`) against its conventional RAG counterpart (`Ret-Llama2`, `Ret-Alpaca`, `Ret-Llama2-c`, `Ret-ChatGPT`) reveals a sharp split between extractive tasks (where answers can be copied from passages) and reasoning/verification tasks (where irrelevant passages distract the model).
3. Can a 7B or 13B model with learned self-reflection outperform much larger models ($65\text{B}$ `CoVe`) and proprietary RLHF systems (`ChatGPT`, `Ret-ChatGPT`, `Perplexity.ai`)?

---

## 2. Complete Taxonomy of All Baselines in Table 2 (Section 4.2)

Section 4.2 groups the compared models into three categories:

| Category | Model Name in Table 2 | Base Architecture & Scale | Training Data & Method | How Retrieval Is Used at Test Time |
|---|---|---|---|---|
| Proprietary / Private Data LMs | `Llama2-c 13B` | Llama-2 $13\text{B}$ | Meta's private SFT + multi-reward RLHF (`Llama-2-13b-chat-hf`) | None (parametric only) |
| Proprietary / Private Data LMs | `Ret-Llama2-c 13B` | Llama-2 $13\text{B}$ | Same as `Llama2-c 13B` | Prepend top-$K$ retrieved passages to `[INST]...[/INST]` prompt |
| Proprietary / Private Data LMs | `ChatGPT` | OpenAI `gpt-3.5-turbo` | Proprietary SFT + RLHF | None (parametric only) |
| Proprietary / Private Data LMs | `Ret-ChatGPT` | OpenAI `gpt-3.5-turbo` | Proprietary SFT + RLHF | Prepend top-$K$ retrieved passages to prompt |
| Proprietary / Private Data LMs | `Perplexity.ai` | InstructGPT-based | Production commercial search engine | Commercial web search + RAG pipeline |
| Baselines Without Retrieval | `Llama2 7B`, `13B` | Llama-2 $7\text{B}$, $13\text{B}$ | Pre-trained base models (`meta-llama/Llama-2-7b-hf`, `13b-hf`) | None (parametric only) |
| Baselines Without Retrieval | `Alpaca 7B`, `13B` | Llama-2 $7\text{B}$, $13\text{B}$ | Authors' replication of Stanford Alpaca SFT on Llama-2 | None (parametric only) |
| Baselines Without Retrieval | `CoVe 65B*` | LLaMA $65\text{B}$ (the paper writes "Llama2 65B", but Llama 2 has no 65B model; CoVe used the original LLaMA 65B) | Chain-of-Verification: multi-turn iterative prompting (draft $\to$ verification questions $\to$ verify $\to$ revise) | None (iterative self-verification prompting) |
| Baselines With Retrieval | `Toolformer* 6B` | GPT-J $6\text{B}$ | Pre-trained to generate inline Wikipedia API search calls | Calls Wikipedia API during generation |
| Baselines With Retrieval | `Llama2 7B`, `13B` (with retrieval) | Llama-2 $7\text{B}$, $13\text{B}$ | Pre-trained base models | Prepend top-$K$ retrieved passages to prompt |
| Baselines With Retrieval | `Alpaca 7B`, `13B` (with retrieval) | Llama-2 $7\text{B}$, $13\text{B}$ | Stanford Alpaca SFT on Llama-2 | Prepend top-$K$ retrieved passages to prompt |
| Baselines With Retrieval | `Llama2-FT 7B` | Llama-2 $7\text{B}$ | Fine-tuned on the exact same $150\text{k}$ instruction-output pairs as Self-RAG, without reflection tokens or passages | Prepend top-$K$ retrieved passages to prompt |
| Baselines With Retrieval | `SAIL* 7B` | LLaMA $7\text{B}$ | Instruction-tuned on Alpaca data with top retrieved passages prepended before instructions | Prepend top retrieved passages to prompt |
| Self-RAG | `Our SELF-RAG 7B`, `13B` | Llama-2 $7\text{B}$, $13\text{B}$ | Trained on $\mathcal{D}_{\text{gen}}$ ($150\text{k}$ pairs augmented offline by Critic $\mathcal{C}$ and Retriever $\mathcal{R}$ with reflection tokens) | Adaptive retrieval ($\delta$) + parallel per-passage critique beam search |

---

## 3. Complete Quantitative Results (Table 2) and Five Core Findings (Section 5.1)

For direct reference, here is the complete Table 2 from Section 5.1 of the paper:

| Model | PopQA (`acc`) | TQA (`acc`) | Pub (`acc`) | ARC (`acc`) | Bio (`FS`) | ASQA (`em`) | ASQA (`rg`) | ASQA (`mau`) | ASQA (`pre`) | ASQA (`rec`) |
|---|---|---|---|---|---|---|---|---|---|---|
| `Llama2-c 13B` | $20.0$ | $59.3$ | $49.4$ | $38.4$ | $55.9$ | $22.4$ | $29.6$ | $28.6$ | – | – |
| `Ret-Llama2-c 13B` | $51.8$ | $59.8$ | $52.1$ | $37.9$ | $79.9$ | $32.8$ | $34.8$ | $43.8$ | $19.8$ | $36.1$ |
| `ChatGPT` | $29.3$ | $74.3$ | $70.1$ | $75.3$ | $71.8$ | $35.3$ | $36.2$ | $68.8$ | – | – |
| `Ret-ChatGPT` | $50.8$ | $65.7$ | $54.7$ | $75.3$ | – | $40.7$ | $39.9$ | $79.7$ | $65.1$ | $76.6$ |
| `Perplexity.ai` | – | – | – | – | $71.2$ | – | – | – | – | – |
| `Llama2 7B` (No Ret) | $14.7$ | $30.5$ | $34.2$ | $21.8$ | $44.5$ | $7.9$ | $15.3$ | $19.0$ | – | – |
| `Alpaca 7B` (No Ret) | $23.6$ | $54.5$ | $49.8$ | $45.0$ | $45.8$ | $18.8$ | $29.4$ | $61.7$ | – | – |
| `Llama2 13B` (No Ret) | $14.7$ | $38.5$ | $29.4$ | $29.4$ | $53.4$ | $7.2$ | $12.4$ | $16.0$ | – | – |
| `Alpaca 13B` (No Ret) | $24.4$ | $61.3$ | $55.5$ | $54.9$ | $50.2$ | $22.9$ | $32.0$ | $70.6$ | – | – |
| `CoVe 65B*` (No Ret) | – | – | – | – | $71.2$ | – | – | – | – | – |
| `Toolformer* 6B` | – | $48.8$ | – | – | – | – | – | – | – | – |
| `Llama2 7B` (With Ret) | $38.2$ | $42.5$ | $30.0$ | $48.0$ | $78.0$ | $15.2$ | $22.1$ | $32.0$ | $2.9$ | $4.0$ |
| `Alpaca 7B` (With Ret) | $46.7$ | $64.1$ | $40.2$ | $48.0$ | $76.6$ | $30.9$ | $33.3$ | $57.9$ | $5.5$ | $7.2$ |
| `Llama2-FT 7B` (With Ret) | $48.7$ | $57.3$ | $64.3$ | $65.8$ | $78.2$ | $31.0$ | $35.8$ | $51.2$ | $5.0$ | $7.5$ |
| `SAIL* 7B` | – | – | $69.2$ | $48.4$ | – | – | – | – | – | – |
| `Llama2 13B` (With Ret) | $45.7$ | $47.0$ | $30.2$ | $26.0$ | $77.5$ | $16.3$ | $20.5$ | $24.7$ | $2.3$ | $3.6$ |
| `Alpaca 13B` (With Ret) | $46.1$ | $66.9$ | $51.1$ | $57.6$ | $77.7$ | $34.8$ | $36.7$ | $56.6$ | $2.0$ | $3.8$ |
| `Our SELF-RAG 7B` | $54.9$ | $66.4$ | $72.4$ | $67.3$ | $81.2$ | $30.0$ | $35.7$ | $74.3$ | $66.9$ | $67.8$ |
| `Our SELF-RAG 13B` | $55.8$ | $69.3$ | $74.5$ | $73.1$ | $80.2$ | $31.7$ | $37.0$ | $71.6$ | $70.3$ | $71.3$ |

Let us analyze the five major empirical findings established by Table 2.

### 3.1 Finding 1: Isolating the Self-RAG Framework Gain (`Self-RAG 7B` vs `Llama2-FT 7B`)

Compare `Llama2-FT 7B` and `Our SELF-RAG 7B`. Both start from the exact same `Llama-2 7B` base weights, both are trained on the exact same $150\text{k}$ instruction-output pairs, and both receive the exact same retrieved passages at test time. The only difference is that `Self-RAG 7B` is trained with interleaved passages and reflection tokens and uses critique-guided decoding:

| Metric | `Llama2-FT 7B` (Standard SFT + Test-Time RAG) | `Our SELF-RAG 7B` | Absolute Improvement ($\Delta$) |
|---|---|---|---|
| PopQA (`acc`) | $48.7\%$ | $54.9\%$ | $+6.2\%$ |
| TriviaQA (`acc`) | $57.3\%$ | $66.4\%$ | $+9.1\%$ |
| PubHealth (`acc`) | $64.3\%$ | $72.4\%$ | $+8.1\%$ |
| ARC-Challenge (`acc`) | $65.8\%$ | $67.3\%$ | $+1.5\%$ |
| Biography (`FactScore`) | $78.2\%$ | $81.2\%$ | $+3.0\%$ |
| ASQA Correctness (`str-em`) | $31.0\%$ | $30.0\%$ | $-1.0\%$ |
| ASQA Correctness (`ROUGE-L`) | $35.8$ | $35.7$ | $-0.1$ |
| ASQA Fluency (`MAUVE`) | $51.2$ | $74.3$ | $+23.1$ |
| ASQA Citation Precision (`pre`) | $5.0\%$ | $66.9\%$ | $+61.9\%$ ($13.4\times$) |
| ASQA Citation Recall (`rec`) | $7.5\%$ | $67.8\%$ | $+60.3\%$ ($9.0\times$) |

This comparison argues against the hypothesis that Self-RAG's performance comes merely from its $150\text{k}$ instruction-tuning mixture: without reflection tokens and parallel passage critique, `Llama2-FT 7B` lags behind on every column except ASQA correctness, where it is marginally higher (`str-em` $31.0$ vs $30.0$, `ROUGE-L` $35.8$ vs $35.7$), and it collapses on citation accuracy ($5.0\%$ vs $66.9\%$). The paper's own wording is that Llama2-FT "lags behind Self-RAG".

### 3.2 Finding 2: Why Conventional RAG Helps on Extractive Tasks (`PopQA`, `Bio`) but Not Reliably on Closed-Set Tasks (`PubHealth`, `ARC-C`)

Compare what happens when conventional test-time RAG is added to `Llama2`, `Alpaca`, `Llama2-c`, and `ChatGPT`:
- Extractive tasks (`PopQA` and `Bio`):
  When answering a rare entity question on `PopQA` or writing a biography on `Bio`, the model can often extract or paraphrase a span directly from a relevant Wikipedia passage. Consequently, conventional RAG produces large gains over no-retrieval (for example, `Alpaca 7B` improves from $23.6\% \to 46.7\%$ on PopQA and from $45.8\% \to 76.6\%$ on Bio).
- Reasoning and verification tasks (`PubHealth` and `ARC-C`):
  On `PubHealth` and `ARC-C`, the target output (`"true"`/`"false"` or `"A"`..`"D"`) does not appear as a literal substring inside the retrieved Wikipedia passage; the model must reason over the passage and map its conclusion to a class label. The paper summarizes that baselines with retrieval "do not improve performance notably" on these two tasks, and the per-model changes are mixed:
  - `PubHealth`: `ChatGPT` $70.1 \to 54.7$ ($-15.4$), `Alpaca 7B` $49.8 \to 40.2$ ($-9.6$), `Alpaca 13B` $55.5 \to 51.1$ ($-4.4$), `Llama2 7B` $34.2 \to 30.0$ ($-4.2$), but `Llama2 13B` $29.4 \to 30.2$ ($+0.8$) and `Llama2-c 13B` $49.4 \to 52.1$ ($+2.7$).
  - `ARC-C`: `Llama2 7B` $21.8 \to 48.0$ ($+26.2$), `Alpaca 7B` $45.0 \to 48.0$ ($+3.0$), `Alpaca 13B` $54.9 \to 57.6$ ($+2.7$), `ChatGPT` unchanged at $75.3$, `Llama2-c 13B` $38.4 \to 37.9$ ($-0.5$), and `Llama2 13B` $29.4 \to 26.0$ ($-3.4$).

  Why does conventional RAG not help reliably here? The paper's explanation is that these baselines "provide limited solutions for tasks where we cannot simply copy or extract sub-strings of retrieved passages". A plausible additional factor (not tested in the paper) is that off-the-shelf Contriever retrieves passages that share terms with the claim without addressing it, and concatenating those noisy passages distracts the model (Failure Mode 2 from Module 02 Part 2).
  In contrast, Self-RAG processes passages in parallel, down-weights `[Irrelevant]` passages, and aggregates critique scores across options (`closed = True`); `Self-RAG 7B` ($72.4\%$) and `13B` ($74.5\%$) outperform both `ChatGPT` ($70.1\%$) and `SAIL 7B` ($69.2\%$) on PubHealth. On ARC-C, however, `ChatGPT` ($75.3\%$) still beats both Self-RAG models ($67.3\%$ and $73.1\%$).

### 3.3 Finding 3: Massive Gains in Citation Precision and Recall on ASQA

Look at the last two columns of Table 2 (`(pre)` and `(rec)` on ASQA):
- Every open-weights baseline with retrieval (`Llama2 7B/13B`, `Alpaca 7B/13B`, `Llama2-FT 7B`) scores below $6.0\%$ citation precision and below $8.0\%$ citation recall, and even `Ret-Llama2-c 13B` reaches only $19.8\%$ precision and $36.1\%$ recall.
- Why do conventional RAG baselines fail so badly at citations? Because when all $5$ passages `[1]..[5]` are pasted into one prompt, a model that was not trained with segment-level attribution tokens (`IsSup`) either forgets to output `[k]` brackets or guesses passage numbers that do not entail the sentence under the T5-XXL NLI check.
- `Self-RAG 7B` ($66.9\%$ precision, $67.8\%$ recall) and `Self-RAG 13B` ($70.3\%$ precision, $71.3\%$ recall) achieve over $10\times$ higher citation accuracy than open-weights RAG baselines, and both 7B and 13B surpass `Ret-ChatGPT` ($65.1\%$) in citation precision.

### 3.4 Finding 4: Why `Self-RAG 7B` Slightly Outperforms `Self-RAG 13B` on `FactScore` ($81.2$ vs $80.2$) and `MAUVE` ($74.3$ vs $71.6$)

Compare the last two rows of Table 2 (`Our SELF-RAG 7B` vs `Our SELF-RAG 13B`):
- On $8$ out of $10$ columns (`PopQA`, `TQA`, `Pub`, `ARC`, and ASQA `em`, `rg`, `pre`, `rec`), scaling from $7\text{B}$ to $13\text{B}$ improves performance (for example, `ARC-C` jumps from $67.3\% \to 73.1\%$ and `TQA` jumps from $66.4\% \to 69.3\%$).
- However, on biography `FactScore` (`81.2` vs `80.2`) and ASQA `MAUVE` (`74.3` vs `71.6`), the smaller $7\text{B}$ model scores slightly higher than the $13\text{B}$ model!

The authors explain only the factual-precision gap (Section 5.1): `Self-RAG 7B` tends to generate "precisely grounded yet shorter outputs", which favors a precision-based metric like `FactScore`. The paper offers no explanation for the `MAUVE` gap. Extending the argument, the larger model may add details that the retrieved passage cannot verify, but this is an interpretation that the paper does not test.

### 3.5 Finding 5: Outperforming `CoVe 65B` and `Perplexity.ai` on Biographies

Finally, on biography generation (`Bio (FS)`), `Self-RAG 7B` ($81.2\%$) and `13B` ($80.2\%$) outperform:
- `CoVe 65B` ($71.2\%$), which is nearly $10\times$ larger than `Self-RAG 7B` and uses multi-step Chain-of-Verification prompting.
- `Perplexity.ai` ($71.2\%$) and `ChatGPT` ($71.8\%$).
- `Ret-Llama2-c 13B` ($79.9\%$).

### 3.6 Reading the Paper's Headline Claims Against Table 2

When quoting the paper's summary claims, check them against the table:
- Section 5.1 says Self-RAG outperforms ChatGPT on PubHealth, PopQA, biographies, and ASQA (ROUGE and MAUVE). This holds for both sizes except ASQA ROUGE, where only the 13B model ($37.0$) beats ChatGPT ($36.2$); the 7B model scores $35.7$.
- The abstract says Self-RAG outperforms ChatGPT on open-domain QA, reasoning, and fact verification. ChatGPT is higher on TriviaQA ($74.3$ vs $69.3$ for 13B) and on ARC-Challenge ($75.3$ vs $73.1$), so this holds only for PopQA and PubHealth.
- Section 5.1 also says Self-RAG obtains "the best performance among non-proprietary LM-based models on all tasks". On ASQA `str-em`, `Alpaca 13B` with retrieval ($34.8$) is higher than both Self-RAG models ($30.0$ and $31.7$).

`study/paper_audit.md` lists these and the other issues in the paper in one place.
