# Tasks, Datasets, and External Metrics Explained

## 1. Plain-English Intuition: Why Evaluating Self-RAG Requires Six Benchmarks Across Three Regimes

If we only tested a retrieval-augmented language model on simple trivia questions (where the answer is a single name or date that can be copied verbatim from a passage), we would not know whether the model can reason over evidence, write coherent multi-sentence paragraphs, or cite its sources accurately.

To test factuality, reasoning, fluency, and source verifiability holistically, Section 4.1 of the Self-RAG paper evaluates models in a zero-shot setting (providing task instructions without any few-shot examples) across six benchmarks grouped into three regimes:
1. Closed-set reasoning and verification (PubHealth and ARC-Challenge): Tests whether the model can synthesize retrieved facts to classify complex health claims (`"true"` vs `"false"`) or solve multi-step science exam questions (`"A"`..`"D"`) where the answer cannot simply be copied as a substring from a passage.
2. Short-form open-domain QA (PopQA and TriviaQA-unfiltered): Tests factual recall on both rare, long-tail entities ($<100$ monthly Wikipedia page views on PopQA) and broad open-domain trivia ($11,313$ test queries on TriviaQA).
3. Long-form generation with citations (Biography generation and ALCE-ASQA): Tests whether multi-sentence outputs remain factually accurate at the level of individual atomic claims (`FactScore`), cover all interpretations of an ambiguous question (`str-em`, `ROUGE-L`), maintain natural human-like fluency (`MAUVE`), and attach accurate inline citations `[1][2]` to every sentence (`Citation Precision` and `Citation Recall`).

---

## 2. Complete Specification of the Six Benchmarks and Retrieval Setups (Section 4.1, 4.3 & Appendix B.2)

The table below summarizes the dataset splits, retrieval corpora, decoding limits, and evaluation metrics for all six tasks (from Section 4.1, Section 4.3, Appendix B.2, and Table 5):

| Benchmark | Regime | Test Set Size | Retrieval Corpus & Top-$K$ Passages | Max New Tokens | Zero-Shot Instruction (Table 5) | Primary Metric(s) |
|---|---|---|---|---|---|---|
| PubHealth (`Pub`) | Closed-set fact verification | Test split | 2018 Wikipedia via Contriever-MS MARCO ($K = 5$) | $50$ | `"Is the following statement correct or not? Say true if it's correct; otherwise, say false. Don't capitalize or add periods, just say 'true' or 'false'."` | Accuracy (`acc`) |
| ARC-Challenge (`ARC`) | Closed-set scientific MCQA | Test split | 2018 Wikipedia via Contriever-MS MARCO ($K = 5$) | $50$ | `"Given four answer candidates, A, B, C and D, choose the best answer choice. Please answer with the capitalized alphabet only, without adding any extra phrase or period."` | Accuracy (`acc`) |
| PopQA (`PopQA`) | Short-form open-domain QA | $1,399$ long-tail queries ($<100$ views/mo) | Dec 2020 Atlas Wikipedia via Contriever ($5$) + Google Programmable Search Wiki intros ($5$, total $K = 10$) | $100$ | None (raw question used directly) | Inclusion Match (`acc`) |
| TriviaQA-unfiltered (`TQA`) | Short-form open-domain QA | $11,313$ open test queries | 2018 Wikipedia via Contriever ($5$) + Google Programmable Search Wiki intros ($5$, total $K = 10$) | $100$ | None (raw question used directly) | Inclusion Match (`acc`) |
| Biography Generation (`Bio`) | Long-form generation | Entity bio prompts | 2018 Wikipedia via Contriever ($5$) + Google Programmable Search Wiki intros ($5$) | $500$ (baselines) / $200$ per segment (Self-RAG; the README command uses $300$) | `"Tell me a bio about [Person Name]"` | `FactScore` (`FS`) |
| ALCE-ASQA (`ASQA`) | Long-form ambiguous QA with citations | ALCE test split | Author-provided top-$5$ passages from GTR-XXL ($K = 5$, shared across all models) | $100$ (baselines) / $300$ per segment in the README command (Self-RAG; the paper states no Self-RAG limit for ASQA but sets $\delta = 0$) | `"Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."` | `str-em` (`em`), `ROUGE-L` (`rg`), `MAUVE` (`mau`), Citation Precision (`pre`) & Recall (`rec`) |

Notice two experimental design choices:
1. Train-test domain separation on closed-set tasks (an observation from Table 3 and Section 4.1): During training (Module 04 Part 3), the seed dataset included Wikipedia fact verification (`FEVER`) and easier science questions (`ARC-Easy` and `OpenBookQA`), whereas test-time evaluation uses public-health clinical claims (`PubHealth`) and the harder challenge split (`ARC-Challenge`).
2. Controlling output length on ALCE-ASQA (Appendix B.2): In the original ALCE benchmark (`max_tokens = 300`), models like Llama-2-chat generate long $100$-token responses while ChatGPT generates concise $40$-token responses. Because `str-em` checks whether gold short answers appear anywhere as substrings in the output, longer responses artificially inflate `str-em`. Limiting baselines to `max_new_tokens = 100` ensures all models produce comparable $30$ to $60$ token outputs.

---

## 3. Step-by-Step Build-Up of the Specialized Long-Form Metrics

In Module 09 Part 2 (`metrics.py`), we already walked through the code for short-form and closed-set metrics (`match` and `accuracy`). Here, we build up the four external long-form evaluation metrics (`FactScore`, `str-em` / `ROUGE-L`, `MAUVE`, and ALCE `Citation Precision` / `Recall`) from first principles so every column of Table 2 is completely transparent.

### 3.1 `FactScore` for Biography Generation (`Bio (FS)`)

Suppose a model is prompted with `"Tell me a bio about Ronaldinho"` and generates a 4-sentence paragraph $y$. Why can we not evaluate $y$ by computing word overlap (BLEU or ROUGE) against one reference paragraph? Because a Wikipedia article contains hundreds of facts, and two biographies can be $100\%$ factually accurate while selecting different subsets of facts or phrasings.

`FactScore` (`factscore==0.1.5`, run via `python -m factscore.factscorer --model_name retrieval+ChatGPT`) evaluates factual precision in two stages:

1. Stage 1 (Decompose $y$ into atomic facts):
   An LLM splits each generated sentence in $y$ into a list of short, indivisible statements called atomic facts $\mathcal{A}(y) = \{a_1, a_2, \dots, a_M\}$, where each $a_m$ contains a single piece of information.
   For example, the generated sentence:
   `"Ronaldinho is a retired professional footballer from Brazil who won two FIFA World Player of the Year awards."`
   is decomposed into $M = 4$ atomic facts:
   - $a_1$: `"Ronaldinho is a retired professional footballer."`
   - $a_2$: `"Ronaldinho is from Brazil."`
   - $a_3$: `"Ronaldinho won the FIFA World Player of the Year award."`
   - $a_4$: `"Ronaldinho won the FIFA World Player of the Year award twice."`

2. Stage 2 (Verify each atomic fact against Wikipedia):
   For each atomic fact $a_m \in \mathcal{A}(y)$, passages are retrieved from the entity's Wikipedia article, and the evaluator checks whether $a_m$ is supported by the article ($\mathbb{I}[\text{supported}(a_m)] = 1$) or unsupported ($0$).
   The `FactScore` of biography $y$ is the percentage of generated atomic facts that are supported:

   $$\text{FactScore}(y) = \frac{1}{\vert \mathcal{A}(y) \vert} \sum_{m=1}^{\vert \mathcal{A}(y) \vert} \mathbb{I}\left[\text{atomic fact } a_m \text{ is supported by Wikipedia}\right] \times 100\%$$

   where:
   - $\mathcal{A}(y)$ is the set of atomic facts extracted from the model's biography $y$.
   - $\vert \mathcal{A}(y) \vert \in \mathbb{Z}^+$ is the total number of atomic claims made by the model.
   - Notice that because $\vert \mathcal{A}(y) \vert$ is in the denominator, `FactScore` is a measure of factual precision: a model that writes a shorter biography containing $8$ atomic facts that are all true ($8 / 8 = 100\%$) achieves a higher `FactScore` than a model that writes $12$ true facts plus $2$ unverified details ($12 / 14 = 85.7\%$).

### 3.2 ASQA Correctness: Disambiguated String Exact Match (`str-em`) and `ROUGE-L` (`rg`)

In ASQA (Ambiguous Question Answering), each test question $x$ is underspecified and has $I \ge 2$ distinct valid interpretations $\{q_1, q_2, \dots, q_I\}$, where interpretation $q_i$ has a set of acceptable gold short-answer strings $G_i$.

For example, if the ambiguous question is `"Who was the ruler of the Roman Empire when Sunday became a day of rest and when did it happen?"`, $G_1 = \{\text{``Constantine''}, \text{``Constantine the Great''}\}$ and $G_2 = \{\text{``321 AD''}, \text{``AD 321''}, \text{``March 7, 321''}\}$.

1. String Exact Match (`str-em`, column `(em)` in Table 2):
   Measures completeness by checking what fraction of the $I$ disambiguated answer groups $G_1, \dots, G_I$ have at least one normalized gold string $g \in G_i$ present inside the model's normalized long-form output $y$:

   $$\text{str-em}(y) = \frac{1}{I} \sum_{i=1}^{I} \mathbb{I}\left[\exists \, g \in G_i \text{ such that } \text{normalize}(g) \text{ is a substring of } \text{normalize}(y)\right] \times 100\%$$

2. `ROUGE-L` (column `(rg)` in Table 2):
   Measures n-gram sequence overlap against the human-written reference long answer $y_{\text{ref}}$ using the length of their Longest Common Subsequence $\text{LCS}(y, y_{\text{ref}})$ (the longest sequence of words appearing in both $y$ and $y_{\text{ref}}$ in the same relative order):

   $$P_{\text{LCS}} = \frac{\text{LCS}(y, y_{\text{ref}})}{\vert y \vert}, \quad R_{\text{LCS}} = \frac{\text{LCS}(y, y_{\text{ref}})}{\vert y_{\text{ref}} \vert}, \quad \text{ROUGE-L}(y, y_{\text{ref}}) = \frac{2 \, P_{\text{LCS}} R_{\text{LCS}}}{P_{\text{LCS}} + R_{\text{LCS}}} \times 100\%$$

### 3.3 ASQA Fluency: `MAUVE` (`mau`)

A model could cheat on `str-em` by simply outputting a comma-separated list of keywords (`"Constantine, 321 AD, March 7"`) without writing fluent English sentences.

`MAUVE` (column `(mau)` in Table 2) measures how closely the distribution of model-generated paragraphs $Q$ matches the distribution of human-written reference paragraphs $P$ across the dataset:
1. Every human paragraph $y_{\text{ref}}$ and model paragraph $y$ is encoded into a vector using a pre-trained language model (GPT-2 Large).
2. The embedding space is quantized via $k$-means clustering into $C$ discrete clusters, producing a discrete probability distribution $P = (p_1, \dots, p_C)$ for human responses and $Q = (q_1, \dots, q_C)$ for model responses.
3. For a mixture weight $\lambda \in (0, 1)$, let $R_\lambda = \lambda P + (1 - \lambda) Q$. Computing the Kullback-Leibler divergences $D_{\text{KL}}(P \Vert R_\lambda)$ (which measures whether $Q$ misses modes of human text) and $D_{\text{KL}}(Q \Vert R_\lambda)$ (which measures whether $Q$ produces unnatural, repetitive, or truncated text) across all $\lambda \in (0, 1)$ traces a divergence frontier curve.
4. `MAUVE` is the area under this frontier curve (scaled from $0$ to $100$), where higher values indicate more fluent, natural, human-like prose.

### 3.4 ASQA Citation Accuracy: `Citation Recall` (`rec`) and `Citation Precision` (`pre`)

In ALCE-ASQA, the model is given $5$ retrieved passages $\{d_1, d_2, d_3, d_4, d_5\}$ and generates a response $y$ consisting of $T$ sentences $s_1, s_2, \dots, s_T$. Each sentence $s_t$ contains a set of bracketed passage citations $C_t \subseteq \{d_1, \dots, d_5\}$ (for example, `"Constantine made Sunday a day of rest in 321 AD [1][2]."` has cited set $C_t = \{d_1, d_2\}$).

To evaluate whether the citations in $C_t$ genuinely support the claim in sentence $s_t$, ALCE uses an 11-billion-parameter Natural Language Inference model (`google/t5_xxl_true_nli_mixture`). Given a premise passage (or concatenation of passages) $P$ and a hypothesis sentence $s_t$ (with `[k]` brackets removed), let:

$$\text{NLI}(P,\; s_t) = \begin{cases} 1 & \text{if premise } P \text{ entails hypothesis } s_t \\ 0 & \text{otherwise} \end{cases}$$

From this binary entailment check, ALCE defines `Citation Recall` and `Citation Precision`:

1. Citation Recall (`rec`):
   For each sentence $s_t$, concatenate all passages cited by $s_t$ ($\bigoplus_{d \in C_t} d$). The sentence has recall $1$ if $C_t \neq \emptyset$ and the concatenated cited passages entail $s_t$:

   $$\text{Recall}(s_t) = \begin{cases} \text{NLI}\left(\bigoplus_{d \in C_t} d,\; s_t\right) & \text{if } C_t \neq \emptyset \\ 0 & \text{if } C_t = \emptyset \end{cases}$$

   Dataset-level `Citation Recall` is the average of $\text{Recall}(s_t)$ across all sentences.

2. Citation Precision (`pre`):
   Why is `Citation Recall` alone not enough? Because a model could game `Citation Recall` by citing all five passages `[1][2][3][4][5]` at the end of every sentence, even if only `[1]` was relevant and `[2][3][4][5]` were completely unrelated!
   `Citation Precision` checks every individual citation $d_k \in C_t$ to verify that $d_k$ is not an irrelevant tag-along. Specifically, a cited passage $d_k \in C_t$ in sentence $s_t$ is counted as precise ($\text{Precise}(d_k, s_t) = 1$) if and only if:
   - The full cited set $C_t$ entails $s_t$ ($\text{Recall}(s_t) = 1$), AND
   - Either $d_k$ alone entails $s_t$ ($\text{NLI}(d_k, s_t) = 1$), OR removing $d_k$ from $C_t$ breaks entailment ($\text{NLI}(\bigoplus_{d \in C_t \setminus \{d_k\}} d, s_t) = 0$, meaning $d_k$ was strictly necessary to complete the proof of $s_t$).

   Let us trace a concrete 1-sentence example to see how `Recall` and `Precision` work together:
   - Suppose Sentence $s_1$ says `"Constantine made Sunday a day of rest in 321 AD [1][2]."`, where Passage $d_1$ fully supports the entire sentence on its own ($\text{NLI}(d_1, s_1) = 1$), while Passage $d_2$ is an unrelated article about Texas baby names ($\text{NLI}(d_2, s_1) = 0$).
   - Because $d_1 \oplus d_2$ together still entail $s_1$, $\text{Recall}(s_1) = 1$ ($100\%$).
   - However, when testing citation $d_2$ for precision: $d_2$ alone does not entail $s_1$ ($\text{NLI}(d_2, s_1) = 0$), AND removing $d_2$ leaves $C_1 \setminus \{d_2\} = \{d_1\}$, which still entails $s_1$ ($\text{NLI}(d_1, s_1) = 1$)! Therefore, $d_2$ is exposed as a redundant, irrelevant citation ($\text{Precise}(d_2, s_1) = 0$), cutting the precision on $s_1$ from $100\%$ down to $1 / 2 = 50\%$!
   - Because Self-RAG processes passages one by one in parallel and attaches only the single passage $d_{k^*}$ that generated segment $s_t$ ($\vert C_t \vert = 1$), it never attaches redundant multi-passage guesses, achieving $66.9\%$ (7B) and $70.3\%$ (13B) citation precision on ASQA (outperforming even `Ret-ChatGPT` at $65.1\%$).
