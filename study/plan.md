# Self-RAG Study Plan

This file defines the logical sequence of modules, part files, source mappings, and ordered technical topics for `study/`.

Status: all $34$ part files below are written and were reviewed line by line against the paper and the code. Two reference appendices (listed at the end) complete the set: `paper_audit.md` for issues in the paper and `codebase_audit.md` for issues in the released code. The part files keep the paper's own statements and note its errors rather than silently correcting them. Code line numbers cited in the part files were recorded while writing and some can drift from `reference/code/` by up to about $90$ lines; the numbers in `codebase_audit.md` are re-verified, so search by the quoted identifier when a cited line does not match.

## Pedagogical Structure for Every Part File

Every part file is narrowly scoped to a single cohesive topic and follows a four-step progression so that a reader starting with zero prior background can build complete technical mastery without logical jumps:
1. Plain-English Intuition and Motivation: Explain what problem we are trying to solve, what goes wrong with a naive approach, and the core idea that fixes it using simple language and a small, concrete 3-to-5 token toy scenario.
2. Step-by-Step Mathematical Build-Up: Introduce one symbol at a time, plug concrete numbers from the toy scenario into the formula alongside the general equation, show every intermediate algebraic step, and define every symbol, subscript, and tensor shape immediately after each equation.
3. Explicit Index and Tensor Walkthrough: Show small, concrete tables of array indices, token strings, tensor dimensions, and values before and after each transformation.
4. Paper and Codebase Grounding: Connect the concept directly to the exact lines of Python code in `reference/code/` and the exact sections, equations, tables, and figures in `reference/paper/2310.11511.md`, explaining every line and any implementation bugs or edge cases.

---

## Module 01: How Language Models Read and Generate Text (From First Principles)

- [x] `01_lm_foundations/01_tokens_vocabulary_and_embeddings.md`
  - Read `reference/paper/2310.11511.md` (Section 3.1)
  - Read `reference/code/data_creation/train_special_tokens.py` (lines 27-31, 140-170) and `reference/code/retrieval_lm/finetune.py` (lines 222-236, 378-402)
  - Plain-English intuition: why neural networks cannot ingest raw text strings directly and how text is broken into numbered subword pieces called tokens
  - Subword tokenization (SentencePiece Byte-Pair Encoding in Llama-2): how words and punctuation are split into subwords, what the vocabulary $\mathcal{V}$ of size $V = 32000$ is, and the role of structural boundary tokens (`<s>`, `</s>`, `<unk>`, `<pad>`)
  - From integer token IDs to continuous vectors: what an input embedding matrix $E_{\text{in}} \in \mathbb{R}^{V \times d}$ is, how looking up token ID $u_k$ selects row $E_{\text{in}}[u_k, :] \in \mathbb{R}^d$, worked out with a concrete $V = 6, d = 3$ numerical table
  - How a single sentence of length $L$ becomes a 2D matrix of shape $[L, d]$, how a batch of $B$ sentences of different lengths is padded to the longest sequence length in the batch to form a 3D tensor $[B, L, d]$, and what `attention_mask` ($1$ for real tokens, $0$ for pad tokens) tracks

- [x] `01_lm_foundations/02_causal_attention_logits_and_softmax.md`
  - Read `reference/paper/2310.11511.md` (Section 3.1)
  - Plain-English intuition: once each token is a vector of numbers, how does the model combine meaning across earlier words in a sentence to predict the next word?
  - Step-by-step build-up of self-attention: Query ($q$, what the current position is looking for), Key ($k$, what each earlier position contains), and Value ($v$, the information passed forward), worked out with a tiny 3-token numerical example
  - Why causal (left-to-right) masking is required in text generation so position $i$ can only attend to positions $\le i$ and never peeks at future tokens $> i$
  - From the final hidden state vector $h_t \in \mathbb{R}^d$ to vocabulary logits $z_j = W_{\text{out}} h_t \in \mathbb{R}^V$: how the output projection matrix (LM head) computes a raw score (dot product) for every token in the vocabulary
  - Turning raw logits into a valid probability distribution via the softmax function $p_\theta(y_j = v \mid x, y_{<j})$, worked out with a 4-token numerical example, why subtracting $\max_{v'} z_{j,v'}$ prevents floating-point overflow, and why log-probabilities $\log p_\theta(y_j \mid x, y_{<j})$ are negative numbers in $(-\infty, 0]$
  - Combining single-token probabilities into the probability of a full response via the chain rule $p_\theta(y \mid x) = \prod_{j=1}^n p_\theta(y_j \mid x, y_{<j})$, and why summing log-probabilities $\sum_{j=1}^n \log p_\theta(y_j \mid x, y_{<j})$ prevents numerical underflow

- [x] `01_lm_foundations/03_training_teacher_forcing_and_loss_masking.md`
  - Read `reference/code/data_creation/train_special_tokens.py` (lines 171-210) and `reference/code/retrieval_lm/finetune.py` (lines 238-272)
  - Plain-English intuition: how do we train a language model to answer a prompt $x$ with a target response $y$?
  - Why letting an untrained model generate its own tokens during training fails (one wrong word corrupts the rest of the sentence), and how Teacher Forcing solves this by always feeding the true ground-truth prefix from the dataset
  - Why causal masking allows us to concatenate prompt $x$ and response $y$ into a single sequence $u$ of length $L = m + n$ and compute predictions for all prefixes simultaneously in one single GPU forward pass
  - Step-by-step visual and numerical walkthrough of the shift-by-one alignment (`shift_logits` and `shift_labels`) on an 8-token example (`["Capital", "of", "France?", "Paris", "is", "the", "capital.", "</s>"]`):
    - Why `labels` is initially created as an exact copy of `input_ids` in the codebase
    - Why `logits[i]` (computed after reading token $i$) is the prediction for token $i + 1$ (`labels[i + 1]`), and why comparing `logits[i]` to `labels[i]` would merely copy the current token
    - How slicing `shift_logits = logits[..., :-1, :]` (dropping the last logit) and `shift_labels = labels[..., 1:]` (dropping the first label) inside `model(**batch)` aligns `logits[k]` with `labels[k + 1]` at every index $k$
  - Cross-entropy (negative log-likelihood) loss derived step-by-step with concrete numbers, and how setting `labels[:source_len] = -100` (`IGNORE_INDEX`) masks out the user prompt $x$ so loss and gradients are computed strictly on the target response $y$

- [x] `01_lm_foundations/04_decoding_strategies_and_sequence_scoring.md`
  - Read `reference/code/retrieval_lm/run_baseline_lm.py` (lines 47-53), `reference/code/retrieval_lm/run_short_form.py` (lines 80-96, 147-152), and `reference/code/retrieval_lm/run_long_form_static.py` (lines 19-33, 78-83)
  - Plain-English intuition: at test time, there is no ground-truth target $y$; the model must generate one token at a time and append each chosen token to its own input for the next step
  - Step-by-step walkthrough of token selection strategies using a concrete 5-token probability table:
    - Greedy decoding ($\hat{y}_j = \arg\max_v p_\theta(v \mid x, y_{<j})$)
    - Temperature scaling ($z_{j,v} / T$) and how $T < 1$, $T = 1$, $T > 1$, and $T \to 0^+$ reshape the distribution
    - Nucleus (`top_p`) sampling: sorting probabilities, accumulating up to threshold $p_{\text{top}}$, zeroing the tail, and renormalizing
  - Comparing candidate sentences of different lengths: why raw cumulative log-probability $\sum_{j=1}^n \log p_\theta(y_j \mid \cdot)$ penalizes longer sentences (since every log-probability is negative), and how dividing by token length $n$ (`seq_score = cumulative_logprob / max(len(token_ids), 1)`) and exponentiating (`np.exp(seq_score)`) recovers the geometric mean token probability in $(0, 1]$

---

## Module 02: Parametric Memory Limits, Conventional RAG, and Baseline Code

- [x] `02_rag_and_baselines/01_parametric_memory_and_hallucination.md`
  - Read `reference/paper/2310.11511.md` (Abstract, Section 1, Section 4.1, Section 5.1, Table 2)
  - Plain-English intuition: what it means for a language model to store world knowledge inside its weights (parametric memory $\theta$), and why a fluent model confidently states false facts (hallucination)
  - How pre-training acts as lossy compression of a text corpus into a fixed number of parameters ($7\text{B}$ or $13\text{B}$ floats): why high-frequency facts are retained while long-tail entities (such as the 1,399 PopQA entities with $<100$ monthly Wikipedia page views, where Llama-2 7B/13B score only $14.7\%$) are forgotten
  - Why maximizing next-token plausibility $p_\theta(y_j \mid x, y_{<j})$ produces fluent grammatical templates filled with fabricated dates and names when factual weights are weak (illustrated with biography generation and FactScore in Table 2)
  - Why scaling model parameters ($7\text{B} \to 13\text{B} \to 65\text{B}$) and instruction tuning (Alpaca, Llama2-chat, ChatGPT) fail to eliminate hallucinations or provide verifiable source citations

- [x] `02_rag_and_baselines/02_conventional_rag_and_its_three_failure_modes.md`
  - Read `reference/paper/2310.11511.md` (Abstract, Section 1, Section 2)
  - Read `reference/paper/images/_page_1_Figure_1.jpeg` and `reference/code/images/teaser_self_rag_v8.png`
  - Plain-English intuition: how non-parametric memory (an external searchable index of millions of Wikipedia passages $\mathcal{C}_{\text{corpus}}$) supplements parametric memory via Retrieval-Augmented Generation (RAG)
  - Mathematical formulation of conventional RAG (Figure 1 left): retrieving a fixed number of top-$K$ passages $D_K = (d_1, \dots, d_K)$ given query $x$, prepending all $K$ passages to the prompt, and generating $p_\theta(y \mid d_1, \dots, d_K, x)$
  - How concatenating $K$ passages into one long prefix affects self-attention weights: showing algebraically and intuitively how irrelevant passages compete in the softmax denominator and inject distracting value vectors into the hidden state
  - Detailed walkthrough of the three structural failure modes of conventional RAG using Figure 1 (left):
    - Failure Mode 1 (Indiscriminate retrieval): retrieving Wikipedia passages even for creative prompts (`Write an essay of your best summer vacation`), hurting versatility and adding unnecessary latency
    - Failure Mode 2 (Distraction and contradiction from concatenated passages): tracing `How did US states get their names?` with Passages 1, 2, and 3 to see why the LM contradicts Passage 3 (claiming California was named after Christopher Columbus) and hallucinates from keyword overlap in Passage 2 (Texas baby names $\to$ Texas named after a Native American tribe)
    - Failure Mode 3 (Single-pass static retrieval without verification): retrieving only once at $t = 0$ and never verifying whether each generated claim is actually entailed by a cited passage

- [x] `02_rag_and_baselines/03_baseline_evaluation_code_walkthrough.md`
  - Read `reference/paper/2310.11511.md` (Section 4.2, Appendix B.2, Table 5)
  - Read `reference/code/retrieval_lm/run_baseline_lm.py` and `reference/code/retrieval_lm/utils.py` (lines 6-44, 155-190)
  - Plain-English overview of how the non-retrieval and conventional RAG baselines in Table 2 are executed in `reference/code/`
  - Line-by-line walkthrough of `PROMPT_DICT` and `TASK_INST` in `retrieval_lm/utils.py`: comparing the prompt templates for Alpaca/Llama2-FT (`prompt_no_input`, `prompt_no_input_retrieval`), Open-Instruct (`<user>` / `<assistant>`), and Llama2-chat (`[INST]` / `[/INST]`)
  - Line-by-line walkthrough of `run_baseline_lm.py`:
    - Loading pre-retrieved documents in `--mode retrieval`, slicing `ctxs[:args.top_n]`, formatting `[1] title\ntext\n[2] ...`, and storing the concatenated string in `item["paragraph"]`
    - Unifying ground-truth answer fields (`output`, `answers`, `possible_answers` via `ast.literal_eval`, `answerKey`) into `item["golds"]` and prepending `TASK_INST` for `fever` (PubHealth) and `arc_c`
    - Running vLLM (`temperature=0.8, top_p=0.95`, truncating at `"\n\n"[0]`) vs OpenAI APIs (`text-davinci-003` and `gpt-3.5-turbo`)
    - Closed-set answer normalization (`postprocess_answers_closed`) and diagnosing the two bugs in `run_baseline_lm.py`: line 43 assigning `results` instead of `result` on API error, and line 169 (`item["output"] = pred`) immediately overwriting `postprocess_answers_closed` in the main batch loop

---

## Module 03: The Self-RAG Architecture and Surrounding Literature

- [x] `03_self_rag_overview/01_self_rag_core_idea_and_inference_cycle.md`
  - Read `reference/paper/2310.11511.md` (Section 1, Section 3.1, Table 1, Algorithm 1)
  - Read `reference/paper/images/_page_1_Figure_1.jpeg`
  - Plain-English intuition: how Self-RAG fixes all three failure modes of conventional RAG by turning the language model itself into an active researcher and self-critic
  - Segment-level generation $y = (y_1, y_2, \dots, y_T)$ over an expanded vocabulary $\mathcal{V}_{\text{exp}} = \mathcal{V} \cup \mathcal{V}_{\text{ref}}$ where each segment $y_t$ is one sentence interleaved with special reflection tokens
  - Step-by-step walkthrough of the three-step Self-RAG inference cycle (Algorithm 1 and Figure 1 right):
    - Step 1 (Retrieve on demand): predicting `Retrieve` (`[Retrieval]` vs `[No Retrieval]`) given $(x, y_{<t})$ so creative prompts skip retrieval completely while factual claims trigger retriever $\mathcal{R}$ at any sentence boundary
    - Step 2 (Generate segments in parallel per passage): why processing $K$ retrieved passages as $K$ separate batch items (`Prompt + d_1`, `Prompt + d_2`, ..., `Prompt + d_K`) instead of concatenating them eliminates cross-passage distraction, and how $\mathcal{M}$ predicts `IsRel` (`[Relevant]` vs `[Irrelevant]`) before generating candidate sentence $y_t^{(k)}$
    - Step 3 (Critique outputs and select best segment): how $\mathcal{M}$ predicts `IsSup` (`[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`) and `IsUse` (`[Utility:1]`..`[Utility:5]`) to score, rank, and cite the best candidate sentence (tracing the exact US state names example in Figure 1 right)

- [x] `03_self_rag_overview/02_self_rag_two_model_training_overview.md`
  - Read `reference/paper/2310.11511.md` (Section 3.1 Training overview, Section 3.2, Figure 2, Appendix A.2 Algorithm 2)
  - Read `reference/paper/images/_page_4_Diagram_1.jpeg`
  - Plain-English intuition: how do we teach a base Llama-2 model to output reflection tokens and grounded sentences when standard training datasets only contain plain $(x, y)$ text pairs?
  - Step-by-step walkthrough of the four-stage offline training pipeline (Algorithm 2 and Figure 2):
    - Stage 1 (GPT-4 supervision collection): prompting GPT-4 on a small sampled subset $(X_{\text{sample}}, Y_{\text{sample}})$ to collect reflection token labels $\mathcal{D}_{\text{critic}}$
    - Stage 2 (Critic LM training): fine-tuning a Llama-2 7B Critic $\mathcal{C}$ on $\mathcal{D}_{\text{critic}}$ via conditional next-token prediction (Equation 1) so we have a fast, local, reproducible evaluator
    - Stage 3 (Generator corpus construction): running Retriever $\mathcal{R}$ and Critic $\mathcal{C}$ offline across all 145,619 seed training examples to insert `<paragraph>d</paragraph>` blocks and reflection tokens at every sentence, creating $\mathcal{D}_{\text{gen}}$ (walking through both the summer vacation essay and the US state names examples in Figure 2)
    - Stage 4 (Generator LM training): training the final Generator $\mathcal{M}$ (Llama-2 7B or 13B) on $\mathcal{D}_{\text{gen}}$ with standard next-token prediction loss (Equation 2) while masking `<paragraph>d</paragraph>` tokens, so $\mathcal{M}$ internalizes both generation and self-critique in a single model

- [x] `03_self_rag_overview/03_comparison_with_rlhf_control_tokens_and_concurrent_rag.md`
  - Read `reference/paper/2310.11511.md` (Section 2, Section 3.2.2)
  - Read `reference/code/README.md`, `reference/code/setup.sh`, `reference/code/requirements.txt`, `reference/code/environment.yml`, and `reference/code/LICENSE`
  - First-principles explanation and comparison of related training and control paradigms:
    - Reinforcement Learning from Human Feedback (RLHF) via PPO: explaining how online PPO works (actor policy $\pi_\theta$, frozen reference $\pi_{\text{ref}}$, value critic $V_\phi$, reward models $R_g$, KL penalty) and why Self-RAG's offline critique insertion avoids multi-model GPU memory overhead, avoids policy-gradient instability, and enables changing preference weights at inference time without retraining
    - Prefix control tokens (CTRL, QUARK): why conditioning on a single reward token at the start of a prompt ($c \to y$) cannot evaluate individual retrieved passages or verify post-hoc sentence entailment ($d \to \text{IsRel} \to y_t \to \text{IsSup}$)
    - Concurrent RAG and iterative self-refinement methods (SAIL, RA-DIT, Toolformer, Active RAG, NLI passage filtering, Recomp, CoVe, Self-Refine, Refiner, LATS)
  - Complete architectural map of the `reference/code/` repository and its runtime environment (`vllm==0.2.6` and the need for `skip_special_tokens=False`, `flash-attn==2.3.6`, `deepspeed==0.12.6`, `faiss-gpu`, `spacy==3.7.2`, and isolated `factscore==0.1.5`)

---

## Module 04: Reflection Tokens, Vocabulary Expansion, and Seed Datasets

- [x] `04_tokens_and_data/01_reflection_token_taxonomy_and_grammar.md`
  - Read `reference/paper/2310.11511.md` (Section 3.1, Table 1, Figure 2, Appendix A.1, Appendix Table 4)
  - Plain-English definition, input arguments, output values, and exact role of all 15 special tokens added in Self-RAG:
    - Retrieval decision tokens (`Retrieve`): `[Retrieval]`, `[No Retrieval]`, `[Continue to Use Evidence]`
    - Passage relevance critique tokens (`IsRel`): `[Relevant]`, `[Irrelevant]`
    - Groundedness / attribution critique tokens (`IsSup`): `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`
    - Perceived utility critique tokens (`IsUse`): `[Utility:1]`, `[Utility:2]`, `[Utility:3]`, `[Utility:4]`, `[Utility:5]`
    - Context markup delimiters: `<paragraph>`, `</paragraph>`
  - Formal sequence grammar of a Self-RAG output: where each token is allowed to appear, why `IsSup` only appears after a `[Relevant]` passage segment (and is omitted after `[No Retrieval]` or `[Irrelevant]`), and why `IsUse` appears once at the very end of the full response
  - Token-by-token walkthrough of all 5 real training examples in Appendix Table 4 (computer memory, task delegation, sunset poem, Walking Dead Season 7, and German Shepherd chat)

- [x] `04_tokens_and_data/02_tokenizer_mechanics_and_embedding_expansion_code.md`
  - Read `reference/code/retrieval_lm/utils.py` (lines 45-98), `reference/code/data_creation/train_special_tokens.py` (lines 27-31, 140-170, 275-330), `reference/code/retrieval_lm/finetune.py` (lines 378-412), and `reference/code/data_creation/generator/postprocess_data.py` (lines 11-36)
  - Plain-English intuition: what happens if you feed `"[Fully supported]"` to a standard Llama-2 tokenizer without registering it as a special token (it fragments into multiple subword tokens), and why Self-RAG requires each reflection token to be a single atomic integer ID
  - How `tokenizer.add_special_tokens` registers the 15 tokens + pad token (`[PAD]` vs `<pad>`), expanding the vocabulary size from $32000$ to $32016$ and resizing both the input embedding matrix $E_{\text{in}}$ and output head $W_{\text{out}}$ from $[32000, d]$ to $[32016, d]$
  - Line-by-line walkthrough of `load_special_tokens`, `control_tokens`, `postprocess`, and `fix_spacing` in `reference/code/retrieval_lm/utils.py`
  - Comparing special token lists across `retrieval_lm/utils.py`, `train_special_tokens.py`, `finetune.py`, and `postprocess_data.py` (including the swapped variable names `ground_tokens_names` and `utility_tokens_names` in `postprocess_data.py` lines 13-16, and why `[Continue to Use Evidence]` is missing from `postprocess` in `utils.py` and `postprocess_data.py`)

- [x] `04_tokens_and_data/03_seed_datasets_and_preprocessing_scripts.md`
  - Read `reference/paper/2310.11511.md` (Section 4.3, Appendix A.2 Table 3)
  - Read `reference/code/data_creation/process_data/task_instructions.py`, `tulu.py`, `kilt_dpr.py`, `arc.py`, `obqa.py`, and `asqa.py`
  - Plain-English motivation: why the 145,619 generator seed corpus (Table 3) combines 92,008 general instruction-following examples (GPT-4 Alpaca, Stanford Alpaca, FLAN-V2, ShareGPT, OpenAssistant 1) with 53,611 knowledge-intensive examples (Wizard of Wikipedia, Natural Questions, FEVER, OpenBookQA, ARC-Easy, ASQA)
  - Target JSONL schema (`instruction`, `output`, `input`, `topic`, `id`, `dataset_name`) and `TASK_INST` in `task_instructions.py`
  - Line-by-line walkthrough of `tulu.py`: single-turn filtering (`len(messages) > 2`), 500-word length caps, newline cleanup, and explaining the Python boolean precedence bug on line 36 (`if dataset_name == "sharegpt" or "oasst1":`) that evaluates to truthy `"oasst1"` and runs `langdetect.detect` on every dataset
  - Line-by-line walkthrough of `kilt_dpr.py`, `arc.py`, `obqa.py`, and `asqa.py`: extracting KILT positive contexts, mapping FEVER `REFUTES`/`SUPPORTS` to `false`/`true`, mapping ARC numeric labels `1..4` to `A..D`, skipping questions with $<4$ choices, and extracting ASQA `long_answer` strings

---

## Module 05: The Dense Retrieval System from First Principles (Contriever and FAISS)

- [x] `05_dense_retrieval/01_dense_retrieval_intuition_and_contriever_encoder.md`
  - Read `reference/paper/2310.11511.md` (Section 3.1, Section 4.3, Appendix B.2)
  - Read `reference/code/retrieval_lm/src/__init__.py`, `contriever.py`, and `utils.py`
  - Plain-English intuition: how do we search 21 million Wikipedia passages when a user's question uses different words than the relevant Wikipedia article?
  - Sparse keyword matching (TF-IDF / BM25) vs Dense Bi-Encoder retrieval: mapping queries $q$ and passages $d$ independently into a shared 768-dimensional vector space $\mathbb{R}^{768}$ where semantic similarity is measured by the inner product (dot product) $\langle E(q), E(d) \rangle$
  - How a bidirectional transformer encoder (`BertModel` in `Contriever` and `XLMRobertaModel` in `XLMRetriever`) differs from a causal decoder: every token attends to both left and right context simultaneously
  - Step-by-step derivation and numerical 4-token padded example of masked average pooling in `src/contriever.py`:
    - Why padding tokens in a batch have non-zero hidden states after passing through transformer layers
    - How `last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)` zeroes out padding vectors across all 768 dimensions
    - How summing over sequence length (`dim=1`) and dividing by `attention_mask.sum(dim=1)[..., None]` computes the exact mean over real tokens only, contrasted with `cls` pooling (`last_hidden[:, 0]`) and optional $\ell_2$ normalization
  - Line-by-line walkthrough of `load_retriever` in `src/contriever.py` (stripping `encoder_q.` and `encoder.` prefixes from checkpoints) and training utilities in `src/utils.py` (`WarmupLinearScheduler`, `CosineScheduler`, `WeightedAvgStats`, `load_hf`)

- [x] `05_dense_retrieval/02_contrastive_learning_inbatch_and_moco.md`
  - Read `reference/code/retrieval_lm/src/inbatch.py`, `moco.py`, and `dist_utils.py`
  - Plain-English intuition: how do you train a bi-encoder so that matching query-passage pairs have a high dot product and non-matching pairs have a low dot product?
  - Step-by-step derivation of the InfoNCE contrastive loss from multi-class cross-entropy: treating the 1 matching passage $k_i^+$ as the correct class and $M$ non-matching passages $\{k_j^-\}$ as distractor classes, with temperature scaling $\tau$ and label smoothing
  - Line-by-line walkthrough of In-Batch negatives (`src/inbatch.py`) and distributed gradient gathering (`src/dist_utils.py`):
    - How every other passage in a mini-batch of size $B$ acts as a free negative example, forming a $[B, B]$ similarity matrix via `torch.einsum("id, jd->ij", ...)`
    - How `Gather(torch.autograd.Function)` in `src/dist_utils.py` gathers passage vectors across all $W$ GPUs (`dist.all_gather`) to expand the negative pool to $W \times B$, why standard `all_gather` drops gradients, how `dist.all_reduce` in `backward` restores exact gradients, and why target class indices are shifted by `rank * len(kemb)`
  - Line-by-line walkthrough of Momentum Contrast (`src/moco.py`):
    - Why In-Batch negatives are limited by GPU memory batch size $B$, and how MoCo maintains a rolling queue of $65536$ past passage embeddings (`queue` of shape $[768, 65536]$)
    - Why the key encoder $\theta_k$ cannot be updated by backpropagation on 65,536 old queue vectors, and how exponential moving average (momentum) updates $\theta_k \leftarrow m \theta_k + (1 - m)\theta_q$ ($m = 0.999$) keep the queue embeddings consistent, worked out alongside `_compute_logits` and `_dequeue_and_enqueue`

- [x] `05_dense_retrieval/03_retriever_data_augmentation_and_finetuning_data.md`
  - Read `reference/code/retrieval_lm/src/data.py`, `finetuning_data.py`, `normalize_text.py`, `options.py`, and `slurm.py`
  - Plain-English intuition: how Contriever learns general semantic similarity from raw unlabelled text before ever seeing human-written question-answer pairs
  - Line-by-line walkthrough of unsupervised contrastive data generation (`src/data.py`):
    - `Dataset.__getitem__`: slicing a fixed chunk of tokens and extracting two independent random sub-spans (`randomcrop` between `ratio_min` and `ratio_max`) to serve as positive query $q$ and key $k^+$
    - Token-level data augmentations (`deleteword`, `replaceword`, `maskword`, `shuffleword`), `add_bos_eos`, `build_mask` padding, and `MultiDataset` temperature-exponentiated dataset sampling $p_i \propto \vert D_i \vert^\alpha$
  - Line-by-line walkthrough of supervised retriever fine-tuning data (`src/finetuning_data.py`): sampling `positive_ctxs`, `negative_ctxs`, and BM25 `hard_negative_ctxs` (`sample_n_hard_negatives`)
  - Line-by-line walkthrough of Unicode character normalization (`src/normalize_text.py`), command-line configuration (`src/options.py`), and SLURM multi-node distributed initialization and job requeuing (`src/slurm.py`)

- [x] `05_dense_retrieval/04_passage_embedding_faiss_indexing_and_search.md`
  - Read `reference/paper/2310.11511.md` (Section 4.3, Appendix B.2)
  - Read `reference/code/retrieval_lm/generate_passage_embeddings.py`, `passage_retrieval.py`, `download_demo_corpus.sh`, `src/index.py`, `src/evaluation.py`, and `src/beir_utils.py`
  - Plain-English overview of how a trained Contriever model encodes 21M Wikipedia passages offline and answers search queries at runtime
  - Line-by-line walkthrough of `generate_passage_embeddings.py`: loading DPR `.tsv` and `.jsonl` files (`load_passages`), concatenating `title + " " + text`, splitting work across GPU shards (`shard_id`, `num_shards`), encoding in fp16, and saving `(allids, allembeddings)` pickle files
  - First-principles explanation and walkthrough of FAISS indexing (`src/index.py`):
    - Exact inner-product search (`faiss.IndexFlatIP(768)`) vs Product Quantization (`faiss.IndexPQ`): how PQ splits a 768-D float32 vector ($3072$ bytes) into $M$ sub-vectors, quantizes each sub-vector to an $n_{\text{bits}}$-bit centroid ID via $k$-means (`index.train`), and approximates dot products via lookup tables
    - Batch index population (`index_encoded_data`), saving/loading `index.faiss` and `index_meta.faiss`, and translating internal FAISS positions `0..N-1` to external passage IDs (`index_id_to_db_id`)
  - Line-by-line walkthrough of `passage_retrieval.py` and `download_demo_corpus.sh` (full Wikipedia vs `enwiki_2020_intro_only` demo, plus the 2018 vs December 2020 Wikipedia dump difference for PopQA in Appendix B.2)
  - Line-by-line walkthrough of retrieval evaluation in `src/evaluation.py` (`SimpleTokenizer`, NFD normalization, `has_answer`, `calculate_matches`, `count_inversions`) and `src/beir_utils.py` (`DenseEncoderModel`, `evaluate_model`)

---

## Module 06: Critic Model Distillation and Supervised Training

- [x] `06_critic_model/01_distilling_gpt4_supervision_and_prompts.md`
  - Read `reference/paper/2310.11511.md` (Section 3.2.1, Appendix A.1, Appendix Section D Tables 8-12)
  - Plain-English intuition: why annotating reflection tokens by hand is too expensive, why calling GPT-4 on all 145,619 training examples (and millions of candidate passages) is too costly and non-reproducible, and how knowledge distillation transfers GPT-4's evaluation skill into an in-house 7B Critic LM $\mathcal{C}$
  - Formal distillation setup $p_{\text{GPT-4}}(r \mid I_{\text{type}}, x, y)$, dataset collection counts (12,594 `Retrieve`, 19,317 `IsRel`, 11,181 `IsSup`, 3,831 `IsUse`), and human agreement rates (95% `Retrieve`, 95% `IsRel`, 90% `IsSup`, 80% `IsUse`)
  - Detailed walkthrough of all five GPT-4 instruction rubrics and few-shot demonstrations in Appendix Section D (Tables 8-12):
    - Table 8: Initial binary retrieval necessity (`Need retrieval? [Yes]` vs `[No]`) given instruction $x$ only
    - Table 9: Three-way sentence-level retrieval decision (`[Continue to Use Evidence]`, `[No Retrieval]`, `[Retrieval]`) given instruction $x$, preceding sentences $y_{<t}$, current evidence $d$, and target sentence $y_t$
    - Table 10: Passage relevance (`[Relevant]` vs `[Irrelevant]`) given instruction $x$, preceding context $y_{<t}$, and evidence $d$
    - Table 11: Three-way factual entailment (`[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`) strictly prohibiting external knowledge outside evidence $d$
    - Table 12: Five-point perceived utility (`1` to `5`) given instruction $x$ and complete output $y$

- [x] `06_critic_model/02_gpt4_annotation_scripts_and_code_bugs.md`
  - Read `reference/code/data_creation/critic/gpt4_reward/README.md`, `chatgpt_need_retrieval.py`, `chatgpt_relevance.py`, `chatgpt_groundness.py`, `chatgpt_utility.py`, and `chatgpt_inference.py`
  - Line-by-line walkthrough of `chatgpt_need_retrieval.py`: input schema, resuming from `--previous_results`, `PROMPT_DICT` variants (`context`, `multi_retrieval`, `multi_retrieval_no_preceding`, `multi_retrieval_three_way`, `multi_retrieval_three_way_no_preceding`), OpenAI API backoff, and `postprocess` splitting on `"\nExplanation:"`
  - Line-by-line walkthrough of `chatgpt_relevance.py`, `chatgpt_groundness.py`, `chatgpt_utility.py`, and `chatgpt_inference.py` (including `KNOWLEDGE_INSTRUCTIONS` for `nq`, `fever`, `wow` and filtering `"As a language model, I cannot"`)
  - Detailed diagnosis of the four implementation bugs across the `gpt4_reward/` scripts:
    - Bug 1 (`chatgpt_need_retrieval.py` lines 125-137): when `sent_idx == 0` and `three_way == True`, `process_input` selects `multi_retrieval_three_way` (which requires `{preceding_sentences}`), whereas when `sent_idx > 0` and `three_way == True`, it selects `multi_retrieval_three_way_no_preceding` (swapping the two templates)
    - Bug 2 (`chatgpt_groundness.py` lines 96-103 and `chatgpt_relevance.py` lines 76-83): `postprocess` searches for digits `1..5` inside `score_string`, which returns `score = ""` whenever GPT-4 outputs bracketed strings like `[Fully supported]` or `[Relevant]`
    - Bug 3 (`chatgpt_groundness.py` line 191, `chatgpt_relevance.py` line 154, `chatgpt_utility.py` line 105): `result_list.append` sets `"explanation": score` instead of `"explanation": explanation`
    - Bug 4 (`chatgpt_relevance.py` line 90): `process_input` with `multi_retrieval=False` looks up `PROMPT_DICT["context"]`, which does not exist in `chatgpt_relevance.py`

- [x] `06_critic_model/03_critic_dataset_assembly_and_training_script.md`
  - Read `reference/paper/2310.11511.md` (Section 3.2.1 Equation 1, Appendix A.2 Figure 5)
  - Read `reference/code/data_creation/critic/gpt4_reward/combine_chat_gpt_reward.py` and `reference/code/data_creation/train_special_tokens.py`
  - Line-by-line walkthrough of `combine_chat_gpt_reward.py`:
    - How `if item["score"] == "": item["score"] = raw_output.split("\n")[0]` recovers the bracketed labels lost by Bug 2 in `chatgpt_groundness.py` and `chatgpt_relevance.py`
    - How each task (`utility`, `retrieval`, `multi_retrieval`, `relevance`, `groudness`) is formatted into `{instruction, input, output, task}`
    - Why `[Relevant]` examples are downsampled by 30% (`if label == "[Relevant]" and random.random() > 0.7: continue`), how leaked `"### Response:"` strings are filtered out, and how the shuffled list is split into `dev_data = final_train_data[:1500]` and `train_data = final_train_data[1500:]`
  - Mathematical formulation of Critic training (Equation 1):
    $$\max_{\mathcal{C}} \mathbb{E}_{((x,y),r) \sim \mathcal{D}_{\text{critic}}} \log p_{\mathcal{C}}(r \mid x, y)$$
  - Line-by-line walkthrough of `data_creation/train_special_tokens.py`:
    - Mean embedding initialization in `smart_tokenizer_and_embedding_resize`: why initializing the 16 new token rows to the average of all 32,000 existing embeddings $\mathbf{e}_{\text{avg}} = \frac{1}{V} \sum_{i=1}^V \mathbf{e}_i$ keeps initial logit magnitudes balanced and prevents training instability
    - Tokenization and prompt loss masking in `preprocess` (`label[:source_len] = -100`), plus the optional `skip_tokens` and `context_markups` masking logic
    - Note on line 218 of `train_special_tokens.py`: unpacking undefined keys `PROMPT_DICT["prompt_no_input_paragraph"]` and `PROMPT_DICT["prompt_no_input_separated"]`
    - FSDP training hyperparameters and Critic prediction agreement results (Llama-2 7B vs FLAN-3B in Appendix Figure 5)

---

## Module 07: Generator Training Data Creation (Offline Critique and Passage Insertion)

- [x] `07_generator_data_creation/01_two_stage_retrieval_and_sentence_splitting.md`
  - Read `reference/paper/2310.11511.md` (Section 3.2.2, Appendix A.2 Algorithm 3)
  - Read `reference/code/data_creation/generator/README.md` and `create_retrieval_data.py`
  - Plain-English overview of the 6-step offline pipeline (`data_creation/generator/README.md` and Algorithm 3) that transforms raw seed pairs $(x, y)$ into augmented generator trajectories $\mathcal{D}_{\text{gen}}$
  - Why retrieval for generator training data is performed in two stages, and why the target output $y$ is included in the retrieval query during offline data creation (hindsight retrieval: finding passages that actually support the known ground-truth sentence $y_t$)
  - Line-by-line walkthrough of `create_retrieval_data.py`:
    - Mode 1 (Initial retrieval, `--multiple_sent` not passed): stripping task prefixes on `nq`/`fever`/`wow` and building query `instruction + " " + output` to retrieve top-10 passages for the whole response
    - Mode 2 (Continuous sentence-level retrieval, `--multiple_sent` passed): skipping instances where initial Critic retrieval prediction is `False`, splitting `output` into sentences $y_1, \dots, y_T$ via spaCy `en_core_web_sm`, skipping short sentences (`len(splitted_output) > 2 and len(splitted_output[sent_idx]) < 30`), attaching the top-1 passage from Mode 1 (`id2evidence[q_id]`) as context for 3-way retrieval evaluation, and saving both `MULTI_RETRIEVAL_INPUT` and `MULTI_RETRIEVAL_INPUT_splitted`

- [x] `07_generator_data_creation/02_candidate_pairing_and_vllm_critic_inference.md`
  - Read `reference/code/data_creation/generator/create_prompt_data.py` and `run_reward_vllm.py`
  - Line-by-line walkthrough of `create_prompt_data.py`:
    - Re-attaching `TASK_INST[dataset_name] + "## Input:\n\n"` for task datasets
    - Pairing each sentence `sent_idx` with its top-$N$ (`--top_n`, e.g., 10) retrieved passages (`p_idx = 0..N-1`), skipping `sent_idx > 0` when the Critic's multi-sentence retrieval prediction was `"No Retrieval"`, and splitting into `num_jobs` batch files (`prompt_data_batch_{i}.jsonl`)
  - Line-by-line walkthrough of `run_reward_vllm.py`:
    - Formatting Alpaca-style prompts (`ALPACA_PROMPT_DICT["prompt_input"]`) combining `PROMPT_DICT[inst_mode]` and `PROMPT_DICT[input_mode]` for all five Critic evaluation passes (`retrieval`, `multi_retrieval`, `utility`, `relevance`, `groudness`)
    - Running batched greedy decoding in vLLM (`temperature=0.0, top_p=1.0, max_tokens=15`) and postprocessing special tokens
    - Diagnosing the three code bugs in `run_reward_vllm.py`:
      - Duplicate definition of `process_data` (lines 146-163 and 166-186)
      - Key suffix mismatch (`PROMPT_DICT[input_mode + "_no_preceding"]` on line 177 vs `"_wo_preceding"` defined in `PROMPT_DICT` on lines 57 and 85)
      - Missing `predicted_results.append(copy.deepcopy(item))` in the remainder batch loop (lines 281-299), which drops the final `len(input_data) % batch_size` predictions

- [x] `07_generator_data_creation/03_passage_selection_cascade_and_postprocessing.md`
  - Read `reference/paper/2310.11511.md` (Section 3.2.2, Figure 2, Appendix A.2)
  - Read `reference/code/data_creation/generator/postprocess_data.py`
  - Plain-English intuition: for each sentence $y_t$, we have $K$ candidate passages scored by the Critic for `IsRel` and `IsSup`. How do we choose which passage to insert into the training sequence, and why must the training data include both `[Relevant]` (supported) passages and `[Irrelevant]` passages?
  - Line-by-line walkthrough of `postprocess_data.py`:
    - Normalizing raw Critic outputs into exact special token strings (`convert_score_to_utility_token`, `convert_score_to_retrieval_token`, `convert_score_to_groudness`, `postprocess_relevance_reward_token`)
    - Branch 1 (`need_retrieval_initial_i == "[No Retrieval]"`): forming `[No Retrieval] + output + utility_i_token` and retaining only 40% (`random.random() > 0.6`) so non-retrieval examples do not dominate training
    - Branch 2 (Multi-sentence retrieval assembly):
      - Classifying the $K$ candidate passages at `sent_idx` into four index lists: `fully_supported_p_indices`, `partially_supported_p_indices`, `no_support_p_indices`, and `irrelevant_p_indices`
      - Tracing the passage selection cascade: prioritizing `fully_supported_p_indices[0]` (with 25% random replacement by an irrelevant passage on `nq`), else `partially_supported_p_indices[0]` (with 70% probability), else `no_support_p_indices[0]` (with 25% probability), else a random passage from `irrelevant_p_indices`
      - Tracing the dataset-specific overrides in `postprocess_data.py`: lines 253-254 overriding `p_idx` with a random irrelevant passage on `nq` and `fever` whenever `len(irrelevant_p_indices) > 0`, relabeling `[Relevant]` + `[No support / Contradictory]` to `[Irrelevant]` on `nq`/`fever`/`wow`, and overriding `[Utility:1]` to `[Utility:5]` when `output in ["true", "false"]`
      - Exact string assembly for `[Relevant]` (appending `groundness_token` after `sentence`), `[Irrelevant]` (omitting `groundness_token` after `sentence`), `[Continue to Use Evidence]`, and `[No Retrieval]`, plus final filtering (`output.find("[Continue to Use Evidence]") != 0`)

---

## Module 08: Generator Model Training, Loss Masking, FlashAttention, and ZeRO-3

- [x] `08_generator_training/01_generator_objective_and_paragraph_loss_masking.md`
  - Read `reference/paper/2310.11511.md` (Section 3.2.2 Equation 2, Appendix B.1)
  - Read `reference/code/retrieval_lm/finetune.py` (lines 222-350, 378-455) and `reference/code/data_creation/train_special_tokens.py` (lines 171-210)
  - First-principles derivation of the Generator training objective (Equation 2):
    $$\max_{\mathcal{M}} \mathbb{E}_{(x,y,r) \sim \mathcal{D}_{\text{gen}}} \log p_{\mathcal{M}}(y, r \mid x)$$
    showing how a single cross-entropy loss over $\mathcal{V}_{\text{exp}}$ trains $\mathcal{M}$ to predict task tokens $y_t$ and reflection tokens $r$ jointly
  - Why retrieved passage spans `<paragraph>...</paragraph>` must be masked to `-100` in `labels` while remaining unmasked in `input_ids`: preventing the generator from wasting capacity memorizing raw Wikipedia text while allowing subsequent tokens to attend to the passage
  - Line-by-line walkthrough of `encode_with_prompt_completion_format` in `retrieval_lm/finetune.py` with a concrete token-by-token index table:
    - Comparing prompt masking `labels[:source_len-1] = -100` in `finetune.py` against `label[:source_len] = -100` in `train_special_tokens.py`
    - Tracing the `<paragraph>` (`context_markups[0]`) and `</paragraph>` (`context_markups[1]`) masking loop (`labels[start_idx+1:end_idx] = -100`): why `<paragraph>` and `</paragraph>` themselves remain unmasked (active targets), while all interior tokens `start_idx+1 .. end_idx-1` are set to `-100`
    - Diagnosing the multi-paragraph masking bug in `finetune.py` (lines 255-264) and `train_special_tokens.py` (lines 193-201): because the inner loop `for k, orig_token_2 in enumerate(labels[start_idx:]):` has no `break` when it finds `context_markups[1]`, if an example contains multiple `<paragraph>...</paragraph>` blocks, `end_idx` is overwritten by the *last* `</paragraph>` in the entire sequence, masking out all intermediate sentences and reflection tokens between the first `<paragraph>` and the last `</paragraph>`
  - Comparing tokenizer and embedding setup in `finetune.py` vs `train_special_tokens.py` (`<pad>` vs `[PAD]`, default `resize_token_embeddings` without mean initialization, `encode_with_messages_format`, LoRA configuration, and the `processed.json` dump on lines 433-444)

- [x] `08_generator_training/02_flash_attention_and_rope_monkey_patch.md`
  - Read `reference/paper/2310.11511.md` (Appendix B.1)
  - Read `reference/code/retrieval_lm/llama_flash_attn_monkey_patch.py`
  - Plain-English intuition: why standard self-attention runs out of GPU memory on long sequences ($L = 2048$), and how FlashAttention and padding removal speed up training
  - Step-by-step explanation of Rotary Position Embeddings (RoPE): how rotating query and key vectors by angles proportional to their position index $m$ encodes relative token distance $m - n$ directly into the dot product $\langle q_m, k_n \rangle$
  - How FlashAttention works conceptually: avoiding reading and writing the massive $[B, H, L, L]$ attention matrix to slow GPU High Bandwidth Memory (HBM) by computing exact softmax in small tiles inside fast on-chip SRAM
  - Why unpadding variable-length sequences (`unpad_input`) before FlashAttention saves massive compute: flattening only the real (non-pad) tokens across a batch into a 1D packed array of length $\text{nnz}$ with cumulative sequence boundaries `cu_q_lens`
  - Line-by-line walkthrough of `llama_flash_attn_monkey_patch.py` (`forward`, `_prepare_decoder_attention_mask`, `replace_llama_attn_with_flash_attn`) with exact tensor shapes at every line

- [x] `08_generator_training/03_deepspeed_zero3_and_finetune_execution.md`
  - Read `reference/paper/2310.11511.md` (Section 4.3, Appendix B.1)
  - Read `reference/code/retrieval_lm/stage3_no_offloading_accelerate.conf`, `script_finetune_7b.sh`, `script_finetune_13b.sh`, and `finetune.py` (lines 450-726)
  - Plain-English intuition: why training a 7B or 13B model with AdamW requires $>100\text{ GB}$ of GPU memory (calculating the exact byte counts for 16-bit weights, 16-bit gradients, 32-bit master weights, 32-bit Adam momentum, and 32-bit Adam variance), and how DeepSpeed ZeRO Stage 3 partitions all of them across $N_{\text{GPU}}$ GPUs
  - Line-by-line walkthrough of `stage3_no_offloading_accelerate.conf` (`stage: 3`, `overlap_comm`, `contiguous_gradients`, `stage3_gather_16bit_weights_on_model_save`, `bf16`)
  - Line-by-line walkthrough of `script_finetune_7b.sh` and `script_finetune_13b.sh`: effective batch size math ($4 \text{ GPUs} \times 1 \text{ micro-batch} \times 32 \text{ accumulation steps} = 128$), sequence length limits ($2048$ for 7B vs $1536$ in script / $1524$ in paper for 13B), and learning rate schedule ($2 \times 10^{-5}$, 3% linear warmup, 3 epochs)
  - Line-by-line walkthrough of the training loop in `finetune.py`: AdamW weight decay grouping (`no_decay = ["bias", "layer_norm.weight"]`), why `num_training_steps_for_scheduler` adjusts for `accelerator.num_processes`, gradient accumulation (`with accelerator.accumulate(model):`), checkpoint resuming, and gathering ZeRO-3 sharded weights on save (`accelerator.get_state_dict(model)`)

---

## Module 09: Controllable Inference and Critique Scoring Math

- [x] `09_controllable_inference/01_adaptive_retrieval_and_critique_scoring_math.md`
  - Read `reference/paper/2310.11511.md` (Section 3.1 Algorithm 1, Section 3.3 Equations 3-4, Appendix A.3)
  - Plain-English intuition: how predicting reflection tokens lets us control the model at test time in two ways: (1) turning a knob $\delta$ to control how often the model retrieves, and (2) turning weights $(w_{\text{rel}}, w_{\text{sup}}, w_{\text{use}})$ to steer generation toward strict factual precision or fluent completeness without retraining
  - Step-by-step derivation of the adaptive retrieval threshold condition (Appendix A.3):
    $$\frac{p(\text{[Retrieval]})}{p(\text{[Retrieval]}) + p(\text{[No Retrieval]})} > \delta$$
    comparing hard triggering ($\arg\max = \text{[Retrieval]}$) against soft threshold triggering ($\delta \in [0, 1]$)
  - Step-by-step derivation of the three normalized critique scores $s_t^G$ from vocabulary log-probabilities (Appendix A.3):
    - Relevance score $s_t(\text{IsRel}) \in [0, 1]$ over $\{[\text{Relevant}], [\text{Irrelevant}]\}$
    - Groundedness score $s_t(\text{IsSup}) \in [0, 1]$ with weights $\{1.0, 0.5, 0.0\}$ over $\{[\text{Fully supported}], [\text{Partially supported}], [\text{No support / Contradictory}]\}$
    - Utility score $s_t(\text{IsUse}) \in [-1, 1]$ with weights $\{-1.0, -0.5, 0.0, 0.5, 1.0\}$ over $\{[\text{Utility:1}], \dots, [\text{Utility:5}]\}$
  - Combining critique scores with length-normalized sequence probability (Equations 3-4 vs codebase `final_score = np.exp(seq_score) + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score`)
  - Complete numerical worked example: given concrete log-probabilities for 3 candidate passages, computing every normalized critique score and ranking the candidates step-by-step by hand

- [x] `09_controllable_inference/02_short_form_and_closed_set_inference_code.md`
  - Read `reference/paper/2310.11511.md` (Section 4.1, Section 4.3, Appendix B.2, Table 5)
  - Read `reference/code/retrieval_lm/run_short_form.py` and `reference/code/retrieval_lm/metrics.py`
  - Line-by-line walkthrough of `run_short_form.py`:
    - Preprocessing inputs in `preprocess_input_data` for ARC-Challenge (`arc_c`), PubHealth (`fever`), PopQA, and TriviaQA
    - Step 1 (Initial retrieval decision in `call_model_rerank_w_scores_batch`): running vLLM with `logprobs=32016` and checking `always_retrieve`, `no_retrieval`, or `adaptive_retrieval`
    - Diagnosing the log-probability bug on lines 68-75 of `run_short_form.py`: storing raw negative log-probabilities `score_dict[tok] = float(prob)` without `np.exp(prob)`, and showing with concrete numbers how dividing two negative numbers inverts the threshold inequality
    - Step 2 (Parallel passage generation and critique scoring): building `evidence_augmented_inputs` for top-$K$ (`ndocs`) passages, running vLLM with `logprobs=5000`, reading `relevance_score_dict` at token index $0$, scanning `pred_token_ids` for the first occurrence of `grd_tokens` and `ut_tokens`, and computing `final_score`
    - Step 3 (Open-ended best-passage selection vs closed-set score aggregation):
      - Why `closed=False` (PopQA, TriviaQA) picks the single passage $k^*$ with highest `final_score`
      - Why `closed=True` (`fever`, `arc_c`) strips reflection tokens (`postprocess_answer_option_conditioned`) and sums `score` across all candidate passages that predict the same answer option (`answer2score[answer] += score`), worked out with a 5-passage numerical example
      - Diagnosing the retrieval frequency print bug on line 309 (`count / len(final_results)` dividing by 8 dictionary keys instead of `len(input_data)`)
  - Line-by-line walkthrough of `retrieval_lm/metrics.py`: `normalize_answer`, `exact_match_score`, `qa_f1_score` (token precision, recall, F1 via `Counter`), `match`, `accuracy`, and `find_entity_tags`

---

## Module 10: Long-Form Segment-Level Beam Search and Citation Formatting

- [x] `10_long_form_beam_search/01_segment_level_beam_search_algorithm_and_tree_trace.md`
  - Read `reference/paper/2310.11511.md` (Section 3.1 Algorithm 1, Section 3.3, Section 4.3, Appendix A.3)
  - Plain-English intuition: why generating a multi-sentence biography or long-form answer requires retrieving different passages at different sentences $t = 1, \dots, T$, and how segment-level beam search explores multiple passage trajectories in parallel
  - How segment-level beam search differs from standard token-level beam search: each tree node is a complete sentence segment $y_t$ conditioned on a specific passage $d_k$, and each tree edge is scored by $f(y_t, d_k, \text{Critique})$
  - Step-by-step visual and numerical trace of a beam search tree with beam width $B = 2$, $K = 3$ passages, and max depth $T = 3$:
    - Expanding root node $0$ across $K = 3$ passages to create 3 candidate nodes at depth 1, and pruning to the top $B = 2$ nodes
    - Truncating any multi-sentence generation at the next `[Retrieval]` token so each depth level represents one retrieved segment
    - Expanding each of the $B = 2$ surviving beams across $K = 3$ passages ($2 \times 3 = 6$ candidates) at depth 2, multiplying parent and child scores (`node_score = result["score"] * score`), pruning to the top $B = 2$, and tracing parent pointers from leaf back to root

- [x] `10_long_form_beam_search/02_long_form_static_code_walkthrough_and_bugs.md`
  - Read `reference/code/retrieval_lm/run_long_form_static.py` and `reference/code/retrieval_lm/utils.py` (lines 99-153)
  - Line-by-line walkthrough of `run_step_generation_batch` in `run_long_form_static.py`:
    - Augmenting prompts with `[Retrieval]<paragraph>{title}\n{text}</paragraph>` for each passage in `paragraphs`
    - Computing `relevance_score`, `ground_score`, `utility_score`, and `final_score`
    - Mid-generation `[No Retrieval]` $\to$ `[Retrieval]` remapping (lines 102-135): splitting `pred_text` on `"[No Retrieval]"`, computing `([Retrieval] + [Continue to Use Evidence]) / ([Retrieval] + [No Retrieval]) > threshold` at each occurrence, and rewriting `[No Retrieval]` to `[Retrieval]` when the soft threshold is exceeded
  - Line-by-line walkthrough of `call_model_beam_batch` and `main` in `run_long_form_static.py`:
    - Initial retrieval check (`max_tokens=25, logprobs=32000`, where `np.exp(prob)` is properly applied, but retrieval also requires `"[Retrieval]"` in the greedy text, so the threshold can only suppress retrieval)
    - The "static" design: one fixed list of $K$ pre-retrieved passages reused at every depth, with no per-segment retriever call
    - Tree dictionary structure (`prediction_tree`, `levels`, `best_selections`), segment truncation at `pred.index("[Retrieval]")`, and bottom-up path reconstruction
    - Hard constraint filtering via `--ignore_cont` (dropping any segment containing `"[No support / Contradictory]"`)
    - Output postprocessing for FactScore (`fix_spacing(postprocess(result[0]))`) and ALCE/ASQA inline citation formatting (`postprocessed_result[:-1] + " [{}]".format(idx) + ". "`, deduplicating repeated sentences in `prev_gen`, and replacing `".[Continue to Use Evidence]"` with `" [1]. "`)
  - Detailed diagnosis of the implementation bugs in `run_long_form_static.py`:
    - Bug 1 (Line 68): inside `if ut_tokens is not None:`, iterating over `for token, token_id in grd_tokens.items():` instead of `ut_tokens.items()`, which populates `ut_score_dict` with at most 3 items instead of 5 and forces `utility_score = 0.0` across all long-form runs
    - Bug 2 (Lines 221-224): `call_model_beam_batch` does not pass `use_seqscore=use_seqscore` into `run_step_generation_batch`, leaving `use_seqscore=False` during tree search
    - Bug 3 (Line 128): redundant `if do_retrieve > threshold:` where `do_retrieve` is already a boolean
    - Bug 4 (Line 274): `current_level += 1` incrementing instead of decrementing in the `while current_level > 0 and current_node is not None:` loop (terminating only when `current_node` reaches `parent = None` at root node $0$)
    - Bug 5 (Line 364): `generate()` passing hardcoded `w_rel=1.0, w_sup=1.0, w_use=0.5` instead of `args.w_rel, args.w_sup, args.w_use`
    - Bug 6 (Lines 217-243): `prev_generation` reused across siblings, so stored prompts of non-first children contain the previous sibling's segment
    - Bug 7 (Lines 41-46): a missing relevance log-probability stored as the probability `-100`
    - Also: the FactScore crash when the root does not retrieve (lines 375-380), the `--mode no_retrieval` crash (lines 159-165), misaligned ASQA citation numbers after skipped sentences, and the combined effect of Bugs 1, 2, and 5 on Figure 3b

---

## Module 11: Benchmarks, External Evaluation Metrics, and Main Experimental Results

- [x] `11_evaluation_and_results/01_tasks_datasets_and_external_metrics_explained.md`
  - Read `reference/paper/2310.11511.md` (Section 4.1, Section 4.3, Appendix B.2, Table 5)
  - Plain-English, first-principles explanation of all six evaluation tasks, their dataset splits, retrieval corpora, and evaluation metrics:
    - Closed-set tasks: PubHealth (public health fact verification, `"true"`/`"false"`, accuracy) and ARC-Challenge (scientific multiple-choice exams, `"A"`..`"D"`, accuracy)
    - Short-form open-domain QA: PopQA long-tail subset (1,399 rare entity queries with $<100$ monthly Wikipedia views, December 2020 Wikipedia + Google Programmable Search intro paragraphs, substring match accuracy) and TriviaQA-unfiltered (11,313 open test queries, substring match accuracy)
    - Long-form biography generation evaluated with FactScore: how FactScore splits a generated biography into atomic factual statements and uses retrieval + ChatGPT/InstructGPT to compute the fraction of atomic statements supported by the entity's Wikipedia article
    - Long-form ambiguous QA on ALCE-ASQA (using author-provided top-5 GTR-XXL passages):
      - Correctness (`str-em` and `ROUGE-L`): checking whether gold short answers for all disambiguated interpretations appear in the long-form response
      - Fluency (`MAUVE`): how MAUVE embeds model and human responses, quantizes the embedding space into clusters, and computes the area under the KL-divergence frontier curve
      - Citation Precision (`pre`) and Citation Recall (`rec`): how ALCE uses a T5-XXL (11B) Natural Language Inference (NLI) model to test whether cited passages entail each sentence (recall) and whether every individual citation `[k]` is necessary and relevant (precision)

- [x] `11_evaluation_and_results/02_baseline_taxonomy_and_main_results_analysis.md`
  - Read `reference/paper/2310.11511.md` (Section 4.2, Section 5.1, Table 2)
  - Complete guide to all 18 models compared in Table 2:
    - LMs trained with proprietary data: Llama2-chat 13B, Ret-Llama2-chat 13B, ChatGPT, Ret-ChatGPT, Perplexity.ai
    - Open-weights baselines without retrieval: Llama2 7B/13B, Alpaca 7B/13B, CoVe 65B
    - Open-weights baselines with retrieval: Toolformer 6B, Ret-Llama2 7B/13B, Ret-Alpaca 7B/13B, SAIL 7B, and the critical data-controlled baseline `Llama2-FT 7B` (fine-tuned on the exact same 150k instruction-output pairs as Self-RAG without reflection tokens or retrieved passages, and augmented with retrieval at test time)
  - Comprehensive quantitative analysis of Table 2 across all six tasks:
    - Isolating the architectural gain of Self-RAG: comparing Self-RAG 7B vs `Llama2-FT 7B` (PopQA $54.9$ vs $48.7$, TriviaQA $66.4$ vs $57.3$, PubHealth $72.4$ vs $64.3$, ARC-C $67.3$ vs $65.8$, Bio FactScore $81.2$ vs $78.2$, ASQA MAUVE $74.3$ vs $51.2$, ASQA Citation Precision $66.9\%$ vs $5.0\%$, Citation Recall $67.8\%$ vs $7.5\%$)
    - Why standard RAG baselines fail to improve over no-retrieval on reasoning/verification (PubHealth and ARC-C) where answers cannot be copied as literal substrings from passages
    - Why Self-RAG 7B outperforms Self-RAG 13B on precision-oriented FactScore ($81.2$ vs $80.2$) and ASQA MAUVE ($74.3$ vs $71.6$), while 13B outperforms 7B on PopQA, TriviaQA, PubHealth, ARC-C, and ASQA correctness and citation accuracy

---

## Module 12: Ablations, Inference Customization, Scaling Laws, and Qualitative Analysis

- [x] `12_ablations_and_analysis/01_training_and_inference_ablations_and_customization.md`
  - Read `reference/paper/2310.11511.md` (Section 5.2, Figure 3, Figure 4)
  - Read `reference/paper/images/_page_8_Diagram_2.jpeg`, `_page_8_Figure_1.jpeg`, and `_page_9_Figure_1.jpeg`
  - Detailed walkthrough of the training and inference ablation study on the 50k training subset (Table 3a / Figure 3a):
    - Training ablations: full `Self-RAG (50k)` vs `No Retriever` (instruction-tuned without retrieved passages or reflection tokens) vs `No Critic` (SAIL-style training always prepending top-1 passage without reflection tokens, which causes ASQA `em` to collapse from $32.1$ to $18.1$)
    - Inference ablations: `No retrieval` (PopQA drops from $45.5$ to $24.7$), `Hard constraints` (retrieving only when $\arg\max = \text{[Retrieval]}$ instead of soft threshold $\delta$; PopQA drops to $28.3$), `Retrieve top 1` (conventional RAG using only the top-1 passage without parallel critique reranking; PopQA drops to $41.8$, ASQA to $28.6$), and `Remove IsSup` (ablating groundedness score during beam search; PopQA drops to $44.1$, ASQA to $30.6$)
  - Test-time customization and efficiency trade-offs:
    - Varying the `IsSup` critique weight $w_{\text{sup}}$ from about $0.1$ to $2$ on ASQA (Figure 3b): why increasing $w_{\text{sup}}$ monotonically raises citation precision (from $<70.0$ to $>70.5$) while reducing MAUVE fluency (from $>95$ to $\sim 86$), and why these analysis numbers do not match Table 2
    - Varying the adaptive retrieval threshold $\delta \in [0, 0.7]$ on PubHealth and PopQA (Figure 3c): how $\delta$ controls retrieval frequency and why PopQA accuracy is highly sensitive to retrieval frequency while PubHealth accuracy degrades by less than $2\%$
    - Training data scaling trajectories across $5\text{k}, 10\text{k}, 20\text{k}, 50\text{k}, 150\text{k}$ instances (Figures 4a-4c) on PopQA, PubHealth, and ASQA citation precision

- [x] `12_ablations_and_analysis/02_memory_reliance_human_eval_and_qualitative_cases.md`
  - Read `reference/paper/2310.11511.md` (Section 5.2 Human evaluations, Appendix C.1, C.2, C.3, Tables 6-7)
  - Read `reference/paper/images/_page_9_Figure_1.jpeg`
  - Parametric vs non-parametric memory reliance analysis (Appendix C.1): how the authors test whether correct answers on TriviaQA and PopQA appear as substrings of the retrieved passages, and why only $2\%$ of Self-RAG 7B's correct answers come from unverified parametric memory compared to $15\%$ for "Alpaca" (size not stated), $18\%$ for Llama2-chat 13B, and $20\%$ for the model the paper calls "Alpaca 30B" (not among the models it lists as evaluated)
  - Human evaluation study (Section 5.2, Figure 4d, Appendix C.2 Table 6):
    - Evaluating Supported and Plausible (`S&P`) rates ($92.5\%$ on PopQA, $70.0\%$ on Bio) and human agreement with model-predicted `IsRel` ($95.0\%$ PopQA, $90.0\%$ Bio) and `IsSup` ($90.0\%$ PopQA, $85.0\%$ Bio) tokens
    - Walkthrough of all four human evaluation examples in Appendix Table 6 (astronomy book genre, Sam Harris book title mismatch *Lying* vs *The Lie*, Ronaldinho bio, and Sergei Bodrov bio with unsupported `[Continue to Use Evidence]` sentences)
  - Sentence-by-sentence walkthrough of the qualitative case studies in Appendix Table 7:
    - Multi-document citation synthesis on ASQA (`When did sunday became the seventh day of the week` citing Emperor Constantine across Reference `[1]` and Reference `[2]`)
    - Detecting factual contradiction (`Tell me a bio about Rory Tapner`, where the model flags its own claim `"since 2010"` with `[ISSUP=Contradictory]` because Reference `[1]` states he stepped down in February 2015)
    - Detecting partial support (`Tell me a bio about G. Venugopal`, where the model flags `"in Malayalam, Tamil, and Hindi movies"` with `[ISSUP=Partially Supported]` because Reference `[1]` only mentions Malayalam films)

---

## Reference Appendices

- [x] `paper_audit.md`
  - Read `reference/paper/2310.11511.md` and all five figure images; check uncertain passages against the arXiv v1 LaTeX source
  - Markdown conversion artifacts, typos and editing errors, internal inconsistencies, claims stronger than the reported numbers, errors in showcased examples, missing details and limitations, and a summary of paper vs released code

- [x] `codebase_audit.md`
  - Read the scripts cited by the part files in `reference/code/`, plus the main and generator READMEs
  - Merged, re-verified replacement for the earlier `CODEBASE_AUDIT.md` and `study/codebase_audit_and_bugs.md`: crashes, inference that differs from the paper, training-data construction issues, settings that differ from the paper, minor issues, corrections to the earlier audits, and impact on the paper's claims
