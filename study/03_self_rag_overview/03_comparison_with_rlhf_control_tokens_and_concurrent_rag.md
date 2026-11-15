# Comparison with RLHF, Control Tokens, and Concurrent RAG

## 1. Plain-English Intuition: How Does Self-RAG Compare to Other Ways of Controlling Language Models?

In Part 1 and Part 2, we saw how Self-RAG trains a single language model $\mathcal{M}$ to retrieve passages on demand, process them in parallel, and evaluate its own sentences using reflection tokens.

To appreciate why Self-RAG is designed this way, it helps to compare it against four other approaches that researchers use to make language models more factual or controllable (discussed in Section 2 and Section 3.2.2 of the paper):
1. Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy Optimization (PPO).
2. Prefix control tokens (such as CTRL and QUARK).
3. RAG systems with fixed retrieval, heuristic triggers, or external filtering models (SAIL, Toolformer, Active RAG, Recomp).
4. Iterative natural-language self-refinement prompts (Chain-of-Verification / CoVe, Self-Refine) and LM-guided tree search with prompted value scores (LATS).

Walking through how each of these alternatives works from first principles shows what bottlenecks they hit and how Self-RAG avoids them.

---

## 2. Step-by-Step Comparison: Self-RAG vs RLHF / PPO

### 2.1 How Multi-Reward RLHF with PPO Works

Suppose we want to align a language model (called the actor policy $\pi_\theta$) so that its outputs satisfy multiple quality criteria, such as passage relevance, factual support, and overall helpfulness.

In fine-grained Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO), training requires setting up multiple separate neural networks at the same time:
1. $G$ separate trained reward models $R_1, R_2, \dots, R_G$, where each reward model $R_g(x, y) \in \mathbb{R}$ outputs a scalar score for one aspect of quality (for example, $R_1$ scores relevance, $R_2$ scores factuality, and $R_3$ scores completeness).
2. The trainable actor policy $\pi_\theta(y \mid x)$, which generates responses during training.
3. A frozen copy of the initial model, called the reference policy $\pi_{\text{ref}}(y \mid x)$.
4. A trainable value critic network $V_\phi(x, y_{<j})$, which estimates expected future rewards at each token step to reduce gradient variance.

At every single training step of PPO:
- First, given a batch of prompts $x$, the actor $\pi_\theta$ must generate complete responses $y \sim \pi_\theta(\cdot \mid x)$ token by token (called an online rollout).
- Second, the generated responses $(x, y)$ are passed through all $G$ reward models, and their scores are combined using fixed scalar weights $w_1, \dots, w_G \ge 0$:

$$R_{\text{total}}(x, y) = \sum_{g=1}^{G} w_g R_g(x, y)$$

- Third, to stop the actor $\pi_\theta$ from exploiting quirks in the reward models by outputting unnatural text (reward hacking), PPO subtracts a Kullback-Leibler (KL) divergence penalty weighted by $\beta > 0$ whenever $\pi_\theta$ drifts away from the frozen reference model $\pi_{\text{ref}}$. The overall optimization objective is:

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{g=1}^{G} w_g R_g(x, y) - \beta \sum_{j=1}^{n} \log \frac{\pi_\theta(y_j \mid x, y_{<j})}{\pi_{\text{ref}}(y_j \mid x, y_{<j})} \right]$$

where:
- $\pi_\theta(y \mid x)$ is the active language model being trained.
- $\pi_{\text{ref}}(y \mid x)$ is the frozen initial supervised fine-tuned model.
- $R_g(x, y) \in \mathbb{R}$ is the scalar reward from the $g$-th reward model.
- $w_g \ge 0$ is the fixed weight assigned to the $g$-th reward aspect during training.
- $\beta > 0$ is the KL penalty coefficient keeping $\pi_\theta$ close to $\pi_{\text{ref}}$.

### 2.2 Three Advantages of Self-RAG Over Online RLHF / PPO

Comparing the PPO setup above with Self-RAG's offline training pipeline from Part 2 reveals three major differences:

1. Training memory and compute cost:
   Online PPO must keep at least $3 + G$ large language models in GPU memory simultaneously ($\pi_\theta$, $\pi_{\text{ref}}$, $V_\phi$, and $R_1 \dots R_G$), and every training step is bottlenecked by slow token-by-token generation rollouts $y \sim \pi_\theta(\cdot \mid x)$. Self-RAG runs its Critic $\mathcal{C}$ once offline to insert reflection tokens into the dataset $\mathcal{D}_{\text{gen}}$, and then trains only one model $\mathcal{M}$ using fast, parallel teacher-forcing cross-entropy loss (Module 01 Part 3).
2. Optimization stability:
   Policy gradient methods like PPO suffer from high gradient variance and sensitivity to hyperparameters ($\beta$, reward normalization, value loss clipping). Self-RAG trains both $\mathcal{C}$ and $\mathcal{M}$ with standard maximum likelihood cross-entropy loss (Equations 1 and 2).
3. Test-time customization without retraining:
   In multi-reward RLHF, the preference weights $(w_1, \dots, w_G)$ must be chosen before training begins, and the resulting trade-off is baked permanently into the trained weights $\theta$. If a user wants to prioritize fluency on one task and strict factual support on another task, an RLHF model must be retrained from scratch with new weights $w_g$. Because Self-RAG predicts the probability of each critique token (`IsRel`, `IsSup`, `IsUse`) explicitly at test time, any user can adjust the critique weights $(w_{\text{rel}}, w_{\text{sup}}, w_{\text{use}})$ and retrieval threshold $\delta$ at inference time with zero additional training.

---

## 3. Step-by-Step Comparison: Self-RAG vs Prefix Control Tokens (CTRL and QUARK)

### 3.1 How Prefix Control Tokens Work

Controllable generation methods like CTRL and QUARK also add special control tokens to the model vocabulary, but they place a single control token $c \in \mathcal{V}_{\text{ctrl}}$ at the very beginning of the input prompt $x$:

$$p_\theta(y \mid c, x) = \prod_{j=1}^{n} p_\theta(y_j \mid c, x, y_{<j})$$

In QUARK, for example, training responses are scored by a reward model, bucketed into reward quantiles (from lowest reward $c_1$ to highest reward $c_{\max}$), and prepended with their quantile token $c_k$ during training. At inference time, the user prepends the highest-reward token $c_{\max}$ before the prompt $x$ to steer the model toward generating a high-reward response.

### 3.2 Why Prefix Tokens Are Not Enough for Retrieval and Verification

Compare where the special tokens sit in prefix-control methods versus Self-RAG:
- Prefix control ($c \to x \to y$): The control token $c$ is fed as an input before the model generates a single word of $y$. The model never outputs $c$ to evaluate its own work.
- Self-RAG ($x \to \text{Retrieve} \to d_k \to \text{IsRel} \to y_t^{(k)} \to \text{IsSup} \to \text{IsUse}$): The reflection tokens are generated by the model itself during and after each sentence segment.

Because a prefix token $c$ is fixed at the start of the prompt, it cannot:
1. Decide mid-generation that sentence 1 needed no retrieval while sentence 2 requires a fresh Wikipedia lookup.
2. Inspect $K = 5$ retrieved passages in parallel and flag Passage 1 as `[Relevant]` and Passage 2 as `[Irrelevant]`.
3. Read a generated candidate sentence $y_t^{(k)}$ after it is written and verify whether $y_t^{(k)}$ is `[Fully supported]`, `[Partially supported]`, or `[No support / Contradictory]` with respect to passage $d_k$.

---

## 4. Comparison with Concurrent RAG and Iterative Refinement Methods

Section 2 of the paper compares Self-RAG with several concurrent retrieval-augmented and self-refinement systems:

| Method Family | Representative Models | How It Works | Key Limitation Solved by Self-RAG |
|---|---|---|---|
| Fixed Instruction RAG | SAIL, RA-DIT | Fine-tunes an LM (and retriever in RA-DIT) on instruction datasets with top retrieved documents always prepended to the prompt. | Retrieves indiscriminately on every query; does not critique passage relevance or verify sentence-level support. |
| Heuristic / API Active RAG | Toolformer, Active RAG (FLARE) | Toolformer trains an LM to generate API calls for named entities; Active RAG adaptively retrieves passages during generation on top of a proprietary LLM. | Relies on heuristics or proprietary LLMs; lacks learned reflection tokens to filter irrelevant passages or verify post-generation entailment. |
| External Passage Filters / Compressors | NLI-Filtered RAG, Recomp | Runs a separate external NLI classifier or summarization model to filter or compress retrieved passages before prompting the generator LM. | Requires hosting an extra external model at inference time, and only filters input passages $d$ without checking whether the generated output $y_t$ is supported by $d$. |
| Iterative Prompt Refinement & Search | CoVe (65B), Self-Refine, Refiner, LATS | Prompts an off-the-shelf LLM across multiple sequential turns to draft an answer, write natural-language critique/verification questions, and rewrite the answer. | Very high inference latency (generating hundreds of natural-language feedback tokens sequentially). Self-RAG compresses each critique into a single atomic token (`[Relevant]`, `[Fully supported]`). |

---

## 5. Complete Codebase Map and Runtime Environment (`reference/code/`)

### 5.1 Mapping Every Directory and Script to Our Study Plan

The `reference/code/` repository is organized around the exact pipeline stages we have introduced. The table below maps each directory and file to its role and the module where we walk through its code line by line:

| Directory / File Path | Role in the Self-RAG Pipeline | Covered in Study Module |
|---|---|---|
| `retrieval_lm/utils.py`, `retrieval_lm/run_baseline_lm.py` | Prompt templates, special token ID loaders, and baseline evaluation script | Module 02 & Module 04 |
| `data_creation/process_data/` (`task_instructions.py`, `tulu.py`, `kilt_dpr.py`, `arc.py`, `obqa.py`, `asqa.py`) | Preprocess the 11 raw seed datasets (145,619 examples) into a unified JSONL format | Module 04 |
| `retrieval_lm/src/` (`contriever.py`, `inbatch.py`, `moco.py`, `dist_utils.py`, `data.py`, `finetuning_data.py`, `normalize_text.py`, `options.py`, `slurm.py`, `utils.py`, `index.py`, `evaluation.py`, `beir_utils.py`) | Contriever bi-encoder, contrastive pre-training, FAISS vector indexing, and retrieval evaluation | Module 05 |
| `retrieval_lm/generate_passage_embeddings.py`, `passage_retrieval.py`, `download_demo_corpus.sh` | Encode Wikipedia passages into dense embeddings and run top-$K$ FAISS passage search | Module 05 |
| `data_creation/critic/gpt4_reward/` (`chatgpt_need_retrieval.py`, `chatgpt_relevance.py`, `chatgpt_groundness.py`, `chatgpt_utility.py`, `chatgpt_inference.py`, `combine_chat_gpt_reward.py`) | Stage 1: Prompt GPT-4 to collect reflection labels and assemble the Critic training/dev splits | Module 06 |
| `data_creation/train_special_tokens.py` | Stage 2: Expand vocabulary, initialize new token embeddings to the mean, and train the Critic $\mathcal{C}$ | Module 06 |
| `data_creation/generator/` (`create_retrieval_data.py`, `create_prompt_data.py`, `run_reward_vllm.py`, `postprocess_data.py`) | Stage 3: Split sentences with spaCy, run Retriever $\mathcal{R}$ and Critic $\mathcal{C}$ offline, select passages, and build $\mathcal{D}_{\text{gen}}$ | Module 07 |
| `retrieval_lm/finetune.py`, `llama_flash_attn_monkey_patch.py`, `script_finetune_7b.sh`, `script_finetune_13b.sh`, `stage3_no_offloading_accelerate.conf` | Stage 4: Train Generator $\mathcal{M}$ (7B and 13B) with `<paragraph>` loss masking, FlashAttention, and DeepSpeed ZeRO-3 | Module 08 |
| `retrieval_lm/run_short_form.py`, `retrieval_lm/metrics.py` | Short-form and closed-set Self-RAG inference and evaluation metrics | Module 09 |
| `retrieval_lm/run_long_form_static.py` | Long-form segment-level beam search tree decoding and ALCE citation formatting | Module 10 |

### 5.2 Key Software Dependencies in `setup.sh`, `requirements.txt`, and `environment.yml`

Inspecting `reference/code/setup.sh`, `reference/code/requirements.txt`, and `reference/code/environment.yml` highlights five technical details of the runtime environment:
1. `vllm==0.2.6` and `skip_special_tokens=False`:
   Self-RAG uses vLLM for fast batched generation and log-probability extraction, and `requirements.txt` pins `vllm==0.2.6`. The Installation section of `reference/code/README.md` asks for a recent vLLM because older versions may not let you set `skip_special_tokens` through `SamplingParams`. The Quick start example passes `SamplingParams(..., skip_special_tokens=False)` so that `[Retrieval]`, `[No Retrieval]`, `[Relevant]`, `[Fully supported]`, and the other reflection tokens stay in `pred.outputs[0].text`; with vLLM's default (`skip_special_tokens=True`) they are removed from the decoded text. The evaluation and data-creation scripts do not pass this flag even though parts of their logic search the decoded text for reflection tokens (see `study/codebase_audit.md`).
2. `flash-attn==2.3.6` on CUDA 12.1 (`cuda-nvcc=12.1.66`, `torch==2.1.2`):
   Inserting retrieved Wikipedia passages into training examples produces sequences up to $2048$ tokens long. `setup.sh` installs `flash-attn==2.3.6` so `llama_flash_attn_monkey_patch.py` can replace standard quadratic-memory attention with fast FlashAttention kernels.
3. `deepspeed==0.12.6` and `accelerate==0.25.0`:
   Used by `script_finetune_7b.sh` and `script_finetune_13b.sh` to shard model parameters, gradients, and AdamW optimizer states across 4-8 A100 GPUs via DeepSpeed ZeRO Stage 3.
4. `faiss-gpu` and `spacy==3.7.2`:
   `setup.sh` installs `faiss-gpu` via `conda install -c conda-forge faiss-gpu` for fast inner-product search over 21M passage embeddings, while `spacy` (`en_core_web_sm`) segments training outputs into individual sentences $y_1, \dots, y_T$.
5. Isolated `factscore==0.1.5` environment:
   At the bottom of `requirements.txt` (lines 30-33), `# factscore==0.1.5` is commented out with a note to install it in a separate conda environment because its pinned dependencies conflict with `transformers==4.36.2` and `vllm==0.2.6`.
