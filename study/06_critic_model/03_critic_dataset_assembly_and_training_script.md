# Critic Dataset Assembly and Training Script

## 1. Plain-English Intuition: Turning Raw GPT-4 Outputs into a Trained Critic Model ($\mathcal{C}$)

In Part 1 and Part 2, we saw how the four `chatgpt_*.py` scripts query GPT-4 and save JSON files containing `"input"`, `"score"` (or `"decision_token"`), and `"raw_output"`.

To turn those raw JSON files into a working Llama-2 7B Critic model $\mathcal{C}$, the repository executes two scripts:
1. Step 1 (`reference/code/data_creation/critic/gpt4_reward/combine_chat_gpt_reward.py`):
   Cleans and merges the GPT-4 outputs from all reflection tasks into a single uniform instruction-tuning dataset $\mathcal{D}_{\text{critic}}$, recovers the bracketed labels (`[Fully supported]`, `[Relevant]`) that were dropped by Bug 2 in Part 2, downsamples `[Relevant]` examples by $30\%$ so the Critic does not become biased toward predicting `[Relevant]`, and splits the data into a training set (`_train.json`) and a $1,500$-example validation set (`_dev.json`).
2. Step 2 (`reference/code/data_creation/train_special_tokens.py`):
   Loads `meta-llama/Llama-2-7b-hf`, adds the $15$ Self-RAG special tokens plus `[PAD]`, initializes the $16$ new embedding rows to the mean vector of the $32,000$ existing embeddings, masks all prompt tokens with `-100`, and trains $\mathcal{C}$ for $3$ epochs using PyTorch FSDP (`full_shard auto_wrap`).

---

## 2. Line-by-Line Walkthrough of `combine_chat_gpt_reward.py`

### 2.1 Converting Few-Shot GPT-4 Prompts into Zero-Shot Critic Prompts (`PROMPT_DICT`, Lines 7-86)

When we prompted GPT-4 in Part 1, we had to include long few-shot demonstrations (`##\nInstruction: ... Explanation: ...`) in the prompt because GPT-4 was not fine-tuned on our custom token names.

Because the Critic $\mathcal{C}$ is directly fine-tuned on tens of thousands of examples, it does not need few-shot demonstrations in its input prompt! Instead, `PROMPT_DICT` in `combine_chat_gpt_reward.py` (lines 7-86) splits each task into a concise rubric `"instruction"` and a formatted instance `"input"`:

| Reflection Task | `"instruction"` Key in `PROMPT_DICT` | `"input"` Key in `PROMPT_DICT` | Target `"output"` Token $r$ |
|---|---|---|---|
| Initial `Retrieve` ($t=1$) | `"retrieval_instruction"` | `"retrieval_input"` (`Task instruction: {instruction}`) | `[Retrieval]`, `[No Retrieval]` |
| Multi-sentence `Retrieve` ($t>1$) | `"multi_retrieval_three_way_instruction"` | `"multi_retrieval_three_way_input"` | `[Continue to Use Evidence]`, `[No Retrieval]`, `[Retrieval]` |
| Passage Relevance (`IsRel`) | `"relevance_instruction"` | `"relevance_input"` (`Task instruction: {instruction}\nEvidence: {evidence}`) | `[Relevant]`, `[Irrelevant]` |
| Factual Support (`IsSup`) | `"ground_multi_instruction"` | `"ground_multi_input"` or `"ground_multi_input_wo_preceding"` | `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]` |
| Perceived Utility (`IsUse`) | `"utility_instruction"` | `"utility_input"` (`Task instruction: {instruction}\nOutput: {output}`) | `[Utility:1]` .. `[Utility:5]` |

Notice a minor copy-paste oversight on lines 57-67 of `combine_chat_gpt_reward.py`: `"multi_retrieval_three_way_input"` and `"multi_retrieval_three_way_input_wo_preceding"` are defined as identical strings (both include `"Preceding sentences: {preceding_sentences}\n"`), whereas `"ground_multi_input_wo_preceding"` (lines 31-35) properly omits `"Preceding sentences:"`.

### 2.2 Recovering Bracketed Labels and Balancing Classes (Lines 88-196)

Let us inspect the five task-formatting functions in `combine_chat_gpt_reward.py`:

1. `create_utility_data(input_data)` (lines 89-104):
   ```python
   def create_utility_data(input_data):
       processed_data = []
       for item in input_data:
           input = item["input"]
           raw_output = item["raw_output"]
           if item["score"] == "":
               item["score"] = raw_output.split("\n")[0]
           output = item["score"]
           if output not in [1,2,3,4,5] or len(str(output)) == 0:
               continue
           label = "[Utility:{}]".format(output)
           processed_data.append({"instruction": PROMPT_DICT["utility_instruction"], "input": PROMPT_DICT["utility_input"].format_map(input), "output": label, "task": "utility"})
   ```
   Converts integer utility scores `1..5` into the special token strings `"[Utility:1]"` through `"[Utility:5]"`.

2. `create_retrieval_data_input_only(input_data)` (lines 130-149) and `create_retrieval_data(input_data)` (lines 107-127):
   - For input-only initial retrieval (`create_retrieval_data_input_only`), GPT-4 answered `"[Yes]"` or `"[No]"`. Lines 138-143 map `"Yes" in output` to `"[Retrieval]"` and `"No" in output` to `"[No Retrieval]"`.
   - For multi-sentence retrieval (`create_retrieval_data`), GPT-4's output was already parsed into `item["decision_token"]`, and line 115 keeps only valid 3-way tokens `["[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"]`.

3. `create_groundness_data(input_data, multi_retrieval=True)` (lines 152-174):
   ```python
   def create_groundness_data(input_data, multi_retrieval=False):
       processed_data = []
       for item in input_data:
           input = item["input"]
           raw_output = item["raw_output"]
           if item["score"] == "":
               item["score"] = raw_output.split("\n")[0]
           if len(item["score"]) > 0 and item["score"][-1] == " ":
               item["score"] = item["score"][:-1]
           if len(item["score"]) == 0 or item["score"] not in ["[No support / Contradictory]", "[Fully supported]", "[Partially supported]"]:
               continue
   ```
   Look at lines 158-161: in Part 2 (Bug 2), we saw that `chatgpt_groundness.py` saved `item["score"] = ""` whenever GPT-4 output a bracketed token instead of a digit `1..5`. Here, `if item["score"] == "": item["score"] = raw_output.split("\n")[0]` grabs the first line of `"raw_output"` (before `"\nExplanation:"`), strips any trailing space, and recovers `"[Fully supported]"`, `"[Partially supported]"`, or `"[No support / Contradictory]"`.

4. `create_relevance_data(input_data)` (lines 177-196):
   ```python
   def create_relevance_data(input_data):
       processed_data = []
       for item in input_data:
           input = item["input"]
           raw_output = item["raw_output"]
           if item["score"] == "":
               item["score"] = raw_output.split("\n")[0]
           if len(item["score"]) > 0 and item["score"][-1] == " ":
               item["score"] = item["score"][:-1]
           if item["score"] not in ["[Relevant]", "[Irrelevant]"]:
               continue
           label = item["score"]
           if label == "[Relevant]" and random.random() > 0.7:
               continue
           processed_data.append({"instruction": PROMPT_DICT["relevance_instruction"], "input": PROMPT_DICT["relevance_input"].format_map(input), "output": label, "task": "relevance"})
   ```
   Besides recovering `item["score"]` from `raw_output.split("\n")[0]`, look at lines 190-191:
   `if label == "[Relevant]" and random.random() > 0.7: continue`
   Why does the script randomly discard $30\%$ of `[Relevant]` examples (keeping $70\%$)?
   Because the passages evaluated by GPT-4 were top-$K$ results retrieved by Contriever, `[Relevant]` passages outnumber `[Irrelevant]` passages in the raw data. Downsampling `[Relevant]` by $30\%$ balances the class distribution so the Critic $\mathcal{C}$ learns a well-calibrated boundary between `[Relevant]` and `[Irrelevant]`.

### 2.3 Shuffling and Splitting into `_train.json` and `_dev.json` (`main`, Lines 236-254)

In `main()` (lines 236-254), the five lists are concatenated into `final_train_data`, any item containing `"### Response:"` inside its `"input"` or `"instruction"` is filtered out so it cannot confuse the prompt delimiter, and the shuffled list is split into $1,500$ validation examples and the remaining training examples:

```python
    final_train_data = [item for item in final_train_data if "### Response:" not in item["input"] and "### Response:" not in item["instruction"]]
    random.shuffle(final_train_data)
    train_data = final_train_data[1500:]
    dev_data = final_train_data[:1500]
```

---

## 3. Mathematical Build-Up: Critic Objective and Mean Embedding Initialization (`train_special_tokens.py`)

### 3.1 How `train_special_tokens.py` Implements Equation 1

In `reference/code/data_creation/train_special_tokens.py` (lines 32-39 and 219-224), each example in `_train.json` is formatted into a source string `s` and a target string `t`:
- Source `s`: `"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"`
- Target `t`: `"{output}</s>"` (where `{output}` is the single reflection token $r$, such as `"[Relevant]"`).

As we derived in Module 01 Part 3, `preprocess` (line 178) sets `label[:source_len] = IGNORE_INDEX` (`-100`), masking out the entire source string `s`. Consequently, the cross-entropy loss is evaluated strictly on predicting the reflection token $r$ (and `</s>`) given the conditioning context $(x, y)$ in `s`, which is Equation 1 of the paper:

$$\max_{\mathcal{C}} \mathbb{E}_{((x, y), r) \sim \mathcal{D}_{\text{critic}}} \log p_{\mathcal{C}}(r \mid x, y)$$

### 3.2 Why `smart_tokenizer_and_embedding_resize` Initializes New Tokens to the Mean Embedding

Look at `smart_tokenizer_and_embedding_resize` in `reference/code/data_creation/train_special_tokens.py` (lines 142-162):

```python
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print("# of new special tokens: {}".format(num_new_tokens))
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
```

Let us understand the math and intuition behind lines 155-162.

When `num_new_tokens` ($M = 16$) new tokens are added to a vocabulary of size $V_{\text{old}} = 32000$, `input_embeddings[:-num_new_tokens]` slices the original $32000$ pre-trained rows (indices $0 \dots 31999$), and `.mean(dim=0, keepdim=True)` computes their average row vector $[1, d]$:

$$\mathbf{e}_{\text{avg}}^{\text{in}} = \frac{1}{V_{\text{old}}} \sum_{v=0}^{V_{\text{old}}-1} E_{\text{in}}[v, :] \in \mathbb{R}^d, \quad \mathbf{w}_{\text{avg}}^{\text{out}} = \frac{1}{V_{\text{old}}} \sum_{v=0}^{V_{\text{old}}-1} W_{\text{out}}[v, :] \in \mathbb{R}^d$$

Then `input_embeddings[-num_new_tokens:] = input_embeddings_avg` copies that exact average vector into all $M = 16$ newly added rows (indices $32000 \dots 32015$).

Why is initializing new special tokens to the average of existing tokens much better than random initialization?
1. Balanced initial logit scale:
   At the start of training, given any final hidden vector $h_t \in \mathbb{R}^d$, the logit assigned to a newly added token $v_{\text{new}} \in \{32000, \dots, 32015\}$ is the dot product between $\mathbf{w}_{\text{avg}}^{\text{out}}$ and $h_t$. By the linearity of the dot product:

   $$z_{j, v_{\text{new}}} = \left\langle \mathbf{w}_{\text{avg}}^{\text{out}}, h_t \right\rangle = \left\langle \frac{1}{V_{\text{old}}} \sum_{v=0}^{V_{\text{old}}-1} W_{\text{out}}[v, :],\; h_t \right\rangle = \frac{1}{V_{\text{old}}} \sum_{v=0}^{V_{\text{old}}-1} \langle W_{\text{out}}[v, :], h_t \rangle = \frac{1}{V_{\text{old}}} \sum_{v=0}^{V_{\text{old}}-1} z_{j, v}$$

   The initial logit of every new special token is guaranteed to equal the exact average logit of all $32000$ existing words! It can never start out wildly positive or wildly negative, preventing large gradient spikes on step $1$.
2. Unbiased initial probabilities within each reflection group:
   Because `[Relevant]` ($32004$) and `[Irrelevant]` ($32003$) both start with the exact same output vector $\mathbf{w}_{\text{avg}}^{\text{out}}$, at step $0$ of training their logits are identical ($z_{j, 32004} = z_{j, 32003}$), so the model starts with an unbiased $50\% / 50\%$ relative prior between `[Relevant]` and `[Irrelevant]` before learning from the data.

### 3.3 Two Startup Bugs in `train_special_tokens.py` (Lines 27 and 236)

First, line 27 imports the FlashAttention patch with a relative import, `from ..retrieval_lm.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn`. The README runs the file as a script (`cd data_creation` followed by `torchrun ... train_special_tokens.py`), and a script run this way has no parent package, so Python raises `ImportError: attempted relative import with no known parent package` before training starts.

Second, inspecting `SupervisedDataset.__init__` in `reference/code/data_creation/train_special_tokens.py` (line 236) reveals another bug:

```python
prompt_input, prompt_no_input, prompt_no_input_paragraph, prompt_no_input_separated = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["prompt_no_input_paragraph"], PROMPT_DICT["prompt_no_input_separated"]
```

On lines 34-41 of `train_special_tokens.py`, `PROMPT_DICT` only defines `"prompt_input"` and `"prompt_no_input"`. The two extra variables `prompt_no_input_paragraph` and `prompt_no_input_separated` are never used in `SupervisedDataset.__init__` (the following lines only use `prompt_input` and `prompt_no_input`), so looking up those two missing keys in `PROMPT_DICT` raises a `KeyError` unless they are removed from line 236.

---

## 4. FSDP Training Configuration and Critic Agreement Results (Appendix Figure 5)

### 4.1 Critic Training Command (`reference/code/README.md`, Critic training Section)

In the Critic training section of `reference/code/README.md`, the Critic $\mathcal{C}$ is trained with `torchrun` and PyTorch Fully Sharded Data Parallelism (`--fsdp "full_shard auto_wrap"`):

```bash
cd data_creation
torchrun --nproc_per_node=2 \
  --master_port=2568 train_special_tokens.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path PATH_TO_TRAIN_DATA_FILE \
  --bf16 True \
  --output_dir PATH_TO_CRITIC_MODEL \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --fsdp "full_shard auto_wrap"
```

Note that in the `README.md` command above, `--use_special_token True` should also be passed so lines 310-319 of `train_special_tokens.py` register the $15$ special tokens via `smart_tokenizer_and_embedding_resize`.

This command also differs from the training setup in Appendix B.1 of the paper, which states that all models are trained for $3$ epochs with a batch size of $128$, a peak learning rate of $2 \times 10^{-5}$, $3\%$ warmup, and linear decay. Here the effective batch size is $2 \text{ GPUs} \times 1 \times 8 = 16$, warmup is $1\%$ (`--warmup_ratio 0.01`), and the schedule is cosine; only the epochs and learning rate match.

### 4.2 Critic Prediction Accuracy (Appendix A.2, Figure 5)

Appendix A.2 (Figure 5) evaluates how accurately the trained Critic $\mathcal{C}$ reproduces GPT-4's reflection token predictions on held-out test instances, comparing `Llama2-7B` (the final Critic base model) against a smaller `FLAN-3B` model trained on the exact same $\mathcal{D}_{\text{critic}}$ dataset:

| Critic Base Model | `Retrieve` Accuracy (%) | `IsSup` Accuracy (%) | `IsRel` Accuracy (%) | `IsUse` Accuracy (%) |
|---|---|---|---|---|
| `Llama2-7B` (Used in Self-RAG) | $93.8\%$ | $93.5\%$ | $80.2\%$ | $73.5\%$ |
| `FLAN-3B` | $85.6\%$ | $73.1\%$ | $82.0\%$ | $72.1\%$ |

Look at why `Llama2-7B` was chosen as the Critic $\mathcal{C}$:
1. On both `Retrieve` ($93.8\%$) and `IsSup` ($93.5\%$), `Llama2-7B` matches GPT-4's judgments over $93\%$ of the time, outperforming `FLAN-3B` by $8.2$ percentage points on `Retrieve` and $20.4$ points on `IsSup`. On `IsRel`, `FLAN-3B` is slightly higher ($82.0\%$ vs $80.2\%$).
2. On `IsUse` ($73.5\%$), the paper attributes the lower agreement to confusion between the two highest ratings (`[Utility:4]` vs `[Utility:5]`), a boundary where the authors' own check of GPT-4 labels also disagreed most ($80\%$ agreement in Part 1).
3. Section 3.2.1 of the paper summarizes this table as "higher than 90% agreement ... on most reflection token categories", but only two of the four categories ($93.8$ and $93.5$) exceed $90\%$. The section also cites it as "Appendix Table 5", while the appendix labels it Figure 5.
