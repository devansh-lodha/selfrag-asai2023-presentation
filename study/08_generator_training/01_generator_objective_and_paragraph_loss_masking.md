# Generator Objective and Paragraph Loss Masking

## 1. Plain-English Intuition: How One Loss Function Teaches Generation, Retrieval, and Critique

In Module 07, we saw how Stage 3 constructs the Generator training corpus $\mathcal{D}_{\text{gen}}$, where each example pairs an instruction $x$ with an augmented output string containing reflection tokens, retrieved passages `<paragraph>...</paragraph>`, and grounded response sentences:

- Input $x$: `"when did walking dead season 7 come out"`
- Augmented Output in $\mathcal{D}_{\text{gen}}$:
  `"[Retrieval]<paragraph>The Walking Dead (season 7)\nThe seventh season ... premiered on October 23, 2016 ...</paragraph>[Relevant]October 23 , 2016[Fully supported][Utility:5]"`

Now we want to train our final Generator language model $\mathcal{M}$ (Llama-2 7B or 13B) on $\mathcal{D}_{\text{gen}}$ in `reference/code/retrieval_lm/finetune.py`.

Think about what we want $\mathcal{M}$ to do when it sees this training example:
1. After reading the question $x$, $\mathcal{M}$ should learn to predict `[Retrieval]` and `<paragraph>`.
2. Next comes the retrieved Wikipedia passage (`"The Walking Dead (season 7)\nThe seventh season ..."`). At test time, $\mathcal{M}$ does not generate this Wikipedia passage from memory; the external retriever $\mathcal{R}$ pastes it into the prompt! Therefore, during training, we want $\mathcal{M}$ to read the passage tokens through causal self-attention (so it can extract `"October 23, 2016"`), but we do not want to penalize $\mathcal{M}$ in the loss function for failing to predict the words of the Wikipedia passage itself!
3. Immediately after the passage ends at `</paragraph>`, $\mathcal{M}$ should learn to predict `[Relevant]`, then the answer `"October 23 , 2016"`, then `[Fully supported]`, then `[Utility:5]`, and finally `</s>`.

As we learned in Module 01 Part 3, PyTorch's `IGNORE_INDEX = -100` lets us do this in a single forward pass: we keep the entire sequence (prompt, passage, reflection tokens, and answer) inside `input_ids` so self-attention can read everything, and we set both the prompt tokens and the interior tokens of `<paragraph>...</paragraph>` to `-100` inside `labels`.

---

## 2. Mathematical Formulation: The Generator Objective (Equation 2)

In Section 3.2.2 of the paper, the Generator training objective is written as standard next-token log-likelihood maximization over $\mathcal{D}_{\text{gen}}$ (Equation 2):

$$\max_{\mathcal{M}} \mathbb{E}_{(x, y, r) \sim \mathcal{D}_{\text{gen}}} \log p_{\mathcal{M}}(y, r \mid x)$$

where:
- $\mathcal{M}$ denotes the trainable parameters of the Generator language model (Llama-2 7B or 13B) over the expanded vocabulary $\mathcal{V}_{\text{exp}} = \mathcal{V} \cup \mathcal{V}_{\text{ref}}$ ($V_{\text{new}} = 32016$).
- $x$ is the formatted task instruction prompt.
- $y = (y_1, \dots, y_T)$ is the sequence of natural-language output segments.
- $r$ denotes all interleaved reflection tokens (`Retrieve`, `IsRel`, `IsSup`, `IsUse`) and context markup tags (`<paragraph>`, `</paragraph>`).

Let $u = (u_0, u_1, \dots, u_{L-1})$ denote the complete tokenized sequence of length $L$, formed by concatenating the prompt `source_text` and the augmented `target_text`. Partition the token indices $\{0, 1, \dots, L-1\}$ into three disjoint sets:
1. $\mathcal{I}_{\text{prompt}}$: the indices belonging to the user prompt $x$.
2. $\mathcal{I}_{\text{passage}}$: the indices strictly inside any `<paragraph>...</paragraph>` span (excluding the `<paragraph>` and `</paragraph>` boundary tokens themselves).
3. $\mathcal{I}_{\text{active\_target}}$: all remaining indices (the response sentences $y_t$, the reflection tokens $r$, the boundary tags `<paragraph>` and `</paragraph>`, and `</s>`).

Setting the label array $\ell = (\ell_0, \dots, \ell_{L-1})$ to `-100` on $\mathcal{I}_{\text{prompt}} \cup \mathcal{I}_{\text{passage}}$:

$$\ell_i = \begin{cases} -100 & \text{if } i \in \mathcal{I}_{\text{prompt}} \cup \mathcal{I}_{\text{passage}} \\ u_i & \text{if } i \in \mathcal{I}_{\text{active\_target}} \end{cases}$$

the exact cross-entropy loss minimized by `finetune.py` is:

$$\mathcal{L}_{\text{gen}}(\mathcal{M}) = -\frac{1}{\sum_{k=0}^{L-2} \mathbb{I}[\ell_{k+1} \neq -100]} \sum_{k=0}^{L-2} \mathbb{I}[\ell_{k+1} \neq -100] \log p_{\mathcal{M}}\left(u_{k+1} = \ell_{k+1} \;\middle|\; u_{\le k}\right)$$

---

## 3. Step-by-Step Index Walkthrough of `<paragraph>...</paragraph>` Masking (`finetune.py`, Lines 250-292)

### 3.1 The Masking Code in `encode_with_prompt_completion_format`

Let us inspect `encode_with_prompt_completion_format` in `reference/code/retrieval_lm/finetune.py` (lines 250-292; docstring and comments omitted below):

```python
def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, context_markups=None):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source_text = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    target_text = example['output'] + tokenizer.eos_token
    examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
    sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

    input_ids = examples_tokenized["input_ids"].flatten()
    source_len = sources_tokenized["input_ids_lens"]
    labels = copy.deepcopy(input_ids)
    labels[ :source_len-1] = -100

    if context_markups is not None:
        context_start = False
        for j, orig_token in enumerate(labels[source_len:]):
            if context_start is False and orig_token == context_markups[0]:
                context_start = True
                assert labels[source_len+j] == context_markups[0]
                start_idx = j+source_len
                end_idx = None
                for k, orig_token_2 in enumerate(labels[start_idx:]):
                    if orig_token_2 == context_markups[1]:
                        end_idx = start_idx + k
                if end_idx is None:
                    end_idx =  start_idx + k
                else:
                    assert labels[end_idx] == context_markups[1]
                labels[start_idx+1:end_idx] = -100
                context_start = False
    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten()
    }
```

### 3.2 Concrete 12-Token Index Table: Before and After Shift-by-One Alignment

Why does line 284 mask `labels[start_idx + 1 : end_idx] = -100` (starting at `start_idx + 1` instead of `start_idx`, and stopping before `end_idx`)?

Let us trace a concrete $12$-token example ($i = 0 \dots 11$) where:
- The prompt (`source_text`, indices $0, 1, 2$, so `source_len = 3`) is `["<s>", "Prompt:", "\n"]`.
- The target (`target_text`, indices $3 \dots 11$) is `["[Retrieval]", "<paragraph>", "Oct", "23", "</paragraph>", "[Relevant]", "Oct", "[Fully supported]", "</s>"]`.
- Here, `<paragraph>` sits at `start_idx = 4`, the passage words `"Oct"` and `"23"` sit at indices $5$ and $6$, and `</paragraph>` sits at `end_idx = 7`.

Let us line up `input_ids[i]`, the masked `labels[i]`, and the shifted pairs `(shift_logits[k], shift_labels[k])` inside `model(**batch)` (where `shift_logits[k] = logits[k]` and `shift_labels[k] = labels[k + 1]`):

| Original Index $i$ | `input_ids[i]` | Masked `labels[i]` | Shifted Slot $k = i$ (`shift_logits[i]`) | Paired Target `shift_labels[i]` (`= labels[i + 1]`) | What the Model Is Trained to Predict After Reading `input_ids[0..i]` |
|---|---|---|---|---|---|
| $0$ | `"<s>"` | `-100` | `logits[0]` (after `"<s>"`) | `labels[1]` = `-100` | Ignored (Prompt) |
| $1$ | `"Prompt:"` | `-100` | `logits[1]` (after `"Prompt:"`) | `labels[2]` = `"\n"` | Active: predicts trailing `"\n"` of `### Response:\n` |
| $2$ | `"\n"` | `"\n"` | `logits[2]` (after `"\n"`) | `labels[3]` = `"[Retrieval]"` | Active: predicts `"[Retrieval]"` after prompt |
| $3$ | `"[Retrieval]"` | `"[Retrieval]"` | `logits[3]` (after `"[Retrieval]"`) | `labels[4]` = `"<paragraph>"` | Active: predicts `"<paragraph>"` after `"[Retrieval]"` |
| $4$ (`start_idx`) | `"<paragraph>"` | `"<paragraph>"` | `logits[4]` (after `"<paragraph>"`) | `labels[5]` = `-100` | Ignored: does NOT predict passage word `"Oct"`! |
| $5$ (`start_idx+1`) | `"Oct"` | `-100` | `logits[5]` (after `"Oct"`) | `labels[6]` = `-100` | Ignored: does NOT predict passage word `"23"`! |
| $6$ (`end_idx-1`) | `"23"` | `-100` | `logits[6]` (after `"23"`) | `labels[7]` = `"</paragraph>"` | Active: predicts `"</paragraph>"` at end of passage |
| $7$ (`end_idx`) | `"</paragraph>"` | `"</paragraph>"` | `logits[7]` (after `"</paragraph>"`) | `labels[8]` = `"[Relevant]"` | Active: predicts `"[Relevant]"` right after `"</paragraph>"`! |
| $8$ | `"[Relevant]"` | `"[Relevant]"` | `logits[8]` (after `"[Relevant]"`) | `labels[9]` = `"Oct"` | Active: predicts answer word `"Oct"` |
| $9$ | `"Oct"` | `"Oct"` | `logits[9]` (after `"Oct"`) | `labels[10]` = `"[Fully supported]"` | Active: predicts `"[Fully supported]"` after answer |
| $10$ | `"[Fully supported]"` | `"[Fully supported]"` | `logits[10]` (after `"[Fully supported]"`) | `labels[11]` = `"</s>"` | Active: predicts `"</s>"` |
| $11$ | `"</s>"` | `"</s>"` | (Dropped by `[:-1]`) | None | (End of sequence) |

Look at rows $i = 3, 4, 5, 6, 7$ in the table above to see how every boundary index works:
1. Why `labels[start_idx]` (`i = 4`, `"<paragraph>"`) is NOT set to `-100`:
   Because `labels[4]` is the target for `logits[3]` (when the model has just output `"[Retrieval]"` at $i = 3$). Keeping `labels[4]` unmasked trains the model to output `"<paragraph>"` immediately after `"[Retrieval]"` (which is why in the quick-start demo in Module 03 Part 1, the model output `Sure![Retrieval]<paragraph>`).
2. Why `labels[start_idx + 1 : end_idx]` (`i = 5, 6`, `"Oct", "23"`) ARE set to `-100`:
   `labels[5]` is the target for `logits[4]` (right after `"<paragraph>"`), and `labels[6]` is the target for `logits[5]` (right after `"Oct"`). Setting `labels[5]` and `labels[6]` to `-100` ensures the model incurs zero loss on the words inside the Wikipedia passage.
3. Why `labels[end_idx + 1]` (`i = 8`, `"[Relevant]"`) is NOT set to `-100`:
   `labels[8]` is the target for `logits[7]` (right after reading `"</paragraph>"` at $i = 7$). Keeping `labels[8]` unmasked trains the model to output `"[Relevant]"` (or `"[Irrelevant]"`) at the very first token position after `"</paragraph>"`.

Also note why line 267 writes `labels[:source_len - 1] = -100` instead of `labels[:source_len] = -100`:
In `finetune.py` (lines 41-47), `PROMPT_DICT["prompt_no_input"]` ends with `"### Response:\n"` (including a trailing newline `\n` as the last token of `source_text` at index `source_len - 1`), whereas in `train_special_tokens.py` (lines 34-41), `PROMPT_DICT` ends with `"### Response:"` (no trailing `\n`). Masking `[:source_len - 1]` in `finetune.py` leaves the trailing `\n` token at index `source_len - 1` unmasked.

---

## 4. Diagnosing the Multi-Paragraph Masking Bug (`finetune.py` Lines 269-285)

Look closely at the inner `for k` loop inside `encode_with_prompt_completion_format` (lines 269-285 of `finetune.py`, and the same logic without the assertions on lines 212-224 of `train_special_tokens.py`):

```python
        for j, orig_token in enumerate(labels[source_len:]):
            if context_start is False and orig_token == context_markups[0]:
                context_start = True
                assert labels[source_len+j] == context_markups[0]
                start_idx = j+source_len
                end_idx = None
                for k, orig_token_2 in enumerate(labels[start_idx:]):
                    if orig_token_2 == context_markups[1]:
                        end_idx = start_idx + k
                if end_idx is None:
                    end_idx =  start_idx + k
                else:
                    assert labels[end_idx] == context_markups[1]
                labels[start_idx+1:end_idx] = -100
                context_start = False
```

What happens if a single training example in $\mathcal{D}_{\text{gen}}$ contains two (or more) retrieved passages `<paragraph>d_1</paragraph>` and `<paragraph>d_2</paragraph>` across multiple sentences (as in Figure 2 Right and Appendix Table 4 Examples 1 and 2)?

Let us trace the indices when an example has two `<paragraph>...</paragraph>` spans:
- Span 1: `<paragraph>` #1 is at index $10$, and `</paragraph>` #1 is at index $30$.
- Sentence 1 and its reflection tokens (`"[Relevant] Sentence 1 [Fully supported] [Retrieval]"`) sit at indices $31 \dots 45$.
- Span 2: `<paragraph>` #2 is at index $46$, and `</paragraph>` #2 is at index $70$.
- Sentence 2 and its reflection tokens (`"[Relevant] Sentence 2 [Partially supported] [Utility:5] </s>"`) sit at indices $71 \dots 85$.

Now trace what the code above executes when the outer loop `for j` reaches `<paragraph>` #1 at `start_idx = 10`:
1. The inner loop `for k, orig_token_2 in enumerate(labels[10:]):` scans every token from index $10$ all the way to the end of the sequence (index $85$).
2. When `10 + k` reaches index $30$ (`</paragraph>` #1), `orig_token_2 == context_markups[1]` is `True`, so `end_idx` is set to $30$.
3. However, there is no `break` statement after `end_idx = start_idx + k`!
4. The inner loop continues scanning past index $30$, and when `10 + k` reaches index $70$ (`</paragraph>` #2), `orig_token_2 == context_markups[1]` is `True` again, overwriting `end_idx = 70`!
5. Next, line 284 executes `labels[start_idx + 1 : end_idx] = -100`, which sets `labels[11 : 70] = -100`!

Because `end_idx` was overwritten by the last `</paragraph>` in the sequence (index $70$), `labels[11 : 70] = -100` masks out not only Passage 1 (indices $11 \dots 29$) and Passage 2 (indices $47 \dots 69$), but also Sentence 1 and its reflection tokens (indices $31 \dots 46$)! Only the final sentence after the last `</paragraph>` (indices $71 \dots 85$) remains unmasked in `labels`. (Also note that if a single long passage is truncated by `max_seq_length` before its closing `</paragraph>`, `end_idx` starts as `None` and lines 280-281 set `end_idx = start_idx + k`, masking to the end of the truncated sequence except its last token.)

This bug is silent: no exception is raised. `labels` is a PyTorch tensor, and iterating over it yields views into the same storage, so by the time the outer `for j` loop reaches `<paragraph>` #2 at index $46$, that position already reads `-100`. The `if ... orig_token == context_markups[0]` test is therefore false, and the `assert` inside the branch never runs. (Two earlier audit notes in this repository claimed an `AssertionError` here; that is not what happens.) The practical effect is that, for every multi-passage training example, the intermediate sentences and their reflection tokens (`[Relevant]`, `[Fully supported]`, `[Retrieval]`, and so on) receive no training signal.

---

## 5. Tokenizer Setup, LoRA Support, and Dataset Mapping in `finetune.py`

Finally, let us note six implementation details in `main()` of `reference/code/retrieval_lm/finetune.py` (lines 444-530):

1. Embedding initialization without mean averaging (lines 474-476):
   Unlike `train_special_tokens.py` (which initialized the $16$ new token rows to the mean of the $32,000$ existing rows via `smart_tokenizer_and_embedding_resize`), `finetune.py` calls `model.resize_token_embeddings(len(tokenizer))` directly, leaving the $16$ new rows with Hugging Face's default normal initialization.
2. Optional LoRA fine-tuning (`--use_lora`, around line 481):
   If `--use_lora` is passed, `finetune.py` wraps the model in a Low-Rank Adaptation (`peft.LoraConfig`) layer with rank `r=64`, `lora_alpha=16`, and `lora_dropout=0.1`. (In the official Self-RAG 7B and 13B training scripts `script_finetune_7b.sh` and `script_finetune_13b.sh`, `--use_lora` is not passed, so full-parameter fine-tuning is performed.)
3. Filtering fully masked examples (line 515):
   After running `raw_datasets.map(encode_function, ...)`, line 515 runs:
   ```python
   lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())
   ```
   This discards any training example whose active target tokens were completely cut off by `max_seq_length` truncation (which would otherwise have all `labels == -100`, causing a `0 / 0 = NaN` loss).
4. Debug dump to `processed.json` (lines 518-528):
   Lines 518-522 print five sample items (`0, 1000, 500, 2000, 10000`), and lines 523-528 loop through the whole `train_dataset`, printing every item and dumping its `labels` and `input_ids` into a local file `"processed.json"` before the training loop starts. Because `print(train_dataset[10000])` assumes more than $10,000$ examples, the script raises an `IndexError` on smaller training sets, such as the $5\text{k}$ and $10\text{k}$ subsets of the data-scaling study.
5. Training without reflection tokens fails (lines 449-454): `special_token_dict` is created only inside `if args.use_special_tokens is True:`, but the next line writes `special_token_dict["bos_token"] = "<s>"` unconditionally, so running without `--use_special_tokens` (as a baseline such as `Llama2-FT`, trained without reflection tokens, would need) raises a `NameError`.
6. Per-call printing: `_tokenize_fn` (line 241) prints the token length of every string it tokenizes.
