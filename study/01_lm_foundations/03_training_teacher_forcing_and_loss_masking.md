# Training, Teacher Forcing, and Loss Masking

## 1. Plain-English Intuition: How Do We Train a Next-Token Predictor?

In Part 1 and Part 2, we saw how a language model takes a sequence of input tokens, blends their representations using causal self-attention, and outputs a probability distribution over the vocabulary $\mathcal{V}$ for the next token.

Now suppose we want to train the model's weights $\theta$ on a question-answering example from our training dataset:
- Prompt $x$ ($m = 3$ tokens): `["Capital", "of", "France?"]`
- Target response $y$ ($n = 5$ tokens): `["Paris", "is", "the", "capital.", "</s>"]`

For the model to generate the response $y$ given the prompt $x$, it needs to make $5$ correct next-token predictions in a row:
1. After reading `["Capital", "of", "France?"]`, predict `"Paris"`.
2. After reading `["Capital", "of", "France?", "Paris"]`, predict `"is"`.
3. After reading `["Capital", "of", "France?", "Paris", "is"]`, predict `"the"`.
4. After reading `["Capital", "of", "France?", "Paris", "is", "the"]`, predict `"capital."`.
5. After reading `["Capital", "of", "France?", "Paris", "is", "the", "capital."]`, predict `"</s>"`.

### 1.1 What Goes Wrong If the Model Generates Freely During Training?

Suppose that during training, we fed only the prompt `["Capital", "of", "France?"]` into the model, let the model pick its own predicted token at step 1, and then fed the model's predicted token back into the input for step 2.

Early in training, the model's weights are not yet trained. At step 1, given `["Capital", "of", "France?"]`, the model might guess a wrong word such as `"Banana"`. If we append `"Banana"` to the input for step 2, the model now sees:

`["Capital", "of", "France?", "Banana"]`

Asking the model at step 2 to predict `"is"`, and at step 3 to predict `"the"` after `"Banana"`, trains the model on a corrupted, nonsensical context. A single mistake at the first token derails the entire rest of the sentence. Moreover, waiting for step 1 to finish before running step 2 would require running $5$ separate, sequential forward passes on the GPU.

### 1.2 How Teacher Forcing and Causal Masking Solve Both Problems

Teacher forcing fixes this with a simple rule: during training, regardless of what token the model's probabilities favored at step 1, we always feed the true ground-truth token from the dataset (the "teacher's" token, `"Paris"`) as the input history for step 2.

Even better, look at the $5$ ground-truth prefixes required across the $5$ steps above:
- Prefix 1: `["Capital", "of", "France?"]`
- Prefix 2: `["Capital", "of", "France?", "Paris"]`
- Prefix 3: `["Capital", "of", "France?", "Paris", "is"]`
- Prefix 4: `["Capital", "of", "France?", "Paris", "is", "the"]`
- Prefix 5: `["Capital", "of", "France?", "Paris", "is", "the", "capital."]`

Every one of those prefixes is just a left-to-right slice of the single $8$-token concatenated sequence $u$ formed by joining $x$ and $y$:

$$u = (\text{``Capital''},\; \text{``of''},\; \text{``France?''},\; \text{``Paris''},\; \text{``is''},\; \text{``the''},\; \text{``capital.''},\; \text{``</s>''})$$

Recall from Part 2 that causal self-attention blocks any position $i$ from seeing future positions $> i$. Therefore, if we feed the entire $8$-token sequence $u$ into the transformer in one single forward pass:
- At index $i = 2$ (`"France?"`), causal attention only sees indices $0, 1, 2$ (`["Capital", "of", "France?"]`), which is Prefix 1.
- At index $i = 3$ (`"Paris"`), causal attention only sees indices $0, 1, 2, 3$ (`["Capital", "of", "France?", "Paris"]`), which is Prefix 2.
- At index $i = 4$ (`"is"`), causal attention only sees indices $0, 1, 2, 3, 4$ (`["Capital", "of", "France?", "Paris", "is"]`), which is Prefix 3.

Because of the causal mask, feeding the concatenated sequence $u$ of length $L = m + n = 8$ into the transformer computes the next-token predictions for every prefix in a single parallel GPU forward pass.

---

## 2. Step-by-Step Walkthrough: Shift-by-One Alignment (`shift_logits` and `shift_labels`)

### 2.1 What the Transformer Outputs at Each Position (`logits[i]`)

When we pass the $8$-token array `input_ids` into the transformer, the model outputs a logit vector `logits[i]` (of length $V = 32000$) at every input position $i \in \{0, 1, 2, 3, 4, 5, 6, 7\}$.

At position $i$, the transformer has just finished reading `input_ids[0]` through `input_ids[i]`, so `logits[i]` is the model's prediction for the next token at position $i + 1$:
- At index $i = 0$ (input `"Capital"`), `logits[0]` predicts what comes after `"Capital"`, which is the token at index $1$ (`"of"`).
- At index $i = 1$ (input `"of"`), `logits[1]` predicts what comes after `"Capital", "of"`, which is the token at index $2$ (`"France?"`).
- At index $i = 2$ (input `"France?"`), `logits[2]` predicts what comes after `"Capital", "of", "France?"`, which is the token at index $3$ (`"Paris"`).
- At index $i = 6$ (input `"capital."`), `logits[6]` predicts what comes after `"capital."`, which is the token at index $7$ (`"</s>"`).
- At index $i = 7$ (input `"</s>"`), `logits[7]` predicts what comes after the end of the sequence, which does not exist.

### 2.2 Why Comparing `logits[i]` to `labels[i]` at the Same Index Would Be Wrong

In the Self-RAG training scripts (`reference/code/data_creation/train_special_tokens.py` line 176 and `reference/code/retrieval_lm/finetune.py` line 246), the target array `labels` is initially created as an exact copy of `input_ids`:

```python
input_ids = examples_tokenized["input_ids"]
labels = copy.deepcopy(input_ids)
```

Let us put `input_ids`, `logits`, and the copied `labels` in a table by their array index $i \in \{0, \dots, 7\}$:

| Array Index $i$ | `input_ids[i]` | `labels[i]` (Exact Copy of `input_ids[i]`) | What `logits[i]` Is Actually Predicting | Where That True Target Token Sits in `labels` |
|---|---|---|---|---|
| $0$ | `"Capital"` | `"Capital"` | `"of"` | `labels[1]` |
| $1$ | `"of"` | `"of"` | `"France?"` | `labels[2]` |
| $2$ | `"France?"` | `"France?"` | `"Paris"` | `labels[3]` |
| $3$ | `"Paris"` | `"Paris"` | `"is"` | `labels[4]` |
| $4$ | `"is"` | `"is"` | `"the"` | `labels[5]` |
| $5$ | `"the"` | `"the"` | `"capital."` | `labels[6]` |
| $6$ | `"capital."` | `"capital."` | `"</s>"` | `labels[7]` |
| $7$ | `"</s>"` | `"</s>"` | (After end of sequence) | None |

Look at row $i = 2$ in the table above:
- At index $i = 2$, the model has just read `input_ids[2] = "France?"`, so `logits[2]` is its prediction for the next word `"Paris"`.
- However, `"Paris"` is not at `labels[2]` (`labels[2]` is `"France?"`). `"Paris"` is at `labels[3]`!
- If we compared `logits[2]` against `labels[2]`, we would be training the model at position $2$ (which was just handed `"France?"` as input) to output `"France?"`. Across the whole sequence, comparing `logits[i]` to `labels[i]` would merely train the network to copy the current input token `input_ids[i]` to the output, rather than predicting the next token `input_ids[i + 1]`.

### 2.3 How Slicing Creates `shift_logits` and `shift_labels`

To train next-token prediction properly, we must pair `logits[i]` with `labels[i + 1]` for every $i \in \{0, 1, 2, 3, 4, 5, 6\}$:
- `logits[0]` must be paired with `labels[1]` (`"of"`).
- `logits[1]` must be paired with `labels[2]` (`"France?"`).
- `logits[2]` must be paired with `labels[3]` (`"Paris"`).
- `logits[3]` must be paired with `labels[4]` (`"is"`).
- `logits[4]` must be paired with `labels[5]` (`"the"`).
- `logits[5]` must be paired with `labels[6]` (`"capital."`).
- `logits[6]` must be paired with `labels[7]` (`"</s>"`).

Notice which two elements are left over and never used:
1. The last logit vector `logits[7]` (at index $L - 1 = 7$) is never used, because `"</s>"` is the end of the sequence and there is no `labels[8]`.
2. The first label `labels[0]` (`"Capital"`, at index $0$) is never used, because there is no token before index $0$ to predict it.

Therefore, inside Hugging Face's `LlamaForCausalLM.forward` (which runs automatically when the training script calls `outputs = model(**batch)`), PyTorch aligns the two arrays in two lines of code:

```python
# 1. Drop the last logit (index 7), keeping indices 0, 1, 2, 3, 4, 5, 6
shift_logits = logits[..., :-1, :].contiguous()  # shape: [B, 7, V]

# 2. Drop the first label (index 0), keeping indices 1, 2, 3, 4, 5, 6, 7
shift_labels = labels[..., 1:].contiguous()      # shape: [B, 7]
```

Here, Python's slice `[:-1]` means "take every element from index $0$ up to (but not including) the last element", and `[1:]` means "take every element starting from index $1$ to the end".

Look at how `shift_logits` and `shift_labels` line up at every new index $k \in \{0, 1, 2, 3, 4, 5, 6\}$ after those two slices:

| New Aligned Index $k$ | `shift_logits[k]` (Comes from `logits[k]`) | Context Read by Model | `shift_labels[k]` (Comes from `labels[k + 1]`) | Prediction Evaluated at Index $k$ |
|---|---|---|---|---|
| $k = 0$ | `logits[0]` | `["Capital"]` | `labels[1]` = `"of"` | `"Capital"` $\to$ `"of"` |
| $k = 1$ | `logits[1]` | `["Capital", "of"]` | `labels[2]` = `"France?"` | `"Capital", "of"` $\to$ `"France?"` |
| $k = 2$ | `logits[2]` | `["Capital", "of", "France?"]` | `labels[3]` = `"Paris"` | `"Capital", "of", "France?"` $\to$ `"Paris"` |
| $k = 3$ | `logits[3]` | `..., "Paris"]` | `labels[4]` = `"is"` | `..., "Paris"` $\to$ `"is"` |
| $k = 4$ | `logits[4]` | `..., "Paris", "is"]` | `labels[5]` = `"the"` | `..., "is"` $\to$ `"the"` |
| $k = 5$ | `logits[5]` | `..., "is", "the"]` | `labels[6]` = `"capital."` | `..., "the"` $\to$ `"capital."` |
| $k = 6$ | `logits[6]` | `..., "the", "capital."]` | `labels[7]` = `"</s>"` | `..., "capital."` $\to$ `"</s>"` |

Because `model(**batch)` performs this `[:-1]` and `[1:]` shift internally, the dataset preparation code in `train_special_tokens.py` and `finetune.py` does not need to shift `labels` by hand; it simply creates `labels = copy.deepcopy(input_ids)` of length $L$, masks out the positions that should be ignored, and passes both `input_ids` and `labels` to the model.

---

## 3. Step-by-Step Build-Up: Cross-Entropy Loss and Prompt Masking with `-100`

### 3.1 How Cross-Entropy Loss Penalizes Wrong Predictions

At each aligned slot $k$ in the table above, `shift_logits[k]` gives the model's softmax probability distribution over all $V$ tokens in the vocabulary, and `shift_labels[k]` (which is $u_{k+1}$) tells us the single ground-truth token that should have received $100\%$ of the probability.

Let $p_k = p_\theta(u_{k+1} \mid u_{\le k}) \in (0, 1)$ denote the probability that the model assigned to the true target token $u_{k+1}$ at slot $k$. How do we turn $p_k$ into a loss number $\ell_k$ that the optimizer tries to minimize?
- If the model is completely confident in the right answer ($p_k = 1.0$), the loss $\ell_k$ should be $0$.
- As the probability assigned to the right answer drops toward zero ($p_k \to 0$), the loss $\ell_k$ should grow larger and larger.

Recall from Part 2 that $\log(1.0) = 0$ and $\log(p_k)$ becomes a large negative number when $p_k$ is small. Flipping the sign gives the negative log-likelihood, known as the cross-entropy loss at slot $k$:

$$\ell_{\text{CE}}(k) = -\log p_\theta(u_{k+1} \mid u_{\le k}) \in [0, \infty)$$

For example:
- If the model assigns $p_k = 0.80$ ($80\%$) to `"Paris"`, the loss is $-\log(0.80) \approx 0.223$.
- If the model assigns $p_k = 0.01$ ($1\%$) to `"Paris"`, the loss is $-\log(0.01) \approx 4.605$ (a much larger penalty).

### 3.2 Why We Must Mask Out the Prompt Tokens ($k = 0$ and $k = 1$)

Look again at rows $k = 0$ and $k = 1$ in our aligned table from Section 2.3:
- At $k = 0$, `shift_labels[0]` is `"of"` (the second word of the user's question `"Capital of France?"`).
- At $k = 1$, `shift_labels[1]` is `"France?"` (the third word of the user's question).
- Starting at $k = 2$, `shift_labels[2]` is `"Paris"` (the first word of the model's answer).

During instruction tuning, our goal is to train the model to generate the response $y$ given the prompt $x$, which corresponds to maximizing $\log p_\theta(y \mid x)$ (slots $k = 2, 3, 4, 5, 6$). We do not want to penalize the model at slots $k = 0$ and $k = 1$ for failing to guess what question the user was going to ask!

However, we cannot remove `"Capital"`, `"of"`, and `"France?"` from `input_ids`, because the model must read those three words in `input_ids` to produce `logits[2]` (the prediction for `"Paris"`).

### 3.3 How `IGNORE_INDEX = -100` Masks Positions Without Removing Them from `input_ids`

PyTorch's cross-entropy loss function (`torch.nn.CrossEntropyLoss`) reserves a special magic integer: `ignore_index = -100` (defined at the top of `reference/code/data_creation/train_special_tokens.py` line 27 as `IGNORE_INDEX = -100`).

Because valid token IDs in the vocabulary are non-negative integers $\{0, 1, \dots, V - 1\}$, `-100` is never a valid token ID. Whenever `CrossEntropyLoss` sees `-100` in `shift_labels[k]`, it completely skips slot $k$: slot $k$ contributes $0$ to the loss, receives $0$ gradient, and is not counted in the number of active tokens when averaging the loss.

Let us trace what happens in `reference/code/data_creation/train_special_tokens.py` (lines 177-179) when the prompt $x = (\text{``Capital''}, \text{``of''}, \text{``France?''})$ has length `source_len = 3`, and the code runs:

```python
for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    label[:source_len] = IGNORE_INDEX  # sets indices 0, 1, 2 of labels to -100
```

Here is the exact index-by-index state of `input_ids` and `labels` before and after `label[:3] = -100`:

| Original Index $i$ | $i = 0$ | $i = 1$ | $i = 2$ | $i = 3$ | $i = 4$ | $i = 5$ | $i = 6$ | $i = 7$ |
|---|---|---|---|---|---|---|---|---|
| Role | Prompt $x_1$ | Prompt $x_2$ | Prompt $x_3$ | Target $y_1$ | Target $y_2$ | Target $y_3$ | Target $y_4$ | Target $y_5$ |
| `input_ids[i]` | `"Capital"` | `"of"` | `"France?"` | `"Paris"` | `"is"` | `"the"` | `"capital."` | `"</s>"` |
| `labels[i]` (before mask) | `"Capital"` | `"of"` | `"France?"` | `"Paris"` | `"is"` | `"the"` | `"capital."` | `"</s>"` |
| `labels[i]` (after `labels[:3] = -100`) | `-100` | `-100` | `-100` | `"Paris"` | `"is"` | `"the"` | `"capital."` | `"</s>"` |

Now watch what happens when `input_ids` and the masked `labels` are passed into `model(**batch)` and sliced into `shift_logits = logits[:, :-1, :]` and `shift_labels = labels[:, 1:]`:
- Dropping index $0$ of `labels` (`labels[0] = -100`) shifts indices $1 \dots 7$ into slots $k = 0 \dots 6$.
- Let us inspect all $7$ aligned slots $k \in \{0, \dots, 6\}$ together with concrete example probabilities $p_k$:

| Shifted Slot $k$ | `shift_logits[k]` (From `logits[k]`) | `shift_labels[k]` (From `labels[k + 1]`) | Active in Loss? | Example Model Prob $p_k$ | Slot Loss $-\log(p_k)$ |
|---|---|---|---|---|---|
| $k = 0$ | `logits[0]` (after `"Capital"`) | `labels[1]` = `-100` | Ignored (`-100`) | $0.02$ | $0.0$ (skipped) |
| $k = 1$ | `logits[1]` (after `"of"`) | `labels[2]` = `-100` | Ignored (`-100`) | $0.05$ | $0.0$ (skipped) |
| $k = 2$ | `logits[2]` (after `"France?"`) | `labels[3]` = `"Paris"` | Active ($1$) | $0.80$ | $-\log(0.80) = 0.2231$ |
| $k = 3$ | `logits[3]` (after `"Paris"`) | `labels[4]` = `"is"` | Active ($1$) | $0.50$ | $-\log(0.50) = 0.6931$ |
| $k = 4$ | `logits[4]` (after `"is"`) | `labels[5]` = `"the"` | Active ($1$) | $0.70$ | $-\log(0.70) = 0.3567$ |
| $k = 5$ | `logits[5]` (after `"the"`) | `labels[6]` = `"capital."` | Active ($1$) | $0.60$ | $-\log(0.60) = 0.5108$ |
| $k = 6$ | `logits[6]` (after `"capital."`) | `labels[7]` = `"</s>"` | Active ($1$) | $0.90$ | $-\log(0.90) = 0.1054$ |

Look at slot $k = 2$ in the table above:
- Even though `labels[2]` (at original index $2$, `"France?"`) was set to `-100`, `shift_logits[2]` (the logit vector produced after reading `"France?"`) is paired with `shift_labels[2] = labels[3] = "Paris"`!
- Meanwhile, the predictions for `"of"` ($k = 0$) and `"France?"` ($k = 1$) are paired with `labels[1] = -100` and `labels[2] = -100`, so they are completely skipped.
- There are $5$ active slots ($k = 2, 3, 4, 5, 6$), and the total sequence loss is the average over those $5$ active slots:

$$\mathcal{L} = \frac{0.2231 + 0.6931 + 0.3567 + 0.5108 + 0.1054}{5} = \frac{1.8891}{5} = 0.3778$$

### 3.4 General Mathematical Equation for Masked Cross-Entropy Loss

Writing this general calculation for any concatenated sequence $u = (u_0, u_1, \dots, u_{L-1})$ and unshifted label array $\ell = (\ell_0, \ell_1, \dots, \ell_{L-1})$ where masked positions are set to $\ell_i = -100$:

$$\mathcal{L}(\theta; u, \ell) = -\frac{1}{\sum_{k=0}^{L-2} \mathbb{I}[\ell_{k+1} \neq -100]} \sum_{k=0}^{L-2} \mathbb{I}[\ell_{k+1} \neq -100] \log p_\theta\left(u_{k+1} = \ell_{k+1} \;\middle|\; u_{\le k}\right)$$

where:
- $L \in \mathbb{Z}^+$ is the total token length of the sequence (`len(input_ids)`).
- $k \in \{0, 1, \dots, L - 2\}$ is the shifted slot index (indexing `shift_logits[k]` and `shift_labels[k]`).
- $\ell_{k+1} \in \{0, \dots, V - 1\} \cup \{-100\}$ is the unshifted label at index $k + 1$ (which equals `shift_labels[k]`).
- $\mathbb{I}[\ell_{k+1} \neq -100] \in \{0, 1\}$ is the indicator function that equals $1$ if slot $k$ has an active target token and $0$ if $\ell_{k+1} = -100$.
- $\sum_{k=0}^{L-2} \mathbb{I}[\ell_{k+1} \neq -100]$ is the count of active (non-masked) target tokens in the sequence ($5$ in our example).
- $\log p_\theta(u_{k+1} = \ell_{k+1} \mid u_{\le k})$ is the log-probability assigned by `shift_logits[k]` to the true next token $\ell_{k+1}$.

---

## 4. Paper and Codebase Grounding

### 4.1 Prompt and Passage Masking in `train_special_tokens.py`

Let us examine the complete `preprocess` function in `reference/code/data_creation/train_special_tokens.py` (lines 171-204):

```python
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    skip_tokens:Sequence[int],
    context_markups: Sequence[int],
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    
    # special token mask
    if skip_tokens is not None:
        for i, input_id_list in enumerate(input_ids):
            for j, orig_token in enumerate(input_id_list):
                if orig_token in skip_tokens:
                    labels[i][j] = IGNORE_INDEX

    if context_markups is not None:
        for i, (label_id_list, source_len) in enumerate(zip(labels, sources_tokenized["input_ids_lens"])):
            context_start = False
            for j, orig_token in enumerate(label_id_list[source_len:]):
                if context_start is False and orig_token == context_markups[0]:
                    context_start = True
                    start_idx = j+source_len
                    for k, orig_token_2 in enumerate(label_id_list[start_idx:]):
                        if orig_token_2 == context_markups[1]:
                            end_idx = start_idx + k
                    labels[i][start_idx+1:end_idx] = IGNORE_INDEX
                    context_start = False
    return dict(input_ids=input_ids, labels=labels)
```

Every block in `preprocess` applies the exact `-100` masking logic we traced above:
1. Lines 173-179 (Prompt masking):
   - `examples = [s + t for s, t in zip(sources, targets)]` concatenates each prompt string `s` and target response string `t`.
   - `_tokenize_fn` tokenizes both the full string `s + t` (`examples_tokenized`) and the prompt string `s` alone (`sources_tokenized`) so that `sources_tokenized["input_ids_lens"]` gives `source_len` (the number of tokens in the prompt $x$).
   - `labels = copy.deepcopy(input_ids)` makes an unshifted copy of `input_ids`, and `label[:source_len] = IGNORE_INDEX` sets the first `source_len` positions to `-100`.
2. Lines 182-186 (Optional special token masking):
   - If the ablation flag `--not_consider_special_tokens` is set, `skip_tokens` contains the integer IDs of the reflection tokens, and setting `labels[i][j] = IGNORE_INDEX` excludes those reflection tokens from the loss.
3. Lines 188-200 (Retrieved passage masking via `<paragraph>` and `</paragraph>`):
   - In Self-RAG generator training (Section 3.2.2 of the paper: *"During training, we mask out the retrieved text chunks (surrounded by `<p>` and `</p>` in Figure 2) for loss calculation"*), retrieved Wikipedia passages are inserted directly into the target string between `context_markups[0]` (`<paragraph>`) and `context_markups[1]` (`</paragraph>`).
   - Setting `labels[i][start_idx+1:end_idx] = IGNORE_INDEX` sets all tokens strictly inside the paragraph (`start_idx + 1` up to `end_idx - 1`) to `-100`.
   - As a result, the retrieved passage stays intact inside `input_ids` so subsequent generated sentences can attend to the Wikipedia evidence, while `CrossEntropyLoss` skips all passage tokens (`-100`) so the model is never penalized for not memorizing the Wikipedia passage itself. (We will examine the subtle boundary difference `labels[:source_len-1] = -100` in `finetune.py` in Module 08.)
