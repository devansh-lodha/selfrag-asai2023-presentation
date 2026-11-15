# Tokenizer Mechanics and Embedding Expansion Code

## 1. Plain-English Intuition: Why Reflection Tokens Must Be Registered as Atomic Special Tokens

In Part 1, we learned the $15$ special token strings that Self-RAG uses to control retrieval and self-critique, such as `"[Fully supported]"` and `"[Continue to Use Evidence]"`.

Now consider what happens if we pass the string `"[Fully supported]"` into a standard, unmodified Llama-2 tokenizer (`meta-llama/Llama-2-7b-hf`) without registering it as a special token first.

Because the base Llama-2 vocabulary ($V = 32000$) only contains ordinary subwords and characters, the SentencePiece Byte-Pair Encoding (BPE) algorithm breaks `"[Fully supported]"` into several ordinary pieces: an opening-bracket piece, one or more pieces for the words, and a closing-bracket piece. `"[Partially supported]"` is split the same way, and both splits begin with the same bracket piece. (The exact pieces depend on the SentencePiece vocabulary and on the character before the bracket; the argument below needs only two facts: the split has several pieces, and its first piece is shared.)

If reflection tokens were split into multi-token subword sequences like this, three critical mechanisms in Self-RAG would break:
1. Single-position probability comparison would be impossible: At inference time, immediately after generating a sentence $s_t$, we want to read the model's probability for `"[Fully supported]"` versus `"[Partially supported]"` at a single token index `idx`. Because both strings begin with the exact same bracket piece, at position `idx` both options would predict that same piece!
2. Slower generation and wasted context length: Generating a reflection tag would take several sequential transformer steps instead of $1$ step.
3. Embedding interference with ordinary words: The word pieces of `"[Fully supported]"` would share their embedding vectors with the same pieces in ordinary English text (such as the word "supported"), rather than having a dedicated vector representing the entailment critique `[Fully supported]`.

Registering all $15$ strings via `tokenizer.add_special_tokens({"additional_special_tokens": [...]})` solves all three problems:
- The tokenizer adds an exact-match pre-tokenization rule for each registered string. Whenever `"[Fully supported]"` appears in the text, the tokenizer bypasses BPE splitting and maps the entire string to a single, indivisible integer token ID ($32012$).
- We then resize the model's input embedding matrix $E_{\text{in}}$ and output head $W_{\text{out}}$ from $32000$ rows to $32016$ rows so each new special token gets its own dedicated $d$-dimensional vector.

---

## 2. Step-by-Step Build-Up: Token ID Assignment and Matrix Resizing ($32000 \to 32016$)

### 2.1 Exact Token IDs Assigned to the 16 New Tokens

In both `reference/code/data_creation/train_special_tokens.py` (lines 306-308) and `reference/code/retrieval_lm/finetune.py` (lines 380-385), the tokenizer starts with the $32000$ base Llama-2 tokens (indices $0$ through $31999$) and registers $15$ `additional_special_tokens` plus $1$ `pad_token`:

```python
special_token_dict = {
    "additional_special_tokens": [
        "[No Retrieval]",
        "[Retrieval]",
        "[Continue to Use Evidence]",
        "[Irrelevant]",
        "[Relevant]",
        "<paragraph>",
        "</paragraph>",
        "[Utility:1]",
        "[Utility:2]",
        "[Utility:3]",
        "[Utility:4]",
        "[Utility:5]",
        "[Fully supported]",
        "[Partially supported]",
        "[No support / Contradictory]"
    ]
}
```

When `tokenizer.add_special_tokens(special_token_dict)` is called, Hugging Face appends the new tokens in order to the end of the vocabulary, assigning them integer IDs $32000$ through $32015$:

| New Token Index $v$ | Exact Token String | Token Group in Self-RAG |
|---|---|---|
| $32000$ | `[No Retrieval]` | `Retrieve` |
| $32001$ | `[Retrieval]` | `Retrieve` |
| $32002$ | `[Continue to Use Evidence]` | `Retrieve` |
| $32003$ | `[Irrelevant]` | `IsRel` |
| $32004$ | `[Relevant]` | `IsRel` |
| $32005$ | `<paragraph>` | Context Markup |
| $32006$ | `</paragraph>` | Context Markup |
| $32007$ | `[Utility:1]` | `IsUse` |
| $32008$ | `[Utility:2]` | `IsUse` |
| $32009$ | `[Utility:3]` | `IsUse` |
| $32010$ | `[Utility:4]` | `IsUse` |
| $32011$ | `[Utility:5]` | `IsUse` |
| $32012$ | `[Fully supported]` | `IsSup` |
| $32013$ | `[Partially supported]` | `IsSup` |
| $32014$ | `[No support / Contradictory]` | `IsSup` |
| $32015$ | `<pad>` (or `[PAD]`) | Padding Token |

Notice why the number $32016$ appears in `reference/code/retrieval_lm/run_short_form.py` (line 58: `SamplingParams(..., logprobs=32016)`): $32000$ base tokens $+ 16$ added special tokens $= 32016$ total tokens in $\mathcal{V}_{\text{exp}}$!

### 2.2 Resizing the Input and Output Embedding Matrices

Let $V_{\text{old}} = 32000$ denote the original vocabulary size, let $M = 16$ denote the number of newly added special tokens, and let $V_{\text{new}} = V_{\text{old}} + M = 32016$ denote the expanded vocabulary size.

Recall from Module 01 Part 1 and Part 2 that two weight matrices in the transformer have shape $[V, d]$:
1. The input embedding matrix $E_{\text{in}} \in \mathbb{R}^{V \times d}$ (`model.get_input_embeddings().weight`), which maps input token IDs to vectors of size $d$.
2. The output projection head $W_{\text{out}} \in \mathbb{R}^{V \times d}$ (`model.get_output_embeddings().weight`), which projects the final hidden vector $h_t \in \mathbb{R}^d$ into $V$ vocabulary logits $z_j = W_{\text{out}} h_t \in \mathbb{R}^V$.

When `model.resize_token_embeddings(len(tokenizer))` is called, PyTorch allocates two new matrices $E_{\text{in}}^{\text{new}} \in \mathbb{R}^{32016 \times d}$ and $W_{\text{out}}^{\text{new}} \in \mathbb{R}^{32016 \times d}$, copies the first $32000$ pre-trained rows unchanged:

$$E_{\text{in}}^{\text{new}}[0 : 32000, :] = E_{\text{in}}^{\text{old}}, \quad W_{\text{out}}^{\text{new}}[0 : 32000, :] = W_{\text{out}}^{\text{old}}$$

and appends $16$ new trainable rows at indices $32000 \dots 32015$:

$$E_{\text{in}}^{\text{new}} = \begin{bmatrix} E_{\text{in}}^{\text{old}} \in \mathbb{R}^{32000 \times d} \\ \hline E_{\text{in}}^{\text{added}} \in \mathbb{R}^{16 \times d} \end{bmatrix} \in \mathbb{R}^{32016 \times d}, \quad W_{\text{out}}^{\text{new}} = \begin{bmatrix} W_{\text{out}}^{\text{old}} \in \mathbb{R}^{32000 \times d} \\ \hline W_{\text{out}}^{\text{added}} \in \mathbb{R}^{16 \times d} \end{bmatrix} \in \mathbb{R}^{32016 \times d}$$

Now, whenever the model computes logits $z_j = W_{\text{out}}^{\text{new}} h_t \in \mathbb{R}^{32016}$, coordinates $0 \dots 31999$ score regular subwords, and coordinates $32000 \dots 32014$ score the $15$ Self-RAG special tokens in the exact same vector-matrix multiplication.

---

## 3. Codebase Walkthrough: Loading Token IDs and Stripping Tokens (`retrieval_lm/utils.py`)

Let us walk through lines 43-95 of `reference/code/retrieval_lm/utils.py`, which handle special tokens during inference.

### 3.1 Mapping Token Strings to Integer IDs (`load_special_tokens`)

Lines 43-75 of `reference/code/retrieval_lm/utils.py` define the token name lists and the helper function `load_special_tokens`:

```python
rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                          "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                       "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens
```

What does `load_special_tokens` return when both `use_grounding=True` and `use_utility=True` are passed? It queries `tokenizer.convert_tokens_to_ids(token)` and returns four Python dictionaries mapping each token string to its integer ID in $\{32000, \dots, 32014\}$:
1. `ret_tokens`: `{"[No Retrieval]": 32000, "[Retrieval]": 32001, "[Continue to Use Evidence]": 32002}`
2. `rel_tokens`: `{"[Irrelevant]": 32003, "[Relevant]": 32004}`
3. `grd_tokens`: `{"[Fully supported]": 32012, "[Partially supported]": 32013, "[No support / Contradictory]": 32014}`
4. `ut_tokens`: `{"[Utility:1]": 32007, "[Utility:2]": 32008, "[Utility:3]": 32009, "[Utility:4]": 32010, "[Utility:5]": 32011}`

At inference time, when vLLM returns `pred_log_probs` (indexed by integer token IDs), the inference scripts loop over `rel_tokens.items()`, `grd_tokens.items()`, and `ut_tokens.items()` to look up the exact log-probability of each reflection token ID.

### 3.2 Stripping Special Tokens and Fixing Spacing (`postprocess` and `fix_spacing`)

Once Self-RAG has finished using the reflection tokens to score and select the best response, we want to strip all reflection tokens from the string before evaluating the text or showing it to the user. Lines 78-95 of `reference/code/retrieval_lm/utils.py` implement `fix_spacing` and `postprocess`:

```python
def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred
```

Let us trace how `postprocess` and `fix_spacing` work together on a concrete two-sentence output:
1. Suppose the raw model output across two segments is:
   `"[Relevant]First sentence.[Fully supported][No Retrieval]Second sentence.[Utility:5]</s>"`
2. `postprocess(pred)` deletes `[Relevant]`, `[Fully supported]`, `[No Retrieval]`, `[Utility:5]`, and `</s>`, leaving:
   `"First sentence.Second sentence."`
   Notice that because `[Fully supported][No Retrieval]` sat directly between the period `"."` of the first sentence and `"S"` of `"Second"`, deleting those tokens leaves `"e.S"` with no space after the period!
3. Calling `fix_spacing("First sentence.Second sentence.")` applies the regular expression `r'(?<=\w)([.!?])(?=\w)'`:
   - `(?<=\w)` checks that the character immediately before the punctuation mark is a word character (such as `"e"`).
   - `([.!?])` matches and captures a period, exclamation mark, or question mark.
   - `(?=\w)` checks that the character immediately after the punctuation mark is also a word character (such as `"S"`), meaning a space is missing.
   - `r'\1 '` replaces the punctuation mark with itself followed by a space, turning `"First sentence.Second sentence."` into `"First sentence. Second sentence."`.
4. Side effect: the regex fires on any period, exclamation mark, or question mark between two word characters, so it also splits decimals and abbreviations (`"3.14"` becomes `"3. 14"` and `"U.S."` becomes `"U. S."`). This can change string-match metrics on long-form outputs that contain such tokens.

---

## 4. Codebase Consistency Analysis: Three Subtle Discrepancies Across Scripts

Because we inspected every file in `reference/code/`, let us note three subtle inconsistencies in how special tokens are listed across different files in the repository.

### 4.1 Discrepancy 1: Swapped Variable Names in `postprocess_data.py` (Lines 13-18)

Compare lines 43-49 of `retrieval_lm/utils.py` against lines 13-18 of `data_creation/generator/postprocess_data.py`:

```python
# In data_creation/generator/postprocess_data.py (lines 13-18):
rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]"]
ground_tokens_names = ["[Utility:1]", "[Utility:2]",
                       "[Utility:3]", "[Utility:4]", "[Utility:5]"]
utility_tokens_names = ["[Fully supported]",
                        "[Partially supported]", "[No support / Contradictory]"]
```

In `postprocess_data.py`, the variable `ground_tokens_names` actually contains the `[Utility:1..5]` tokens, and `utility_tokens_names` actually contains the `[Fully supported]...` groundedness tokens!

Why does this swap not cause a bug in `postprocess_data.py`? Because in `postprocess_data.py`, those two variables are only used on line 23 where they are concatenated together to strip all special tokens (`special_tokens = rel_tokens_names + retrieval_tokens_names + ground_tokens_names + utility_tokens_names + other_special_tokens`). In `retrieval_lm/utils.py` (lines 43-49), where the lists are used individually by `load_special_tokens`, the variable names are assigned to the correct lists.

### 4.2 Discrepancy 2: `[Continue to Use Evidence]` Is Missing from `control_tokens` and `postprocess`

Count the tokens in `control_tokens` (line 52) and `special_tokens` inside `postprocess` (line 85) of `retrieval_lm/utils.py`:
- There are $3$ `IsSup` tokens $+ 2$ `Retrieve` tokens (`[No Retrieval]`, `[Retrieval]`) $+ 2$ `IsRel` tokens $+ 2$ `<paragraph>` tokens $+ 5$ `IsUse` tokens $= 14$ tokens.
- The $15$th token, `"[Continue to Use Evidence]"`, is missing from both `control_tokens` and `postprocess(pred)` (and is also missing from `retrieval_tokens_names` in `postprocess_data.py` line 14)!

Because `postprocess(pred)` does not strip `"[Continue to Use Evidence]"`, if the model outputs `"[Continue to Use Evidence]"` during long-form generation, `postprocess(pred)` leaves the literal string `"[Continue to Use Evidence]"` inside the text! That is why in `reference/code/retrieval_lm/run_long_form_static.py` (lines 423-424), the authors added an explicit `.replace` call after `postprocess` to clean it up. The replacement always inserts citation `[1]`, whichever passage the continued sentence actually relied on:

```python
item["output"] = fix_spacing(final_output)
item["output"] = item["output"].replace(
    ".[Continue to Use Evidence]", " [1]. ")
```

### 4.3 Discrepancy 3: `[PAD]` vs `<pad>` Between Critic and Generator Training

In `reference/code/data_creation/train_special_tokens.py` (line 30), the padding token string is defined as:

```python
DEFAULT_PAD_TOKEN = "[PAD]"
```

whereas in `reference/code/retrieval_lm/finetune.py` (line 454), the padding token string is defined as:

```python
special_token_dict["pad_token"] = "<pad>"
```

In both scripts, the padding token is added as the $16$th new token at index $32015$, and both scripts reference it dynamically via `tokenizer.pad_token_id` when building `attention_mask` and padding batches, so the difference in string spelling (`"[PAD]"` vs `"<pad>"`) does not affect model execution.
