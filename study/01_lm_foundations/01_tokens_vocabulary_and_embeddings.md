# Tokens, Vocabulary, and Embeddings

## 1. Plain-English Intuition: Why Neural Networks Need Tokens and Vectors

A language model takes a piece of text as input (such as `"What is the capital of France?"`) and predicts what text should come next (such as `"Paris"`).

However, a neural network is purely a mathematical function made of matrix multiplications and additions. It can only process arrays of floating-point numbers (such as `[0.12, -0.85, 1.04]`). It cannot directly multiply or add raw text strings like `"France"`.

To bridge the gap between human text and matrix math, we have to solve two basic problems:

1. How do we break a text string into discrete units and assign each unit an integer ID?
   Suppose we tried assigning a unique integer ID to every complete word in English (`"apple" -> 1`, `"banana" -> 2`, and so on). This whole-word dictionary approach quickly breaks down:
   - If a user types a rare proper noun, a scientific term, or a slight spelling variation that was not in the dictionary, the model has no ID for it and sees only an "unknown word" placeholder.
   - Related word forms like `"retrieve"`, `"retrieves"`, `"retrieved"`, and `"retrieving"` would be treated as four completely unrelated integer IDs, forcing the model to learn the meaning of the root `"retriev"` four separate times.

   Modern language models solve this using subword tokenization. A tokenizer breaks text into recurring character chunks called tokens. Frequent words (like `"the"` or `"Paris"`) remain intact as single tokens, while rarer or longer words (like `"retrieving"`) are split into smaller, reusable pieces (such as `"retriev"` and `"ing"`). Even if the model encounters a brand-new word it has never seen before, it can break that word down into familiar subword pieces or individual characters.

2. Why can we not feed integer token IDs directly into a neural network equation?
   Suppose our vocabulary assigns `"cat" -> 50`, `"table" -> 51`, and `"kitten" -> 8400`. If we fed those raw integers $x \in \{50, 51, 8400\}$ into a linear equation $y = w \cdot x + b$, the math would treat `"cat"` ($50$) and `"table"` ($51$) as nearly identical numbers, while treating `"cat"` ($50$) and `"kitten"` ($8400$) as wildly different! Integer IDs are arbitrary catalog numbers, not meaningful magnitudes.

   To give each token a rich, flexible representation, we use an input embedding lookup table. Think of this table as a spreadsheet with one row per token in our vocabulary and $d$ columns (for example, $d = 4096$ columns in Llama-2 7B). When the model sees token ID $50$ (`"cat"`), it does not do arithmetic on the number $50$; instead, it looks up row $50$ of the spreadsheet and retrieves a vector of $4096$ floating-point numbers. During training, the model adjusts the numbers in each row so that tokens with similar meanings or grammatical roles end up with similar vectors.

---

## 2. Subword Tokenization and the Vocabulary $\mathcal{V}$

### 2.1 How Llama-2 Tokenizes Text

Self-RAG builds on the Llama-2 family of language models, which uses a SentencePiece Byte-Pair Encoding (BPE) tokenizer.

Byte-Pair Encoding constructs its vocabulary before model training begins by starting with individual characters and repeatedly merging the most frequently co-occurring pair of adjacent pieces in a training corpus until the vocabulary reaches a target size $V$.

To ensure that converting text to tokens and back to text is completely lossless (even with multiple spaces or punctuation), SentencePiece treats whitespace as a normal character represented by the visible symbol `▁` (Unicode U+2581, lower one-eighth block). For example, the string:

`"The sun is setting."`

is split by the Llama-2 tokenizer into subword tokens as:

`["<s>", "▁The", "▁sun", "▁is", "▁setting", "."]`

Notice two details in this token list:
- Spaces before words are attached to the front of the word token (`"▁sun"`, `"▁is"`), while the period `"."` has no leading space and is its own token.
- A special beginning-of-sequence token `"<s>"` is automatically placed at index $0$ to mark the start of the text.

### 2.2 Formal Definition of the Vocabulary and Boundary Tokens

Let $\mathcal{V}$ denote the complete, fixed set of tokens recognized by the tokenizer, and let $V = \vert \mathcal{V} \vert$ denote the total number of tokens in $\mathcal{V}$. Each token $v \in \mathcal{V}$ is paired one-to-one with an integer index in $\{0, 1, \dots, V - 1\}$.

In the base Llama-2 model (before Self-RAG adds its reflection tokens), the vocabulary size is:

$$V = \vert \mathcal{V} \vert = 32000$$

Within $\mathcal{V}$, four structural tokens handle sequence boundaries, unknown bytes, and batch alignment (configured in `reference/code/data_creation/train_special_tokens.py` lines 28-31 and `reference/code/retrieval_lm/finetune.py` lines 381-384):

| Token String | Role Name in Code | Integer ID in Base Llama-2 | Plain-English Purpose |
|---|---|---|---|
| `<unk>` | `unk_token` | $0$ | Fallback token used only if an unrecognizable character byte is encountered. |
| `<s>` | `bos_token` | $1$ | Beginning-of-sequence marker placed at the start of every tokenized prompt. |
| `</s>` | `eos_token` | $2$ | End-of-sequence marker placed at the end of every target response. When the model outputs `</s>` at test time, generation stops. |
| `[PAD]` or `<pad>` | `pad_token` | $32000$ (or $32015$ with reflection tokens) | Filler token appended to shorter sequences when grouping multiple sequences of different lengths into a rectangular batch. |

Given any text string of length $L$ tokens, the tokenizer converts it into an ordered tuple of $L$ integer token IDs:

$$u = (u_1, u_2, \dots, u_L), \quad \text{where } u_k \in \{0, 1, \dots, V - 1\}$$

where:
- $L \in \mathbb{Z}^+$ is the number of tokens in the sequence.
- $k \in \{1, 2, \dots, L\}$ is the position index along the sequence from left to right.
- $u_k$ is the integer vocabulary ID of the $k$-th token in the sequence.

---

## 3. Step-by-Step Build-Up: Input Embedding Lookup

### 3.1 Toy Numerical Scenario ($V = 6, d = 3$)

Before writing the general matrix equations for $V = 32000$ and $d = 4096$, let us trace a tiny toy vocabulary $\mathcal{V}_{\text{toy}}$ with only $V = 6$ tokens, where each token is represented by a vector of $d = 3$ numbers.

Suppose our toy embedding lookup table $E_{\text{in}} \in \mathbb{R}^{6 \times 3}$ contains the following learned numbers:

| Row Index (Token ID $v$) | Token String | Column $1$ ($c=1$) | Column $2$ ($c=2$) | Column $3$ ($c=3$) | Row Vector $E_{\text{in}}[v, :] \in \mathbb{R}^3$ |
|---|---|---|---|---|---|
| $0$ | `<unk>` | $0.0$ | $0.0$ | $0.0$ | $[0.0,\; 0.0,\; 0.0]$ |
| $1$ | `<s>` | $1.0$ | $0.0$ | $-1.0$ | $[1.0,\; 0.0,\; -1.0]$ |
| $2$ | `</s>` | $-1.0$ | $0.0$ | $1.0$ | $[-1.0,\; 0.0,\; 1.0]$ |
| $3$ | `<pad>` | $0.0$ | $0.0$ | $0.0$ | $[0.0,\; 0.0,\; 0.0]$ |
| $4$ | `▁Sun` | $0.8$ | $0.5$ | $0.2$ | $[0.8,\; 0.5,\; 0.2]$ |
| $5$ | `▁sets` | $0.1$ | $-0.4$ | $0.9$ | $[0.1,\; -0.4,\; 0.9]$ |

Now suppose we want to process the $L = 4$ token sequence:

$$\text{Tokens} = (\text{``<s>''}, \text{``▁Sun''}, \text{``▁sets''}, \text{``</s>''}) \quad \longrightarrow \quad u = (1, 4, 5, 2)$$

To convert the integer sequence $u = (1, 4, 5, 2)$ into continuous vectors, we simply grab row $1$, row $4$, row $5$, and row $2$ from $E_{\text{in}}$ and stack them on top of each other:
- Position $k = 1$ has token ID $u_1 = 1$ (`<s>`), which selects row $1$: $e_1 = [1.0,\; 0.0,\; -1.0]$.
- Position $k = 2$ has token ID $u_2 = 4$ (`▁Sun`), which selects row $4$: $e_2 = [0.8,\; 0.5,\; 0.2]$.
- Position $k = 3$ has token ID $u_3 = 5$ (`▁sets`), which selects row $5$: $e_3 = [0.1,\; -0.4,\; 0.9]$.
- Position $k = 4$ has token ID $u_4 = 2$ (`</s>`), which selects row $2$: $e_4 = [-1.0,\; 0.0,\; 1.0]$.

Stacking these $L = 4$ row vectors produces a 2D matrix $H^{(0)} \in \mathbb{R}^{4 \times 3}$:

$$H^{(0)} = \begin{bmatrix} e_1 \\ e_2 \\ e_3 \\ e_4 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 & -1.0 \\ 0.8 & 0.5 & 0.2 \\ 0.1 & -0.4 & 0.9 \\ -1.0 & 0.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{4 \times 3}$$

### 3.2 General Mathematical Formulation of Embedding Lookup

In a full language model, let $E_{\text{in}} \in \mathbb{R}^{V \times d}$ denote the learnable input embedding matrix, where:
- $V \in \mathbb{Z}^+$ is the vocabulary size (number of rows).
- $d \in \mathbb{Z}^+$ is the hidden embedding dimension (number of columns; $d = 4096$ in Llama-2 7B and $d = 5120$ in Llama-2 13B).

For any token ID $u_k \in \{0, 1, \dots, V - 1\}$ at sequence position $k \in \{1, \dots, L\}$, its embedding vector $e_k \in \mathbb{R}^d$ is the $u_k$-th row of $E_{\text{in}}$:

$$e_k = E_{\text{in}}[u_k, :] \in \mathbb{R}^d$$

Mathematically, selecting row $u_k$ is equivalent to multiplying $E_{\text{in}}$ on the left by a one-hot row vector $\mathbf{o}(u_k) \in \{0, 1\}^V$, whose $v$-th entry is $1$ if $v = u_k$ and $0$ otherwise:

$$e_k = \mathbf{o}(u_k) E_{\text{in}} = \sum_{v=0}^{V-1} \mathbb{I}[v = u_k] \, E_{\text{in}}[v, :] = E_{\text{in}}[u_k, :]$$

where:
- $\mathbf{o}(u_k) \in \{0, 1\}^V$ is the one-hot indicator vector for token ID $u_k$.
- $\mathbb{I}[v = u_k] \in \{0, 1\}$ equals $1$ when $v = u_k$ and $0$ for all other $V - 1$ vocabulary items.

In our toy example for $u_2 = 4$ (`▁Sun`), the one-hot vector is $\mathbf{o}(4) = [0, 0, 0, 0, 1, 0]$, and multiplying $\mathbf{o}(4) E_{\text{in}}$ multiplies row $4$ by $1$ and all other rows by $0$, returning $[0.8, 0.5, 0.2]$.

Why is the one-hot equation useful to know if PyTorch (`torch.nn.Embedding`) just indexes row $u_k$ directly in $O(1)$ time? Because during training, the one-hot sum shows how gradients flow back into the embedding table: if token $v$ does not appear in a training batch ($\mathbb{I}[v = u_k] = 0$ for all $k$), the gradient of the loss with respect to row $E_{\text{in}}[v, :]$ is zero, so only the rows of tokens actually present in the batch get updated.

For a full sequence $u = (u_1, \dots, u_L)$, stacking the $L$ row vectors yields the initial hidden state matrix $H^{(0)}$ (superscript $(0)$ means layer $0$, before passing through any transformer layers):

$$H^{(0)} = \begin{bmatrix} E_{\text{in}}[u_1, :] \\ E_{\text{in}}[u_2, :] \\ \vdots \\ E_{\text{in}}[u_L, :] \end{bmatrix} \in \mathbb{R}^{L \times d}$$

---

## 4. Batching Variable-Length Sequences: Padding and `attention_mask`

### 4.1 Why Batching Requires Padding

To utilize GPU parallelism, we process a batch of $B$ sequences simultaneously rather than one sequence at a time. However, GPU tensors must be rectangular: every row in a 2D matrix of token IDs must have the exact same number of columns $L_{\max}$.

Suppose we have a batch of $B = 2$ sequences using our toy vocabulary $\mathcal{V}_{\text{toy}}$:
- Sequence 1 (length $L_1 = 4$): `["<s>", "▁Sun", "▁sets", "</s>"]` $\to$ IDs `(1, 4, 5, 2)`
- Sequence 2 (length $L_2 = 3$): `["<s>", "▁Sun", "</s>"]` $\to$ IDs `(1, 4, 2)`

Because $L_1 = 4$ and $L_2 = 3$, the maximum length in this batch is $L_{\max} = \max(4, 3) = 4$. To make Sequence 2 have length $4$, we pad it with $L_{\max} - L_2 = 1$ copy of the `<pad>` token (ID $3$).

This creates a rectangular 2D integer tensor `input_ids` of shape $[B, L_{\max}] = [2, 4]$:

$$\text{input\_ids} = \begin{bmatrix} 1 & 4 & 5 & 2 \\ 1 & 4 & 2 & 3 \end{bmatrix} \in \{0, \dots, V-1\}^{2 \times 4}$$

Looking up every entry of `input_ids` in $E_{\text{in}} \in \mathbb{R}^{6 \times 3}$ produces a 3D floating-point tensor $H^{(0)}$ of shape $[B, L_{\max}, d] = [2, 4, 3]$.

### 4.2 What `attention_mask` Tracks

Once `input_ids` is padded, entry $(b=2, k=4)$ contains the integer $3$ (`<pad>`), which looks just like any other integer token ID to the embedding layer. If later transformer layers treated that `<pad>` position as a real word, the dummy vector at position $4$ would mix into the representation of Sequence 2 and change the model's predictions.

To tell every downstream layer which positions are real tokens and which positions are dummy padding, we construct a binary matrix called `attention_mask` of the exact same shape $[B, L_{\max}]$:

$$M_{b, k} = \mathbb{I}\left[\text{input\_ids}[b, k] \neq u_{\text{pad}}\right] = \begin{cases} 1 & \text{if position } k \text{ in batch item } b \text{ is a real token} \\ 0 & \text{if position } k \text{ in batch item } b \text{ is a padding token } (u_{\text{pad}}) \end{cases}$$

where:
- $b \in \{1, \dots, B\}$ indexes the example in the batch.
- $k \in \{1, \dots, L_{\max}\}$ indexes the token position along the padded sequence.
- $u_{\text{pad}} \in \{0, \dots, V-1\}$ is the integer ID of the padding token (`tokenizer.pad_token_id`, which is $3$ in our toy table).
- $M_{b, k} \in \{0, 1\}$ is the mask value at batch index $b$ and token position $k$.

For our $B = 2, L_{\max} = 4$ toy batch with $u_{\text{pad}} = 3$, the alignment across `input_ids`, `attention_mask`, and real sequence length is:

| Batch Item $b$ | Position $k=1$ | Position $k=2$ | Position $k=3$ | Position $k=4$ | Real Token Count $\sum_{k=1}^{L_{\max}} M_{b,k}$ |
|---|---|---|---|---|---|
| $b=1$ (`input_ids`) | $1$ (`<s>`) | $4$ (`▁Sun`) | $5$ (`▁sets`) | $2$ (`</s>`) | $4$ tokens |
| $b=1$ (`attention_mask`) | $1$ | $1$ | $1$ | $1$ | $1 + 1 + 1 + 1 = 4$ |
| $b=2$ (`input_ids`) | $1$ (`<s>`) | $4$ (`▁Sun`) | $2$ (`</s>`) | $3$ (`<pad>`) | $3$ tokens |
| $b=2$ (`attention_mask`) | $1$ | $1$ | $1$ | $0$ | $1 + 1 + 1 + 0 = 3$ |

Notice that summing the ones across row $b$ of `attention_mask` ($\sum_{k=1}^{L_{\max}} M_{b,k}$) recovers the exact unpadded token length $L_b$ of sequence $b$.

### 4.3 Right-Padding During Training vs Left-Padding During Inference

In the Self-RAG codebase, you will see `padding_side="right"` in training scripts and `padding_side="left"` in inference scripts:
- Training (`reference/code/data_creation/train_special_tokens.py` line 295): sets `padding_side="right"`. Because training feeds the complete prompt + response sequence at once, placing `<pad>` tokens at the right end (as in $[1, 4, 2, 3]$ above) keeps token indices $0, 1, \dots, L_b - 1$ aligned starting from column $0$.
- Inference (`reference/code/retrieval_lm/run_short_form.py` line 253 and `reference/code/retrieval_lm/run_long_form_static.py` line 285): sets `padding_side="left"` (`AutoTokenizer.from_pretrained(args.model_name, padding_side="left")`). During batched generation, the model generates the next token immediately after the rightmost column $L_{\max} - 1$ of the prompt batch. Left-padding (`[3, 1, 4, 2]`) ensures that the final real token of every prompt in the batch sits at the rightmost column $L_{\max} - 1$.

---

## 5. Paper and Codebase Grounding

### 5.1 Tokenization and Length Counting in `train_special_tokens.py` and `finetune.py`

Let us inspect how `reference/code/data_creation/train_special_tokens.py` (lines 165-186) tokenizes strings and counts their real (unpadded) lengths:

```python
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
```

Tracing each line of `_tokenize_fn`:
1. `tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)` converts a single string `text` into a PyTorch tensor (`"pt"`) of integer token IDs of shape $[1, L]$, truncating the sequence if it exceeds `model_max_length` (for example, $2048$ tokens).
2. `tokenized.input_ids[0]` extracts the 1D tensor of shape $[L]$ for that string.
3. `tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()` computes the exact unpadded length $\sum_{k=1}^L M_{b,k}$ derived in Section 4.2:
   - `.ne(tokenizer.pad_token_id)` stands for "not equal" ($\neq u_{\text{pad}}$), producing a boolean tensor of `True` ($1$) for real tokens and `False` ($0$) for pad tokens.
   - `.sum().item()` adds up the `True` values and converts the result to a standard Python integer.
   - The exact same length-counting expression appears in `reference/code/retrieval_lm/finetune.py` on line 231 (`input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()`).

### 5.2 Dynamic Batch Padding in `DataCollatorForSupervisedDataset`

Because `_tokenize_fn` tokenizes one string at a time inside a list comprehension, each element in `self.input_ids` is a 1D tensor of a different length $L_b$. When PyTorch's `DataLoader` groups a batch of `instances` together during training, `DataCollatorForSupervisedDataset` in `reference/code/data_creation/train_special_tokens.py` (lines 256-272) pads them to the longest sequence in that specific mini-batch:

```python
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
```

Tracing this collator step by step:
1. `input_ids` starts as a Python list of $B$ 1D tensors with varying lengths $[L_1], [L_2], \dots, [L_B]$.
2. `torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)` finds $L_{\max} = \max_b L_b$ across the batch, right-pads every shorter tensor with `pad_token_id`, and stacks them into a 2D tensor of shape $[B, L_{\max}]$ (`batch_first=True` puts the batch dimension $B$ first).
3. `labels` is padded to the exact same shape $[B, L_{\max}]$, but its padding positions are filled with `IGNORE_INDEX` (`-100`) instead of `pad_token_id` so the loss function will ignore all padded positions (which we will derive in full in Part 3).
4. `attention_mask=input_ids.ne(self.tokenizer.pad_token_id)` constructs the $[B, L_{\max}]$ boolean mask tensor where real tokens are `True` ($1$) and padded positions are `False` ($0$).

### 5.3 Registering Special Tokens and Resizing the Embedding Table

Base Llama-2 models come with `bos_token` (`<s>`), `eos_token` (`</s>`), and `unk_token` (`<unk>`), totaling $V = 32000$ tokens, but do not define a `pad_token` by default. Furthermore, as stated in Section 3.1 of the Self-RAG paper, Self-RAG expands the vocabulary $\mathcal{V}$ with reflection tokens (`[No Retrieval]`, `[Retrieval]`, `[Relevant]`, and so on) so the model can predict them just like regular words.

In `reference/code/retrieval_lm/finetune.py` (lines 379-402), the script adds these tokens to the tokenizer and resizes the model's embedding matrix $E_{\text{in}}$ to match the new vocabulary size:

```python
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        if args.use_special_tokens is True:
            special_token_dict = {"additional_special_tokens": ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]}
        special_token_dict["bos_token"] = "<s>"
        special_token_dict["eos_token"] = "</s>"
        special_token_dict["unk_token"] = "<unk>"
        special_token_dict["pad_token"] = "<pad>"
        num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
...
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
```

Look at what happens to the shapes here:
- Before `tokenizer.add_special_tokens(special_token_dict)`, `len(tokenizer)` is $32000$, and `model.get_input_embeddings().weight` ($E_{\text{in}}$) has shape $[32000, d]$ (`embedding_size == 32000`).
- Adding the 15 Self-RAG special tokens plus `<pad>` adds $16$ new entries to $\mathcal{V}$, increasing `len(tokenizer)` from $32000$ to $32016$ (`num_added_tokens == 16`).
- If we tried to look up token ID $32005$ in a $[32000, d]$ matrix, PyTorch would crash with an index-out-of-bounds error. Calling `model.resize_token_embeddings(len(tokenizer))` allocates $16$ new rows at the bottom of the embedding table (indices $32000$ through $32015$), expanding $E_{\text{in}}$ to shape $[32016, d]$ while keeping the first $32000$ pre-trained rows intact.
