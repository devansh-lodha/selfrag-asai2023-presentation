# Dense Retrieval Intuition and the Contriever Encoder

## 1. Plain-English Intuition: How Do We Search 21 Million Passages by Meaning?

In Self-RAG, whenever the Generator $\mathcal{M}$ outputs the `[Retrieval]` token, the Retriever $\mathcal{R}$ must search an external collection $\mathcal{C}_{\text{corpus}}$ of $N \approx 21\text{ million}$ Wikipedia paragraphs and return the top $K$ most relevant passages in milliseconds.

How can a computer measure whether a question $q$ and a Wikipedia paragraph $d$ are about the same topic?

1. Why simple keyword matching (BM25 / TF-IDF) is not enough:
   Traditional search engines count how many exact words overlap between the query $q$ and the passage $d$. However, human language uses many different words to express the same idea (synonyms and paraphrases). If a query asks `"How did US states get their names?"` and a relevant passage says `"The etymology of American commonwealths traces back to indigenous languages and European monarchs"`, they share almost no words! This is called the vocabulary mismatch problem. Conversely, as we saw in Figure 1 of the paper, a passage saying `"Popular names by states. In Texas, Emma is a popular baby name"` shares the keywords `"states"` and `"names"` while being completely off-topic.

2. Why feeding $(q, d)$ together into one transformer (a Cross-Encoder) is too slow:
   If we concatenated the question $q$ and passage $d$ into a single sequence and ran a transformer over $(q, d)$ to score their semantic match, we would have to run $21\text{ million}$ transformer forward passes for every single user question. Even on fast GPUs, that would take hours per question.

3. How a Dense Bi-Encoder (Contriever) solves both problems:
   A dense bi-encoder uses a neural network encoder $E(\cdot)$ that maps any text string independently into a single fixed-length vector of $D = 768$ floating-point numbers:
   - Offline (once, before any user asks a question): We run the encoder $E(d_i)$ on all $21\text{ million}$ Wikipedia passages and store their $768$-dimensional vectors $\mathbf{d}_1, \mathbf{d}_2, \dots, \mathbf{d}_N \in \mathbb{R}^{768}$ on disk.
   - Online (at test time, when a user asks query $q$): We run the encoder $E(q)$ only once on the query $q$ to get a single query vector $\mathbf{q} \in \mathbb{R}^{768}$.
   - We then score every passage $d_i$ by computing the dot product (inner product) between $\mathbf{q}$ and the pre-computed vector $\mathbf{d}_i$:

$$s(q, d_i) = \langle E(q), E(d_i) \rangle = \langle \mathbf{q}, \mathbf{d}_i \rangle = \sum_{c=1}^{768} q_c \, d_{i,c}$$

Because the encoder is trained so that texts with similar meanings point in the same direction in $\mathbb{R}^{768}$, passages that match the meaning of $q$ will have a large positive dot product $\langle \mathbf{q}, \mathbf{d}_i \rangle$ even if they use different words. And because all $21\text{ million}$ passage vectors $\mathbf{d}_i$ are pre-computed offline, finding the top-$K$ dot products at test time requires only fast vector arithmetic.

---

## 2. Step-by-Step Build-Up: Bidirectional Encoding and Masked Average Pooling

### 2.1 Bidirectional Encoder (`BertModel`) vs Causal Decoder (`LlamaModel`)

In Module 01, we studied Llama-2, which is a causal decoder: token $t$ can only attend to tokens $1 \dots t$ on its left because it is trained to generate text left-to-right.

By contrast, the default retriever in Self-RAG is `Contriever` (`facebook/contriever-msmarco`, implemented in `reference/code/retrieval_lm/src/contriever.py`), which is built on `BertModel` (or `XLMRobertaModel` for multilingual retrieval). BERT is a bidirectional transformer encoder:
- When encoding a query $q$ or a passage $d = (u_1, u_2, \dots, u_L)$, the entire text string is already complete; we are not predicting the next token.
- Therefore, there is no causal triangular mask. In every transformer layer of BERT, token position $k$ attends to all real tokens $1, 2, \dots, L$ on both its left and its right simultaneously.
- After passing a batch of $B$ sequences of padded length $L_{\max}$ through BERT's $12$ encoder layers, BERT outputs `last_hidden_state`, a 3D tensor $H \in \mathbb{R}^{B \times L_{\max} \times D}$ where $D = 768$.

### 2.2 The Pooling Problem: Why We Must Zero Out `<pad>` Vectors Before Averaging

Look at the shape of BERT's output `last_hidden_state`: for each sequence $b \in \{1, \dots, B\}$, we have a 2D matrix $H_b \in \mathbb{R}^{L_{\max} \times 768}$ containing $L_{\max}$ separate token vectors (one $768$-number vector for each token position $k \in \{1, \dots, L_{\max}\}$).

However, to represent the whole passage as a single point in $\mathbb{R}^{768}$, we must compress those $L_{\max}$ token vectors into one single vector $\mathbf{e}_b \in \mathbb{R}^{768}$. This compression step is called pooling.

Contriever uses masked average pooling (`pooling="average"`): it computes the average of the token vectors across all real (non-padding) tokens in the sequence.

Why can we not simply call `last_hidden.mean(dim=1)` across all $L_{\max}$ positions?
Because in a batch of variable-length sequences, shorter sequences are padded with `<pad>` tokens (where the `attention_mask` is $M_{b,k} = 0$). Even if a `<pad>` token starts with a zero embedding, after passing through $12$ BERT layers (which add positional embeddings, bias vectors, and LayerNorm shifts), the output vector $H_{b,k} \in \mathbb{R}^{768}$ at a `<pad>` position is non-zero! Averaging without masking would:
1. Add non-zero garbage numbers from `<pad>` positions into the sum.
2. Divide the sum by the padded length $L_{\max}$ instead of the true number of real words $L_b = \sum_{k=1}^{L_{\max}} M_{b,k}$.

### 2.3 Worked 4-Token Toy Numerical Example of Masked Average Pooling ($L_{\max} = 4, D = 2$)

Suppose we have a single sequence $b$ in a batch where $L_{\max} = 4$ and hidden dimension $D = 2$. Suppose the sequence has $3$ real tokens ($M_{b,1} = 1, M_{b,2} = 1, M_{b,3} = 1$) followed by $1$ `<pad>` token ($M_{b,4} = 0$).

Suppose BERT outputs the following 2D hidden vectors $H_{b,k} \in \mathbb{R}^2$ at the four positions:

| Position $k$ | Token Type | `attention_mask` $M_{b,k}$ | BERT Output Vector $H_{b,k} \in \mathbb{R}^2$ | Masked Vector $M_{b,k} H_{b,k} \in \mathbb{R}^2$ |
|---|---|---|---|---|
| $k = 1$ | Real word $1$ | $1$ | $[2.0,\; 4.0]$ | $1 \times [2.0,\; 4.0] = [2.0,\; 4.0]$ |
| $k = 2$ | Real word $2$ | $1$ | $[4.0,\; 2.0]$ | $1 \times [4.0,\; 2.0] = [4.0,\; 2.0]$ |
| $k = 3$ | Real word $3$ | $1$ | $[0.0,\; 6.0]$ | $1 \times [0.0,\; 6.0] = [0.0,\; 6.0]$ |
| $k = 4$ | `<pad>` token | $0$ | $[50.0,\; -30.0]$ | $0 \times [50.0,\; -30.0] = [0.0,\; 0.0]$ |
| Sum over $k = 1 \dots 4$ | | $\sum M_{b,k} = 3$ | $[56.0,\; -18.0]$ (Corrupted!) | $[6.0,\; 12.0]$ (Clean!) |

Now compare unmasked averaging against masked average pooling:
- If we forgot to mask row $k = 4$, summing all $4$ rows gives $[56.0,\; -18.0]$ and dividing by $4$ gives $[14.0,\; -4.5]$, which is completely ruined by the `<pad>` vector $[50.0,\; -30.0]$.
- With masked average pooling:
  1. Step 1: Multiply each row $H_{b,k}$ by its mask $M_{b,k} \in \{0, 1\}$, turning the `<pad>` row at $k = 4$ into $[0.0,\; 0.0]$.
  2. Step 2: Sum the masked vectors over $k = 1 \dots 4$, giving $[2.0 + 4.0 + 0.0 + 0.0,\; 4.0 + 2.0 + 6.0 + 0.0] = [6.0,\; 12.0]$.
  3. Step 3: Divide the sum $[6.0,\; 12.0]$ by the sum of the mask $\sum_{k=1}^4 M_{b,k} = 1 + 1 + 1 + 0 = 3$:

$$\mathbf{e}_b = \frac{[6.0,\; 12.0]}{3} = [2.0,\; 4.0] \in \mathbb{R}^2$$

### 2.4 General Equations for Masked Average Pooling and $\ell_2$ Normalization

For a batch of $B$ sequences with hidden states $H \in \mathbb{R}^{B \times L_{\max} \times D}$ and binary attention mask $M \in \{0, 1\}^{B \times L_{\max}}$, masked average pooling computes the pooled embedding $\mathbf{e}_b \in \mathbb{R}^D$ for each sequence $b \in \{1, \dots, B\}$ as:

$$\mathbf{e}_b = \frac{\sum_{k=1}^{L_{\max}} M_{b,k} \, H_{b,k,:}}{\sum_{k=1}^{L_{\max}} M_{b,k}} \in \mathbb{R}^D$$

Alternatively, if `pooling == "cls"` is selected, the model simply takes the vector at the first token position ($k = 1$, index $0$ in Python):

$$\mathbf{e}_b^{\text{cls}} = H_{b, 1, :} \in \mathbb{R}^D$$

Finally, if the boolean flag `normalize=True` is passed, $\mathbf{e}_b$ is divided by its Euclidean ($\ell_2$) norm so that the resulting vector has length $1$:

$$\hat{\mathbf{e}}_b = \frac{\mathbf{e}_b}{\Vert \mathbf{e}_b \Vert_2} = \frac{\mathbf{e}_b}{\sqrt{\sum_{c=1}^{D} e_{b,c}^2}} \in \mathbb{R}^D$$

where:
- $H_{b,k,:} \in \mathbb{R}^D$ is the final-layer BERT hidden vector for batch item $b$ at token position $k$ ($D = 768$).
- $M_{b,k} \in \{0, 1\}$ is $1$ for real tokens and $0$ for padding tokens.
- $\sum_{k=1}^{L_{\max}} M_{b,k} \in \mathbb{Z}^+$ is the number of real tokens in sequence $b$.
- $\Vert \mathbf{e}_b \Vert_2 \in (0, \infty)$ is the Euclidean length of $\mathbf{e}_b$. Note that in `facebook/contriever-msmarco` inference (`generate_passage_embeddings.py` and `passage_retrieval.py`), `normalize` defaults to `False`, so raw unnormalized dot products $\langle \mathbf{e}_q, \mathbf{e}_d \rangle$ are used.

---

## 3. Line-by-Line Code Walkthrough: `src/contriever.py` and `src/utils.py`

### 3.1 `Contriever` and `XLMRetriever` Classes (`src/contriever.py`, Lines 11-98)

Let us examine `Contriever` in `reference/code/retrieval_lm/src/contriever.py` (lines 11-53; `XLMRetriever` on lines 56-98 is identical except it inherits from `XLMRobertaModel` instead of `BertModel`):

```python
class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb
```

Look closely at the tensor shapes in lines 43-47:
1. `last_hidden = model_output["last_hidden_state"]` has shape $[B, L_{\max}, 768]$, while `attention_mask` has 2D shape $[B, L_{\max}]$.
2. `attention_mask[..., None].bool()` uses `[..., None]` to insert a new dimension of size $1$ at the end, changing the shape of the mask from $[B, L_{\max}]$ to $[B, L_{\max}, 1]$, and converts $1 \to \text{True}$ (real token) and $0 \to \text{False}$ (pad token).
3. The bitwise negation operator `~` flips `True` $\leftrightarrow$ `False`, so `~attention_mask[..., None].bool()` is `True` at padding positions and `False` at real token positions.
4. `last_hidden.masked_fill(..., 0.0)` broadcasts the $[B, L_{\max}, 1]$ boolean mask across all $768$ channels and overwrites every padding position with `0.0` (Step 1 of our toy example).
5. `last_hidden.sum(dim=1)` sums along the sequence length dimension (`dim=1`), reducing $[B, L_{\max}, 768]$ to $[B, 768]$ (Step 2 of our toy example).
6. `attention_mask.sum(dim=1)[..., None]` sums the $1$s along `dim=1` to get the real token counts of shape $[B]$, and appends `[..., None]` to make it shape $[B, 1]$ so PyTorch can divide each row of the $[B, 768]$ sum by its scalar token count (Step 3 of our toy example).

### 3.2 Loading Checkpoints and Stripping Wrapper Prefixes (`load_retriever`, Lines 101-138)

Lines 101-138 of `src/contriever.py` define `load_retriever(model_path, pooling="average", random_init=False)`:

```python
def load_retriever(model_path, pooling="average", random_init=False):
    # try: check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # retriever_model_id = "bert-base-uncased"
            retriever_model_id = "bert-base-multilingual-cased"
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
        if "xlm" in retriever_model_id:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        retriever = model_class(cfg)
        pretrained_dict = pretrained_dict["model"]

        if any("encoder_q." in key for key in pretrained_dict.keys()):  # test if model is defined with moco class
            pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q." in k}
        elif any("encoder." in key for key in pretrained_dict.keys()):  # test if model is defined with inbatch class
            pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if "encoder." in k}
        retriever.load_state_dict(pretrained_dict, strict=False)
    else:
        retriever_model_id = model_path
        if "xlm" in retriever_model_id:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        cfg = utils.load_hf(transformers.AutoConfig, model_path)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
        retriever = utils.load_hf(model_class, model_path)

    return retriever, tokenizer, retriever_model_id
```

Notice how `load_retriever` handles two different ways a Contriever model might be stored:
1. If `model_path` is a Hugging Face Hub identifier (such as `"facebook/contriever-msmarco"` used in Self-RAG), `os.path.exists(path)` is `False`, so it executes the `else:` branch (lines 126-135) and loads the config, tokenizer, and `Contriever` weights directly via Hugging Face's `from_pretrained`.
2. If `model_path` is a local directory containing a custom Contriever training checkpoint `checkpoint.pth`, the model inside `checkpoint.pth` was wrapped inside either the `MoCo` training class (which names its query encoder `self.encoder_q` and its momentum key encoder `self.encoder_k`) or the `InBatch` training class (which names its encoder `self.encoder`). Lines 120-123 filter for `"encoder_q."` (or `"encoder."`) and strip that prefix string from the dictionary keys so the weights match the bare `Contriever` class.

### 3.3 Training Schedulers and Distributed Statistics in `src/utils.py`

`reference/code/retrieval_lm/src/utils.py` provides checkpointing, learning rate schedulers, and metric averaging utilities for Contriever:
- `load_hf(object_class, model_name)` (lines 187-192): tries calling `object_class.from_pretrained(model_name, local_files_only=True)` first to avoid network requests if the model is already cached on disk, and falls back to `local_files_only=False` if not cached.
- `WarmupLinearScheduler` and `CosineScheduler` (lines 84-112): subclass `torch.optim.lr_scheduler.LambdaLR` to compute a step multiplier $\lambda(s) \in [0, 1]$ at optimization step $s$ given warmup steps $W$ (`self.warmup`), total steps $S$ (`self.total`), and minimum ratio $r_{\min}$ (`self.ratio`):
  - In `CosineScheduler`, during warmup ($s < W$), $\lambda(s) = \frac{s}{W}$; after warmup ($s \ge W$), it follows a quarter-period cosine curve (the angle runs from $0$ to $\frac{\pi}{2}$) from $1.0$ down to $r_{\min}$:

$$\lambda_{\text{cosine}}(s) = r_{\min} + (1 - r_{\min}) \cos\left(\frac{\pi}{2} \cdot \frac{s - W}{S - W}\right)$$

- `WeightedAvgStats` (lines 145-184): accumulates batch metrics weighted by batch size (`self.raw_stats[key] += value * weight`, `self.total_weights[key] += weight`) and computes the exact global weighted average across all GPUs via `dist_utils.weighted_average`.
