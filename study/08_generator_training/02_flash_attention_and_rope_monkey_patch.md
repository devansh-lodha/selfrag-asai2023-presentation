# FlashAttention and the RoPE Monkey Patch

## 1. Plain-English Intuition: Why Standard Attention Runs Out of Memory on 2048-Token Sequences

In Self-RAG Generator training, each training example interleaves the user instruction $x$, retrieved Wikipedia passages (`<paragraph>...</paragraph>`), reflection tokens, and multi-sentence answers. To fit these augmented trajectories, the maximum sequence length is set to $L = 2048$ tokens for Llama-2 7B (and $L = 1536$ for 13B in the training script; the paper reports $1,524$).

What happens inside standard PyTorch self-attention when the sequence length reaches $L = 2048$?
1. Quadratic $O(L^2)$ attention matrix memory:
   In standard attention (Module 01 Part 2), every token $t \in \{1, \dots, L\}$ computes a similarity score $\langle q_t, k_i \rangle / \sqrt{d_h}$ with every position $i \in \{1, \dots, L\}$, forming an $L \times L$ matrix ($2048 \times 2048 = 4,194,304$ numbers) for each of the $H = 32$ attention heads in each of the $32$ transformer layers.
   During training, PyTorch must save all of those $L \times L$ softmax matrices in GPU High Bandwidth Memory (HBM) during the forward pass so it can compute gradients during the backward pass. For a single sequence ($B = 1$) in 16-bit precision (`2 bytes` per number), storing $32\text{ layers} \times 32\text{ heads} \times 2048 \times 2048 \times 2\text{ bytes}$ consumes $8.59\text{ GB}$ of GPU memory just for the attention scores!
2. Memory bandwidth bottleneck:
   Modern GPUs (like the A100) can multiply numbers in their tiny on-chip memory (SRAM, about $192\text{ KB}$ per streaming multiprocessor, roughly $20\text{ MB}$ across the whole chip, with about $19\text{ TB/s}$ of bandwidth) much faster than they can read and write tensors to main GPU memory (HBM, $40\text{ GB}$ or $80\text{ GB}$, running at $1.5 - 2.0\text{ TB/s}$). Standard attention repeatedly writes the huge $[B, H, L, L]$ matrix to slow HBM and reads it back for softmax, masking, and value multiplication.

Both `train_special_tokens.py` and `finetune.py` solve this by calling `replace_llama_attn_with_flash_attn()` from `reference/code/retrieval_lm/llama_flash_attn_monkey_patch.py`, which speeds up training and slashes memory usage through two mechanisms:
1. FlashAttention: Computes exact causal self-attention inside fast on-chip SRAM tiles without ever materializing the $[B, H, L, L]$ attention matrix in HBM, reducing attention memory from $O(L^2)$ to $O(L)$.
2. Sequence Unpadding (`unpad_input`): Strips out all `<pad>` tokens across the batch before running FlashAttention so the GPU never wastes compute attending to or from padding positions.

---

## 2. Step-by-Step Build-Up: Rotary Position Embeddings (RoPE, Lines 48-52)

Before calling FlashAttention, lines 48-52 of `llama_flash_attn_monkey_patch.py` apply Rotary Position Embeddings (`apply_rotary_pos_emb`) to the Query and Key tensors:

```python
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
```

### 2.1 Intuition: Why Dot Products Need Relative Distance $m - n$

Look back at the attention dot product $\langle q_m, k_n \rangle$ between a query at token position $m$ and a key at token position $n$. If $q_m = W_Q h_m$ and $k_n = W_K h_n$ only depended on the word embeddings, then the dot product $\langle q_m, k_n \rangle$ would be completely blind to where the words sit in the sentence! Whether `"not"` appeared $1$ word before `"good"` ($m - n = 1$) or $500$ words away ($m - n = 500$), $\langle q_m, k_n \rangle$ would be identical.

How can we modify $q_m$ at position $m$ and $k_n$ at position $n$ so that their dot product $\langle q_m^{\text{rot}}, k_n^{\text{rot}} \rangle$ automatically depends on their relative distance $m - n$?

### 2.2 The 2D Rotation Trick in RoPE

Rotary Position Embeddings (RoPE) group the $d_h = 128$ coordinates of a query or key vector into $d_h / 2 = 64$ pairs of 2D coordinates. Consider a single 2D pair $(x_1, x_2)$ at sequence position $m \in \{0, 1, \dots, L-1\}$.

RoPE rotates the 2D vector $(x_1, x_2)$ counter-clockwise in the 2D plane by an angle $m \theta$, which is proportional to its token position $m$:

$$\begin{bmatrix} q_{m, 1}^{\text{rot}} \\ q_{m, 2}^{\text{rot}} \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_{m, 1} \\ q_{m, 2} \end{bmatrix} = \begin{bmatrix} q_{m, 1}\cos(m\theta) - q_{m, 2}\sin(m\theta) \\ q_{m, 1}\sin(m\theta) + q_{m, 2}\cos(m\theta) \end{bmatrix}$$

And at position $n$, the key pair $(k_{n, 1}, k_{n, 2})$ is rotated by angle $n \theta$:

$$\begin{bmatrix} k_{n, 1}^{\text{rot}} \\ k_{n, 2}^{\text{rot}} \end{bmatrix} = \begin{bmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{bmatrix} \begin{bmatrix} k_{n, 1} \\ k_{n, 2} \end{bmatrix}$$

where:
- $m \in \{0, \dots, L-1\}$ is the query token position index (`position_ids`).
- $n \in \{0, \dots, L-1\}$ is the key token position index.
- $\theta_r = 10000^{-2(r-1)/d_h}$ is the fixed rotation frequency for the $r$-th 2D coordinate pair ($r \in \{1, \dots, 64\}$).

Why does rotating $q_m$ by $m\theta$ and $k_n$ by $n\theta$ encode relative distance $m - n$?
Because rotating two vectors in a plane and taking their dot product only depends on the difference between their rotation angles, $(m - n)\theta$! Using the trigonometric identities $\cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) = \cos((m - n)\theta)$ and $\sin(m\theta)\cos(n\theta) - \cos(m\theta)\sin(n\theta) = \sin((m - n)\theta)$, the dot product of the rotated 2D pairs simplifies to:

$$\left\langle q_m^{\text{rot}}, k_n^{\text{rot}} \right\rangle = (q_{m,1}k_{n,1} + q_{m,2}k_{n,2})\cos\left((m - n)\theta\right) + (q_{m,1}k_{n,2} - q_{m,2}k_{n,1})\sin\left((m - n)\theta\right)$$

Notice that the absolute positions $m$ and $n$ have vanished, leaving only the relative token distance $m - n$! Notice also that in lines 48-52, RoPE is applied only to `query_states` and `key_states` (which determine the attention weights), and not to `value_states`.

---

## 3. Step-by-Step Build-Up: Unpadding Variable-Length Batches (`unpad_input` and `cu_q_lens`)

### 3.1 Why We Pack Non-Pad Tokens into a 1D Stream

In a training batch of $B$ examples, some examples might be $300$ tokens long while the longest example in the batch is $L_{\max} = 2000$ tokens long. If we ran attention on the padded $[B, L_{\max}]$ grid, the shorter examples would waste $1700$ positions of `<pad>` tokens.

To avoid computing attention on `<pad>` tokens, lines 72-89 of `llama_flash_attn_monkey_patch.py` use `unpad_input` and `pad_input` from `flash_attn.bert_padding`.

Let us trace a concrete toy batch of $B = 2$ sequences padded to $L_{\max} = 4$ tokens to see what every variable returned by `unpad_input(x, key_padding_mask)` contains:
- Sequence $b = 0$ has $3$ real tokens (`A0, A1, A2`) and $1$ pad token (`PAD`): `key_padding_mask[0] = [1, 1, 1, 0]`
- Sequence $b = 1$ has $2$ real tokens (`B0, B1`) and $2$ pad tokens (`PAD, PAD`): `key_padding_mask[1] = [1, 1, 0, 0]`

When the $2 \times 4$ grid is flattened into $8$ consecutive slots ($0, 1, 2, 3, 4, 5, 6, 7$):

| Flattened Slot Index | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ |
|---|---|---|---|---|---|---|---|---|
| Batch & Position $(b, k)$ | $(0, 0)$ | $(0, 1)$ | $(0, 2)$ | $(0, 3)$ | $(1, 0)$ | $(1, 1)$ | $(1, 2)$ | $(1, 3)$ |
| Token Content | `A0` | `A1` | `A2` | `PAD` | `B0` | `B1` | `PAD` | `PAD` |
| `key_padding_mask` | $1$ | $1$ | $1$ | $0$ | $1$ | $1$ | $0$ | $0$ |
| Kept by `unpad_input`? | Keep ($0$) | Keep ($1$) | Keep ($2$) | Drop | Keep ($3$) | Keep ($4$) | Drop | Drop |

Calling `x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)` returns four objects:
1. `x_unpad` (shape $[\text{nnz}, \dots]$, where $\text{nnz} = 3 + 2 = 5$ is the total number of non-zero mask positions):
   Contains only the $5$ real token vectors `[A0, A1, A2, B0, B1]` packed end-to-end with zero padding tokens.
2. `indices` (shape $[\text{nnz}] = [5]$):
   `tensor([0, 1, 2, 4, 5])`, which records the original flattened slot indices in $0 \dots 7$ where those $5$ real tokens came from. After FlashAttention finishes, `pad_input(output_unpad, indices, bsz=2, q_len=4)` uses `indices` to scatter the $5$ output vectors back into slots `[0, 1, 2, 4, 5]` and fill slots `[3, 6, 7]` with zeros!
3. `cu_q_lens` (cumulative query lengths, shape $[B + 1] = [3]$, `dtype=torch.int32`):
   `tensor([0, 3, 5])`.
   How does FlashAttention know where Sequence $0$ ends and Sequence $1$ begins inside the 1D packed array `x_unpad = [A0, A1, A2, B0, B1]`?
   From `cu_q_lens`: Sequence $b$ occupies the slice `x_unpad[cu_q_lens[b] : cu_q_lens[b + 1]]`!
   - For $b = 0$: `x_unpad[0 : 3]` $\to$ `[A0, A1, A2]` (length $3 - 0 = 3$).
   - For $b = 1$: `x_unpad[3 : 5]` $\to$ `[B0, B1]` (length $5 - 3 = 2$).
   Inside the CUDA kernel, tokens in slice `[3 : 5]` only attend causally within `[3 : 5]` and never cross the boundary at index $3$ into Sequence $0$.
4. `max_s` (integer):
   `3` (the maximum unpadded sequence length $\max(3, 2) = 3$ in the batch, used by FlashAttention to size its CUDA thread blocks).

---

## 4. Line-by-Line Walkthrough of `llama_flash_attn_monkey_patch.py`

Let us trace the complete `forward` function and monkey-patching hooks in `reference/code/retrieval_lm/llama_flash_attn_monkey_patch.py` (lines 55-107):

```python
    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad,
                          "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None
```

### 4.1 Complete Tensor Shape Trace Through `forward`

The table below tracks the exact tensor shape at every line of `forward` (using Llama-2 7B dimensions: batch size $B = \text{bsz}$, sequence length $L = \text{q\_len}$, $H = \text{num\_heads} = 32$, $d_h = \text{head\_dim} = 128$, and $d = 4096$):

| Code Line / Operation | Tensor Variable | Exact Tensor Shape | What This Operation Accomplishes |
|---|---|---|---|
| Input to `forward` | `hidden_states` | $[B, L, 4096]$ | Input embeddings / previous layer outputs |
| `self.q_proj(...).view(...).transpose(1, 2)` | `query_states` | $[B, 32, L, 128]$ | Projects and splits into $32$ heads of size $128$ |
| `torch.stack([q, k, v], dim=2)` | `qkv` | $[B, 32, 3, L, 128]$ | Packs Q, K, V along a new dimension of size $3$ |
| `qkv.transpose(1, 3)` | `qkv` | $[B, L, 3, 32, 128]$ | Moves sequence length $L$ to dim $1$ for FlashAttention |
| `rearrange(qkv, "b s three h d -> b s (three h d)")` | `x` | $[B, L, 12288]$ | Flattens $3 \times 32 \times 128 = 12288$ so `unpad_input` sees a 3D tensor |
| `unpad_input(x, key_padding_mask)` | `x_unpad` | $[\text{nnz}, 12288]$ | Removes all `<pad>` tokens across the batch |
| `rearrange(x_unpad, "nnz (three h d) -> nnz three h d")` | `x_unpad` | $[\text{nnz}, 3, 32, 128]$ | Restores the packed QKV shape required by FlashAttention |
| `flash_attn_unpadded_qkvpacked_func(..., causal=True)` | `output_unpad` | $[\text{nnz}, 32, 128]$ | Runs exact tiled causal attention in SRAM (with `dropout=0.0`, `softmax_scale=1/\sqrt{128}`) |
| `pad_input(rearrange(...), indices, bsz, q_len)` | `output` | $[B, L, 32, 128]$ | Scatters the $\text{nnz}$ outputs back to $[B, L]$ and fills `<pad>` slots with $0.0$ |
| `self.o_proj(rearrange(output, "b s h d -> b s (h d)"))` | Return value `[0]` | $[B, L, 4096]$ | Concatenates all $32$ heads ($32 \times 128 = 4096$) and applies output projection $W_O$ |

### 4.2 Why `_prepare_decoder_attention_mask` Is Replaced (`replace_llama_attn_with_flash_attn`, Lines 95-107)

Look at the final lines (95-107) of `llama_flash_attn_monkey_patch.py`:

```python
# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
```

Why must `_prepare_decoder_attention_mask` be replaced alongside `LlamaAttention.forward`?
- In standard Hugging Face `LlamaModel`, before calling the attention layers, `_prepare_decoder_attention_mask` takes the 2D padding mask `attention_mask` of shape $[B, L]$ (containing $1$s and $0$s) and expands it into a 4D causal attention mask tensor of shape $[B, 1, L, L]$ (containing $0.0$ and $-\infty$).
- However, FlashAttention already enforces the causal upper-triangular mask internally via the flag `causal=True` (line 80), and `unpad_input(x, key_padding_mask)` requires the 2D `[bsz, seq_len]` padding mask!
- Therefore, `replace_llama_attn_with_flash_attn()` overwrites `LlamaModel._prepare_decoder_attention_mask` on the Python class so that `LlamaModel` passes the raw 2D `attention_mask` $[B, L]$ straight through to `LlamaAttention.forward` without allocating a 4D $[B, 1, L, L]$ tensor.
