# Contrastive Learning: In-Batch Negatives and MoCo

## 1. Plain-English Intuition: How Do We Train a Bi-Encoder Without Representation Collapse?

In Part 1, we saw how the `Contriever` encoder $E_\theta(\cdot)$ turns any query $q$ into a $768$-dimensional vector $\mathbf{q} \in \mathbb{R}^{768}$ and any passage $d$ into a $768$-dimensional vector $\mathbf{k} \in \mathbb{R}^{768}$ (called a key vector), scoring their match by their dot product $\langle \mathbf{q}, \mathbf{k} \rangle$.

How do we train the encoder weights $\theta$ so that matching query-passage pairs $(q_i, k_i^+)$ have a large positive dot product, while non-matching pairs $(q_i, k_j^-)$ have a small or negative dot product?

Suppose we tried a naive training loss that only looked at matching pairs $(q_i, k_i^+)$ and tried to maximize their similarity $\langle \mathbf{q}_i, \mathbf{k}_i^+ \rangle$. What would go wrong?
- The neural network would immediately find a trivial shortcut: it would map every single text string in the world to the exact same constant vector, such as $[10, 10, \dots, 10]$!
- If every query and every passage mapped to the exact same vector, every positive pair would have a huge dot product, the training loss would drop to zero, and the search engine would be completely useless because unrelated passages would score just as high as relevant ones. This failure mode is called representation collapse.

To prevent collapse, training must simultaneously:
1. Pull each query $\mathbf{q}_i$ closer to its true matching passage $\mathbf{k}_i^+$ (the positive key).
2. Push $\mathbf{q}_i$ away from $M$ non-matching distractor passages $\{\mathbf{k}_1^-, \mathbf{k}_2^-, \dots, \mathbf{k}_M^-\}$ (the negative keys).

Contrastive learning accomplishes both goals at once by framing retrieval as a $(1 + M)$-way multiple-choice classification problem: given query $\mathbf{q}_i$, the model computes its dot product with the $1$ positive passage $\mathbf{k}_i^+$ and $M$ negative passages $\mathbf{k}_1^-, \dots, \mathbf{k}_M^-$, and uses the exact same softmax cross-entropy loss from Module 01 to maximize the probability of picking $\mathbf{k}_i^+$ out of all $1 + M$ candidates.

The `retrieval_lm/src/` codebase implements two ways of supplying those $M$ negative passages:
1. In-Batch Negatives (`src/inbatch.py` and `src/dist_utils.py`), where the passages belonging to the other queries in the current mini-batch across all GPUs serve as the negatives.
2. Momentum Contrast (`src/moco.py`), which maintains a rolling First-In-First-Out queue of $M = 65536$ passage vectors from past batches so the model faces $65536$ negatives on every step even when the GPU batch size is small.

---

## 2. Step-by-Step Mathematical Build-Up: The InfoNCE Contrastive Loss

### 2.1 From Dot Products to Multi-Class Cross-Entropy (InfoNCE)

For a single query $q_i$ with encoded vector $\mathbf{q}_i \in \mathbb{R}^D$, let $\mathbf{k}_i^+ \in \mathbb{R}^D$ denote its matching positive passage vector, and let $\{\mathbf{k}_1^-, \dots, \mathbf{k}_M^-\}$ denote $M$ negative passage vectors.

First, we compute the dot product between $\mathbf{q}_i$ and each candidate passage, and divide by a positive temperature hyperparameter $\tau > 0$ (`opt.temperature`) to form $(1 + M)$ logits:

$$z_{i, \text{pos}} = \frac{\langle \mathbf{q}_i, \mathbf{k}_i^+ \rangle}{\tau}, \quad z_{i, j} = \frac{\langle \mathbf{q}_i, \mathbf{k}_j^- \rangle}{\tau} \quad \text{for } j \in \{1, \dots, M\}$$

Second, applying the softmax function over all $1 + M$ candidates gives the probability assigned to the true positive passage $\mathbf{k}_i^+$:

$$p_\theta\left(\mathbf{k}_i^+ \;\middle|\; \mathbf{q}_i\right) = \frac{\exp\left(\frac{\langle \mathbf{q}_i, \mathbf{k}_i^+ \rangle}{\tau}\right)}{\exp\left(\frac{\langle \mathbf{q}_i, \mathbf{k}_i^+ \rangle}{\tau}\right) + \sum_{j=1}^{M} \exp\left(\frac{\langle \mathbf{q}_i, \mathbf{k}_j^- \rangle}{\tau}\right)}$$

Third, taking the negative logarithm of this probability yields the InfoNCE contrastive loss for query $i$:

$$\mathcal{L}_{\text{InfoNCE}}(i) = -\log \frac{\exp\left(\frac{\langle \mathbf{q}_i, \mathbf{k}_i^+ \rangle}{\tau}\right)}{\exp\left(\frac{\langle \mathbf{q}_i, \mathbf{k}_i^+ \rangle}{\tau}\right) + \sum_{j=1}^{M} \exp\left(\frac{\langle \mathbf{q}_i, \mathbf{k}_j^- \rangle}{\tau}\right)}$$

where:
- $\mathbf{q}_i \in \mathbb{R}^D$ is the query embedding vector ($D = 768$).
- $\mathbf{k}_i^+ \in \mathbb{R}^D$ is the positive key embedding vector matching query $i$.
- $\mathbf{k}_j^- \in \mathbb{R}^D$ is the $j$-th negative key embedding vector, for $j \in \{1, \dots, M\}$.
- $M \in \mathbb{Z}^+$ is the number of negative passages per query.
- $\tau > 0$ is the temperature parameter (`opt.temperature`). When vectors are $\ell_2$-normalized ($\Vert \mathbf{q}_i \Vert_2 = \Vert \mathbf{k} \Vert_2 = 1$), their dot product lies in $[-1, 1]$; setting a small temperature such as $\tau = 0.05$ scales the logits into $[-20, 20]$ so the softmax can reach probabilities close to $1.0$.

Notice how minimizing $\mathcal{L}_{\text{InfoNCE}}(i)$ forces the numerator $\exp(\langle \mathbf{q}_i, \mathbf{k}_i^+ \rangle / \tau)$ to grow large (pulling $\mathbf{q}_i$ and $\mathbf{k}_i^+$ together) while forcing every term $\exp(\langle \mathbf{q}_i, \mathbf{k}_j^- \rangle / \tau)$ in the denominator to shrink (pushing $\mathbf{q}_i$ away from all $M$ negatives $\mathbf{k}_j^-$).

---

## 3. Method 1: In-Batch Negatives (`src/inbatch.py`) and Distributed Gathering (`src/dist_utils.py`)

### 3.1 How In-Batch Negatives Work on a Single GPU

Suppose a single GPU loads a mini-batch of $B = 3$ positive query-passage pairs:
- Pair $0$: $(\mathbf{q}_0, \mathbf{k}_0)$
- Pair $1$: $(\mathbf{q}_1, \mathbf{k}_1)$
- Pair $2$: $(\mathbf{q}_2, \mathbf{k}_2)$

Instead of encoding extra negative passages from scratch, we reuse the other passages already in the batch as negatives:
- For query $\mathbf{q}_0$: $\mathbf{k}_0$ is the positive key, while $\mathbf{k}_1$ and $\mathbf{k}_2$ are the negatives (target column index $= 0$).
- For query $\mathbf{q}_1$: $\mathbf{k}_1$ is the positive key, while $\mathbf{k}_0$ and $\mathbf{k}_2$ are the negatives (target column index $= 1$).
- For query $\mathbf{q}_2$: $\mathbf{k}_2$ is the positive key, while $\mathbf{k}_0$ and $\mathbf{k}_1$ are the negatives (target column index $= 2$).

Computing all pairwise dot products between the query matrix $Q \in \mathbb{R}^{B \times D}$ and key matrix $K \in \mathbb{R}^{B \times D}$ produces a $B \times B$ score matrix where the positive pairs sit along the main diagonal $(0, 0), (1, 1), (2, 2)$, so the ground-truth target class labels for rows $0, 1, 2$ are simply `labels = [0, 1, 2]`.

### 3.2 Scaling Across $W$ GPUs and Why Target Labels Shift by `rank * B`

Now suppose we train across $W = 2$ GPUs in parallel, where each GPU holds $B = 3$ pairs:
- GPU $0$ (`rank = 0`) encodes queries $[\mathbf{q}_0, \mathbf{q}_1, \mathbf{q}_2]$ and keys $[\mathbf{k}_0, \mathbf{k}_1, \mathbf{k}_2]$.
- GPU $1$ (`rank = 1`) encodes queries $[\mathbf{q}_3, \mathbf{q}_4, \mathbf{q}_5]$ and keys $[\mathbf{k}_3, \mathbf{k}_4, \mathbf{k}_5]$.

If each GPU only compared its queries against its own $3$ local keys, it would only get $M = 2$ negatives per query. Instead, both GPUs broadcast their key matrices to each other and stack them into a single combined key matrix `gather_kemb` of shape $[W \cdot B, D] = [6, D]$ containing all $6$ keys $[\mathbf{k}_0, \mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4, \mathbf{k}_5]$.

Look at the $[3, 6]$ similarity matrix `scores` computed on GPU $0$ versus GPU $1$ against `gather_kemb`:

| GPU Rank $r$ | Local Query Row $i$ | Col $0$ ($\mathbf{k}_0$) | Col $1$ ($\mathbf{k}_1$) | Col $2$ ($\mathbf{k}_2$) | Col $3$ ($\mathbf{k}_3$) | Col $4$ ($\mathbf{k}_4$) | Col $5$ ($\mathbf{k}_5$) | Target Column Index `labels[i]` |
|---|---|---|---|---|---|---|---|---|
| GPU $0$ ($r = 0$) | Row $0$ ($\mathbf{q}_0$) | Positive | Negative | Negative | Negative | Negative | Negative | $0 + 0 \times 3 = 0$ |
| GPU $0$ ($r = 0$) | Row $1$ ($\mathbf{q}_1$) | Negative | Positive | Negative | Negative | Negative | Negative | $1 + 0 \times 3 = 1$ |
| GPU $0$ ($r = 0$) | Row $2$ ($\mathbf{q}_2$) | Negative | Negative | Positive | Negative | Negative | Negative | $2 + 0 \times 3 = 2$ |
| GPU $1$ ($r = 1$) | Row $0$ ($\mathbf{q}_3$) | Negative | Negative | Negative | Positive | Negative | Negative | $0 + 1 \times 3 = 3$ |
| GPU $1$ ($r = 1$) | Row $1$ ($\mathbf{q}_4$) | Negative | Negative | Negative | Negative | Positive | Negative | $1 + 1 \times 3 = 4$ |
| GPU $1$ ($r = 1$) | Row $2$ ($\mathbf{q}_5$) | Negative | Negative | Negative | Negative | Negative | Positive | $2 + 1 \times 3 = 5$ |

Look at the rightmost column for GPU $1$ (`rank = 1`): because GPU $0$'s $3$ keys occupy columns $0, 1, 2$ of `gather_kemb`, GPU $1$'s keys $\mathbf{k}_3, \mathbf{k}_4, \mathbf{k}_5$ sit at columns $3, 4, 5$! Therefore, on GPU $r$, the target column index for local query $i \in \{0, \dots, B-1\}$ must be shifted by $r \times B$:

$$\text{label}_i^{(r)} = i + r \cdot B$$

### 3.3 Line-by-Line Walkthrough of `InBatch.forward` (`src/inbatch.py`, Lines 58-89)

Let us inspect `InBatch.forward` in `reference/code/retrieval_lm/src/inbatch.py` (lines 58-89):

```python
    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
```

Tracing each line:
1. `labels = torch.arange(0, bsz, ...)` creates `[0, 1, ..., B - 1]`.
2. `qemb` and `kemb` (both of shape $[B, 768]$) are computed by passing the queries and keys through `self.encoder` (`Contriever`).
3. `gather_kemb = gather_fn(kemb)` gathers `kemb` across all $W$ GPUs into a tensor of shape $[W \cdot B, 768]$.
4. `labels = labels + dist_utils.get_rank() * len(kemb)` applies the exact rank offset $i + r \cdot B$ from our table above.
5. `scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)` multiplies `qemb / tau` (shape $[B, 768]$) by the transpose of `gather_kemb` (shape $[W \cdot B, 768]$), summing over the feature dimension `d` to produce the $[B, W \cdot B]$ logit matrix `scores`.
6. `torch.nn.functional.cross_entropy(scores, labels, ...)` computes the InfoNCE loss over all $B$ queries.

### 3.4 Why `dist_utils.Gather` Needs a Custom Backward Pass (`src/dist_utils.py`, Lines 7-26)

Look at how `dist_utils.gather` is implemented in `reference/code/retrieval_lm/src/dist_utils.py` (lines 7-26):

```python
class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.tensor):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather(x: torch.tensor):
    if not dist.is_initialized():
        return x
    x_gather = Gather.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather
```

Why did the authors write a custom `torch.autograd.Function` instead of just calling `dist.all_gather` directly?
1. PyTorch's built-in `dist.all_gather(output, x)` copies tensor bytes across GPUs outside the autograd graph, so `output` has no gradient connection back to `x`.
2. In the forward pass, GPU $r$'s local key tensor `x` ($\mathbf{k}^{(r)}$) is sent to all $W$ GPUs: GPU $0$ uses $\mathbf{k}^{(r)}$ in its loss $\mathcal{L}_0$, GPU $1$ uses $\mathbf{k}^{(r)}$ in its loss $\mathcal{L}_1$, and so on.
3. During backpropagation, each GPU $w$ computes a gradient $\frac{\partial \mathcal{L}_w}{\partial \mathbf{k}^{(r)}}$ (`grads[r]` on GPU $w$). By the chain rule, the total gradient with respect to GPU $r$'s keys $\mathbf{k}^{(r)}$ across all GPUs is the sum of those gradients from every GPU:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathbf{k}^{(r)}} = \sum_{w=0}^{W-1} \frac{\partial \mathcal{L}_w}{\partial \mathbf{k}^{(r)}}$$

In `backward(ctx, *grads)`, `dist.all_reduce(all_gradients)` sums `all_gradients` across all $W$ GPUs, and `return all_gradients[dist.get_rank()]` returns the summed gradient for slice `dist.get_rank()` back to `x = kemb` on GPU $r$ so `self.encoder` receives the exact full-batch gradient.

---

## 4. Method 2: Momentum Contrast (`src/moco.py`)

### 4.1 Intuition: Decoupling Negative Count from Batch Size with a Queue

In `InBatch`, the number of negatives $M = W \cdot B - 1$ is tied to how many sequences can fit in GPU memory in a single step. Because each BERT forward pass stores activations across $12$ layers, $B$ per GPU is typically around $64$.

What if we want $M = 65536$ negative passages on every training step without needing $1024$ GPUs?

Momentum Contrast (`MoCo`, implemented in `reference/code/retrieval_lm/src/moco.py`) achieves this by maintaining a rolling First-In-First-Out (FIFO) queue of $K_{\text{queue}} = 65536$ key vectors (`self.queue` of shape $[768, 65536]$) encoded during previous training steps:
- On the current step, the model encodes $B$ new queries $\mathbf{q}$ and $B$ new positive keys $\mathbf{k}^+$.
- Every query $\mathbf{q}_i$ is scored against its $1$ positive key $\mathbf{k}_i^+$ and all $65536$ negative keys currently sitting in `self.queue`.
- After computing the loss, the $B$ new keys $\mathbf{k}^+$ are written into `self.queue` at pointer `queue_ptr`, overwriting the $B$ oldest keys in the queue, and `queue_ptr` advances by $B$.

### 4.2 Why the Key Encoder $\theta_k$ Uses Momentum Updates Instead of Backpropagation

There is one catch with reusing $65536$ key vectors from hundreds of previous steps:
1. We cannot backpropagate gradients into the $65536$ old queue vectors because their intermediate transformer activations were freed hundreds of steps ago.
2. If we updated a single encoder rapidly by gradient descent on every step, the newest key vectors entering the queue at step $1000$ would come from very different weights than the oldest key vectors still sitting in the queue from step $700$. The query encoder would cheat by simply distinguishing "new weights" from "old weights" rather than learning semantic similarity!

MoCo solves this by keeping two copies of the Contriever encoder:
- Query Encoder $\theta_q$ (`self.encoder_q`): receives gradients and is updated by the AdamW optimizer on every step.
- Key Encoder $\theta_k$ (`self.encoder_k`): has gradients disabled (`param_k.requires_grad = False`). Instead of backpropagation, on every step $\theta_k$ is updated as a slow exponential moving average (momentum update) of $\theta_q$:

$$\theta_k \leftarrow m \, \theta_k + (1 - m) \, \theta_q$$

where $m \in [0, 1)$ is the momentum coefficient (`opt.momentum`, default $m = 0.999$).

Because $m = 0.999$, on each step $\theta_k$ retains $99.9\%$ of its previous weights and adds only $0.1\%$ ($1 - m = 0.001$) of the new query encoder weights $\theta_q$. This makes $\theta_k$ evolve very slowly and smoothly across hundreds of steps, ensuring that all $65536$ vectors in `self.queue` remain compatible with one another.

### 4.3 Line-by-Line Walkthrough of `MoCo` (`src/moco.py`)

Let us trace the three core methods in `reference/code/retrieval_lm/src/moco.py`:

1. Momentum update (`_momentum_update_key_encoder`, lines 73-78):

```python
    def _momentum_update_key_encoder(self):
        """
        Update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
```

This loop directly executes $\theta_k \leftarrow m \theta_k + (1 - m) \theta_q$ parameter tensor by parameter tensor.

2. Computing positive and queue negative logits (`_compute_logits` and `forward`, lines 95-123):

```python
    def _compute_logits(self, q, k):
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        return logits
```

Tracing the tensor shapes in `_compute_logits`:
- `q` has shape $[B, 768]$ (`nc`, where `n` $= B$ and `c` $= 768$), and `k` has shape $[B, 768]$ (`nc`).
- `torch.einsum("nc,nc->n", [q, k])` computes the row-wise dot product $\langle \mathbf{q}_n, \mathbf{k}_n^+ \rangle$ for each pair $n \in \{0, \dots, B-1\}$, producing a 1D tensor of shape $[B]$, and `.unsqueeze(-1)` reshapes it to $[B, 1]$ (`l_pos`).
- `self.queue` has shape $[768, 65536]$ (`ck`, where `c` $= 768$ and `k` $= 65536$). `torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])` multiplies $[B, 768]$ by $[768, 65536]$ to produce `l_neg` of shape $[B, 65536]$.
- `logits = torch.cat([l_pos, l_neg], dim=1)` concatenates the $1$ positive column and $65536$ negative columns into a matrix of shape $[B, 65537]$.
- `_compute_logits` itself does not apply the temperature: `forward` divides the concatenated logits by it on line 118 (`logits = self._compute_logits(q, k) / self.temperature`), giving the $\tau$-scaled InfoNCE logits from Section 2.
- Because the positive logit `l_pos` is placed in column $0$ for every row $n \in \{0, \dots, B-1\}$, line 120 sets the target class index for every row to $0$:

```python
        labels = torch.zeros(bsz, dtype=torch.long).cuda()
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
```

3. Circular queue replacement (`_dequeue_and_enqueue`, lines 80-93):

```python
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f"{batch_size}, {self.queue_size}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
```

Here, `dist_utils.gather_nograd` gathers the newly encoded keys from all GPUs into `keys` of shape $[B_{\text{total}}, 768]$. Transposing `keys.T` gives shape $[768, B_{\text{total}}]$, which overwrites columns `ptr` through `ptr + batch_size - 1` of `self.queue`. Finally, the modulo operation `(ptr + batch_size) % self.queue_size` wraps `ptr` back to $0$ whenever it reaches the end of the $65536$-column buffer.
