# Retriever Data Augmentation and Fine-Tuning Data

## 1. Plain-English Intuition: How Do You Train a Search Engine Before and After Having Labeled Questions?

In Part 2, we saw how `InBatch` and `MoCo` train the Contriever bi-encoder using positive pairs $(q_i, k_i^+)$ and negative distractors $\{k_j^-\}$.

Where do the positive pairs $(q_i, k_i^+)$ come from? Contriever (`facebook/contriever-msmarco`) is trained in two stages, supported by two data modules in `reference/code/retrieval_lm/src/`:

1. Stage 1: Unsupervised Contrastive Pre-Training (`src/data.py`):
   Before showing the model any human-written search queries, we want it to learn general semantic similarity from billions of words of plain, unlabeled text (such as Wikipedia articles). How can we create a "query" $q_i$ and a "matching passage" $k_i^+$ out of a single unlabeled paragraph?
   - We use the Independent Cropping trick: take a $256$-token chunk of text from a document, and cut out two random sub-spans of tokens from that same chunk (for example, a $45$-token span as the query $q_i$ and an $80$-token span as the positive key $k_i^+$).
   - Because both spans come from the same paragraph, they naturally share the same topic and context!
   - To prevent the encoder from cheating when the two random spans happen to overlap (by just matching identical words), we randomly corrupt $10\%$ of the tokens in each span (by deleting, replacing, masking, or shuffling random words).

2. Stage 2: Supervised Fine-Tuning with Hard Negatives (`src/finetuning_data.py`):
   After unsupervised pre-training, the retriever is fine-tuned on real human question-passage datasets (such as MS MARCO). Here, each question $q_i$ comes with a verified relevant passage (`positive_ctxs`) and tricky "hard negative" passages (`hard_negative_ctxs`), which are passages that contain the exact same keywords as $q_i$ without actually answering the question. Training against hard negatives teaches the retriever to look past superficial keyword overlap.

---

## 2. Unsupervised Contrastive Data Pipeline (`src/data.py`)

### 2.1 Independent Random Span Cropping (`Dataset` and `randomcrop`, Lines 84-146)

Let us inspect `Dataset` and `randomcrop` in `reference/code/retrieval_lm/src/data.py` (lines 84-146):

```python
class Dataset(torch.utils.data.Dataset):
    """Monolingual dataset based on a list of paths"""

    def __init__(self, data, chunk_length, tokenizer, opt):

        self.data = data
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer
        self.opt = opt
        self.generate_offset()

    def __len__(self):
        return (self.data.size(0) - self.offset) // self.chunk_length

    def __getitem__(self, index):
        start_idx = self.offset + index * self.chunk_length
        end_idx = start_idx + self.chunk_length
        tokens = self.data[start_idx:end_idx]
        q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        k_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        q_tokens = apply_augmentation(q_tokens, self.opt)
        q_tokens = add_bos_eos(q_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
        k_tokens = apply_augmentation(k_tokens, self.opt)
        k_tokens = add_bos_eos(k_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

        return {"q_tokens": q_tokens, "k_tokens": k_tokens}

    def generate_offset(self):
        self.offset = random.randint(0, self.chunk_length - 1)


def randomcrop(x, ratio_min, ratio_max):

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end].clone()
    return crop
```

Tracing how `Dataset.__getitem__` creates a positive pair `(q_tokens, k_tokens)` from a 1D tensor of pre-tokenized corpus IDs `self.data`:
1. Shifting chunk boundaries across epochs (`generate_offset`):
   Instead of always slicing `self.data` at fixed multiples of `chunk_length` ($0, 256, 512, \dots$), `generate_offset()` picks a random integer `self.offset` in $\{0, 1, \dots, L_{\text{chunk}} - 1\}$ at the start of each epoch so chunk boundaries fall at different places every epoch.
2. Extracting two independent random crops (`randomcrop`):
   Given the $L_{\text{chunk}} = 256$ token slice `tokens`, `randomcrop` draws a random fraction $r \sim \text{Uniform}(r_{\min}, r_{\max})$ (where default `ratio_min = 0.1` and `ratio_max = 0.5`), computes the crop length $L_{\text{crop}} = \lfloor r \cdot L_{\text{chunk}} \rfloor \in [25, 128]$, picks a random starting position `start` in $\{0, \dots, L_{\text{chunk}} - L_{\text{crop}}\}$, and slices `tokens[start : start + length]`. Calling `randomcrop` twice independently produces `q_tokens` and `k_tokens`.

### 2.2 Token-Level Corruptions (`deleteword`, `replaceword`, `maskword`, `shuffleword`, Lines 161-214)

After cropping `q_tokens` and `k_tokens`, `apply_augmentation` applies one of four random token corruptions with probability $p$ (`opt.prob_augmentation`, typically $p = 0.10$):

```python
def deleteword(x, p=0.1):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x


def replaceword(x, min_random, max_random, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else random.randint(min_random, max_random) for e, m in zip(x, mask)]
    return x


def maskword(x, mask_id, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return x


def shuffleword(x, p=0.1):
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x
```

To see how each augmentation modifies a token list `x = [10, 20, 30, 40, 50]` when the random draws `mask = np.random.rand(5)` happen to fall below $p = 0.10$ at indices $1$ (`20`) and $3$ (`40`):
- `deleteword`: drops indices $1$ and $3$, shortening the sequence to `[10, 30, 50]`.
- `replaceword`: replaces indices $1$ and $3$ with random vocabulary IDs (say `999` and `42`), giving `[10, 999, 30, 42, 50]`.
- `maskword`: replaces indices $1$ and $3$ with the `[MASK]` token ID (`103` in BERT), giving `[10, 103, 30, 103, 50]`.
- `shuffleword`: picks $2$ random positions (say indices $1$ and $3$) and permutes their values, giving `[10, 40, 30, 20, 50]`.

Finally, `add_bos_eos` (lines 204-214) prepends `[CLS]` (`bos_token_id`) and appends `[SEP]` (`eos_token_id`) to each span, and `build_mask` (lines 149-158) right-pads each batch with `0`s and builds the boolean attention masks `q_mask` and `k_mask`.

### 2.3 Multi-Dataset Sampling Probabilities (`MultiDataset.set_prob`, Lines 51-81)

When pre-training across $S$ different text corpora $\mathcal{D}_1, \dots, \mathcal{D}_S$ of unequal lengths $\vert \mathcal{D}_i \vert$, `MultiDataset.set_prob(coeff)` (lines 74-81) computes the probability $\tilde{p}_i$ of drawing an example from dataset $i$ using a smoothing exponent $\alpha$ (`coeff`, controlled by `--sampling_coefficient`):

```python
    def set_prob(self, coeff=0.0):

        prob = np.array([float(len(dataset)) for _, dataset in self.datasets.items()])
        prob /= prob.sum()
        prob = np.array([p**coeff for p in prob])
        prob /= prob.sum()
        self.prob = prob
```

Mathematically, let $p_i = \frac{\vert \mathcal{D}_i \vert}{\sum_{j=1}^S \vert \mathcal{D}_j \vert}$ be the raw size fraction of dataset $i$. Raising each fraction to the power $\alpha \ge 0$ and renormalizing gives:

$$\tilde{p}_i = \frac{p_i^\alpha}{\sum_{j=1}^{S} p_j^\alpha}$$

where:
- When $\alpha = 0.0$ (the default), $p_i^0 = 1$ for every dataset, so $\tilde{p}_i = \frac{1}{S}$ (uniform sampling across all datasets regardless of size).
- When $\alpha = 1.0$, $\tilde{p}_i = p_i$ (sampling strictly proportional to dataset size).
- When $\alpha \in (0, 1)$ (such as $\alpha = 0.5$ or $0.7$), smaller datasets are upsampled relative to their raw size without completely ignoring the larger datasets.

### 2.4 Loading Wikipedia Passages for Retrieval (`load_passages`, Lines 217-234)

At the bottom of `src/data.py` (lines 217-234) is `load_passages(path)`, which is called by both `generate_passage_embeddings.py` and `passage_retrieval.py` to load the external Wikipedia corpus $\mathcal{C}_{\text{corpus}}$:

```python
def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
    return passages
```

Notice how the `else:` branch parses the standard DPR Wikipedia split `psgs_w100.tsv`: each tab-separated row has column $0 =$ `"id"`, column $1 =$ `"text"` (the $100$-word passage body), and column $2 =$ `"title"` (the Wikipedia article title), skipping the header row where `row[0] == "id"`.

---

## 3. Supervised Fine-Tuning with Hard Negatives (`src/finetuning_data.py`)

### 3.1 Sampling Random vs Hard Negatives (`sample_n_hard_negatives`, Lines 112-124)

In `reference/code/retrieval_lm/src/finetuning_data.py`, each supervised training example has three lists of candidate passages:
- `positive_ctxs`: passages that contain the answer to `example["question"]`.
- `negative_ctxs`: random passages that do not contain the answer.
- `hard_negative_ctxs`: top BM25 keyword-matched passages that share words with the question but do not contain the answer.

In `Dataset.__getitem__` (lines 33-69) and `sample_n_hard_negatives` (lines 111-123), the dataset samples $1$ `gold` passage from `positive_ctxs` and a total of `self.negative_ctxs` negative passages split between `hard_negative_ctxs` and `negative_ctxs`:

```python
    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives
```

Notice also a subtle Python bug on line 28 of `src/finetuning_data.py`:

```python
from src import normalize_text
...
class Dataset(torch.utils.data.Dataset):
    def __init__(self, ..., normalize=False):
        ...
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
```

Look at the condition `if normalize_text else lambda x: x`: instead of checking the boolean parameter `if normalize:`, the code checks `if normalize_text:`, which refers to the imported module `src.normalize_text`! Because a Python module object is always truthy, `self.normalize_fn` always executes `normalize_text.normalize` even when `normalize=False` is passed.

### 3.2 Packing Gold and Negative Passages in `Collator` (Lines 126-171)

In `Collator.__call__` (lines 131-171), given a batch of $B$ examples, all $B$ gold passages (`golds`) and all sampled negative passages (`negs`) are concatenated into one list `allpassages = golds + negs`, tokenized together in `kout`, and sliced back into gold tokens `g_tokens = k_tokens[:len(golds)]` and negative tokens `n_tokens = k_tokens[len(golds):]`.

---

## 4. Text Normalization (`src/normalize_text.py`), Options (`src/options.py`), and SLURM (`src/slurm.py`)

The remaining three files in `retrieval_lm/src/` provide text cleaning, CLI parsing, and cluster process management:

1. Unicode Punctuation Normalization (`src/normalize_text.py`):
   Wikipedia text contains many different Unicode characters for dashes, quotes, and spaces (such as en-dashes `–`, em-dashes `—`, minus signs `−`, curved quotes `“”‘’`, primes `′″`, and non-breaking spaces). If a query uses a straight ASCII hyphen `-` while a passage uses an en-dash `–`, a tokenizer might split them into different token IDs. `normalize(text)` (lines 130-161) strips invisible control characters (`CONTROLS`), maps all $11$ hyphen and minus variants (`HYPHENS | MINUSES`) to standard ASCII `-`, maps all curved double quotes to `"`, maps all single quotes, apostrophes, and accents to `'`, converts primes (`′`, `″`, `‴`) to ASCII apostrophes (`'`, `''`, `'''`), replaces `…` with `...`, and replaces Unicode fraction slashes with `/`.

2. Command-Line Options (`src/options.py`):
   Defines the `Options` class wrapping `argparse.ArgumentParser` for Contriever pre-training and fine-tuning (`--contrastive_mode`, `--queue_size`, `--temperature`, `--momentum`, `--pooling`, `--ratio_min`, `--ratio_max`, and so on), and saves a formatted table of all active parameters to `os.path.join(opt.output_dir, "opt.txt")`.

3. Distributed Initialization and SLURM Requeuing (`src/slurm.py`):
   `init_distributed_mode(params)` (lines 43-113) is called at the start of `generate_passage_embeddings.py` and `passage_retrieval.py` to configure GPU devices across three environments:
   - SLURM cluster job (`'SLURM_JOB_ID' in os.environ and not 'WORLD_SIZE' in os.environ`): reads `local_rank` from `SLURM_LOCALID`, `global_rank` from `SLURM_PROCID`, and `world_size` from `SLURM_NTASKS`, queries `scontrol show hostnames` to set `MASTER_ADDR`, and initializes `torch.distributed.init_process_group(init_method='env://', backend='nccl')`.
   - Local multi-GPU `torch.distributed.launch` job (`params.local_rank != -1`): reads `RANK` and `WORLD_SIZE` from environment variables and initializes the `nccl` process group.
   - Single-GPU job (`else:`): sets `local_rank = 0`, `global_rank = 0`, `world_size = 1`, and calls `torch.cuda.set_device(0)` without initializing `torch.distributed`.
   - Additionally, `init_signal_handler()` (lines 35-40) traps `SIGUSR1` (sent by SLURM shortly before a job hits its time limit) so that rank $0$ automatically runs `scontrol requeue $SLURM_JOB_ID` to resume training seamlessly in a new job slot.
