# Passage Embedding, FAISS Indexing, and Search

## 1. Plain-English Intuition: From a Trained Encoder to a 21-Million-Passage Search Engine

In Parts 1 through 3, we saw how the Contriever bi-encoder $E_\theta(\cdot)$ is built and trained so that a query $q$ and a relevant passage $d$ map to $768$-dimensional vectors $\mathbf{q}, \mathbf{d} \in \mathbb{R}^{768}$ with a high dot product $\langle \mathbf{q}, \mathbf{d} \rangle$.

Now suppose we have the trained `facebook/contriever-msmarco` model and a file containing $N = 21,015,324$ Wikipedia passages (`psgs_w100.tsv`). How do we turn that into a fast search engine for Self-RAG?

The retrieval pipeline in `reference/code/retrieval_lm/` operates in three steps:
1. Step 1 (Offline Passage Embedding in `generate_passage_embeddings.py`):
   We split the $21\text{ million}$ Wikipedia passages into shards across multiple GPUs, run `Contriever` in `fp16` on each shard, and save the resulting $(N_{\text{shard}}, 768)$ NumPy arrays and passage ID lists into binary `.pkl` files on disk.
2. Step 2 (FAISS Indexing in `src/index.py`):
   We load the pre-computed passage vectors from the `.pkl` files into a FAISS (Facebook AI Similarity Search) index in RAM (`faiss.IndexFlatIP` for exact search or `faiss.IndexPQ` for compressed quantized search), alongside a lookup list `index_id_to_db_id` that remembers which Wikipedia passage ID belongs to each row of the FAISS index.
3. Step 3 (Online Query Search in `passage_retrieval.py`):
   When Self-RAG needs to retrieve passages for a query $q$, it encodes $q$ into a single vector $\mathbf{q} \in \mathbb{R}^{768}$, asks FAISS for the $K$ rows with the highest dot products $\langle \mathbf{q}, \mathbf{d}_i \rangle$, translates those row numbers back to Wikipedia passage IDs, and returns the matching `{"id", "title", "text"}` dictionaries.

---

## 2. Step 1: Offline Passage Embedding (`generate_passage_embeddings.py`)

Let us walk through `reference/code/retrieval_lm/generate_passage_embeddings.py` line by line.

### 2.1 Sharding the Corpus Across GPUs (`main`, Lines 67-93)

Encoding $21\text{ million}$ passages on a single GPU takes several hours. To parallelize this across `args.num_shards` GPUs without inter-GPU communication overhead, `main(args)` (lines 67-93) slices a contiguous range `[start_idx : end_idx]` of the `passages` list for the current `args.shard_id` $\in \{0, 1, \dots, \text{num\_shards} - 1\}$:

```python
    passages = src.data.load_passages(args.passages)

    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
```

To see why lines 80-81 (`if args.shard_id == args.num_shards - 1: end_idx = len(passages)`) are necessary, suppose `len(passages) = 10` and `num_shards = 3`:
- Integer division gives `shard_size = 10 // 3 = 3`.
- Shard $0$ processes indices `[0 : 3]` ($3$ passages).
- Shard $1$ processes indices `[3 : 6]` ($3$ passages).
- Without line 81, Shard $2$ would process `[6 : 9]`, dropping index $9$! Setting `end_idx = len(passages)` on the final shard `num_shards - 1` ensures Shard $2$ processes `[6 : 10]` ($4$ passages), so every passage in the corpus is encoded.

### 2.2 Batched Encoding and Serialization (`embed_passages`, Lines 24-64)

Inside `embed_passages(args, passages, model, tokenizer)` (lines 24-64):

```python
def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in enumerate(passages):
            batch_ids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p["text"]
            else:
                text = p["title"] + " " + p["text"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
...
    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings
```

Tracing each step of `embed_passages`:
1. For each passage dictionary `p`, its Wikipedia article title and body text are joined with a space: `text = p["title"] + " " + p["text"]`.
2. Whenever `batch_text` reaches `per_gpu_batch_size` (default $512$) or hits the last passage in the shard (`k == len(passages) - 1`), `tokenizer.batch_encode_plus` tokenizes and pads the batch up to `passage_maxlength` (default $512$).
3. `embeddings = model(**encoded_batch)` runs `Contriever.forward` (in `fp16` because `model = model.half()` was called on line 73), returning a `[B, 768]` tensor of masked-average-pooled vectors, which are moved to CPU RAM (`embeddings.cpu()`).
4. At the end of the shard, `torch.cat(allembeddings, dim=0).numpy()` stacks all batch tensors into a single 2D NumPy array of shape $[N_{\text{shard}}, 768]$, and lines 89-90 save the tuple `(allids, allembeddings)` to `wikipedia_embeddings/passages_{shard_id:02d}` using `pickle.dump`.

---

## 3. Step 2: FAISS Vector Indexing (`src/index.py`)

### 3.1 Memory Math: Exact Search (`IndexFlatIP`) vs Product Quantization (`IndexPQ`)

Let us calculate how much RAM is required to hold $N = 21,015,324$ vectors of dimension $D = 768$ in 32-bit floating-point (`float32`, which uses $4$ bytes per number):
- Bytes per vector $= 768 \times 4\text{ bytes} = 3,072\text{ bytes}$ ($3\text{ KB}$).
- Total RAM for $21,015,324$ vectors $= 21,015,324 \times 3,072\text{ bytes} \approx 64.56\text{ GB}$.
- Adding the raw text dictionary `self.passage_id_map` in Python brings total system RAM usage to roughly $100\text{ GB}$ (as noted in `reference/code/README.md` line 185).

To support both high-RAM servers (exact search) and lower-RAM machines (compressed search), `Indexer.__init__` in `reference/code/retrieval_lm/src/index.py` (lines 17-24) supports two FAISS index types depending on `n_subquantizers`:

```python
class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        self.index_id_to_db_id = []
```

1. Exact Inner-Product Search (`n_subquantizers == 0`, the default):
   `faiss.IndexFlatIP(768)` (`Flat` = uncompressed vectors, `IP` = Inner Product) stores the exact `float32` vectors. Because `IndexFlatIP` does no clustering or compression, `self.index.is_trained` is already `True`, and `self.index.search(q, top_docs)` computes the exact dot product $\langle \mathbf{q}, \mathbf{d}_i \rangle$ against all $N$ vectors.

2. Product Quantization (`n_subquantizers > 0`):
   How does `faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)` compress a $3,072$-byte vector down to just $M = \text{n\_subquantizers}$ bytes?
   - Step A (Split vector into $M$ sub-vectors):
     Divide the $D = 768$ coordinates of each passage vector $\mathbf{d} \in \mathbb{R}^{768}$ into $M$ smaller sub-vectors of length $D^* = D / M$ (for example, if $M = 96$, each sub-vector has $D^* = 768 / 96 = 8$ numbers):

     $$\mathbf{d} = \left[\mathbf{d}^{(1)},\; \mathbf{d}^{(2)},\; \dots,\; \mathbf{d}^{(M)}\right], \quad \text{where } \mathbf{d}^{(m)} \in \mathbb{R}^{D^*}$$

   - Step B (Learn $2^{n_{\text{bits}}}$ centroids per sub-space via `self.index.train`):
     For $n_{\text{bits}} = 8$, $2^8 = 256$. During `self.index.train(embeddings)`, FAISS runs $k$-means clustering in each sub-space $m \in \{1, \dots, M\}$ to find $256$ representative centroid vectors $\mathbf{c}_{m, 0}, \mathbf{c}_{m, 1}, \dots, \mathbf{c}_{m, 255} \in \mathbb{R}^{D^*}$.
   - Step C (Compress each sub-vector to a 1-byte centroid ID via `self.index.add`):
     Each sub-vector $\mathbf{d}^{(m)} \in \mathbb{R}^{D^*}$ is replaced by the 1-byte integer index $q_m(\mathbf{d}) \in \{0, 1, \dots, 255\}$ of its closest centroid $\mathbf{c}_{m, q_m(\mathbf{d})}$. Storing $M = 96$ one-byte integers takes only $96\text{ bytes}$ per passage instead of $3,072\text{ bytes}$ (a $32\times$ memory reduction, shrinking $64.56\text{ GB}$ to $2\text{ GB}$!).
   - Step D (Fast lookup-table dot products during `self.index.search`):
     When a query $\mathbf{q} = [\mathbf{q}^{(1)}, \dots, \mathbf{q}^{(M)}]$ arrives, FAISS computes the dot product between $\mathbf{q}^{(m)}$ and all $256$ centroids in each sub-space, storing them in an $M \times 256$ lookup table $T[m, c] = \langle \mathbf{q}^{(m)}, \mathbf{c}_{m,c} \rangle$. The approximate dot product with any passage $\mathbf{d}$ is then the sum of $M$ table lookups:

     $$\langle \mathbf{q}, \mathbf{d} \rangle = \sum_{m=1}^{M} \left\langle \mathbf{q}^{(m)}, \mathbf{d}^{(m)} \right\rangle \approx \sum_{m=1}^{M} \left\langle \mathbf{q}^{(m)}, \mathbf{c}_{m, q_m(\mathbf{d})} \right\rangle = \sum_{m=1}^{M} T\left[m, q_m(\mathbf{d})\right]$$

### 3.2 Translating FAISS Internal Indices to Passage IDs (`index_id_to_db_id`, Lines 25-66)

Inside FAISS, vectors are always indexed by sequential integers $0, 1, 2, \dots, N - 1$ in the exact order they are passed to `self.index.add(embeddings)`. However, the passage IDs in `allids` (from the `.pkl` shard files) are external database strings (such as `"10492"` or `"wiki:500"`), and the shard files might be loaded in any order.

Look at how `index_data` and `search_knn` in `src/index.py` (lines 25-46) maintain the mapping between FAISS's internal position $i \in \{0, \dots, N-1\}$ and the external passage ID string:

```python
    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result
```

Tracing `index_data` and `search_knn`:
1. Every time a batch of `ids` and `embeddings` is added via `index_data`, `self._update_id_mapping(ids)` appends `ids` to the Python list `self.index_id_to_db_id` in the exact same order that `self.index.add(embeddings.astype('float32'))` appends the vectors to the FAISS index (note that FAISS requires `float32` input arrays).
2. During `search_knn`, `scores, indexes = self.index.search(q, top_docs)` returns two 2D arrays of shape $[B_q, K]$: `scores` (the top-$K$ dot products) and `indexes` (the internal FAISS row numbers $i \in \{0, \dots, N-1\}$).
3. `db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]` looks up each internal integer `i` in `self.index_id_to_db_id` to recover the external string ID `doc_id`.
4. `serialize` and `deserialize_from` (lines 48-66) save and load both the FAISS C++ index (`index.faiss`) and the Python ID list (`index_meta.faiss`), asserting on load that `len(self.index_id_to_db_id) == self.index.ntotal`.

---

## 4. Step 3: Online Search (`passage_retrieval.py`) and Retrieval Evaluation (`src/evaluation.py`, `src/beir_utils.py`)

### 4.1 Setting Up and Querying the Retriever (`passage_retrieval.py`)

In `reference/code/retrieval_lm/passage_retrieval.py`, the `Retriever` class ties `Contriever`, `Indexer`, and `load_passages` together:
1. `setup_retriever` (lines 118-149):
   - Loads `Contriever` (`facebook/contriever-msmarco`) onto the GPU in `fp16` (`self.model.half()`).
   - Initializes `self.index = src.index.Indexer(self.args.projection_size, self.args.n_subquantizers, self.args.n_bits)`.
   - Finds all shard files matching `self.args.passages_embeddings` (for example, `"wikipedia_embeddings/*"`), streams them via `index_encoded_data` in batches of `indexing_batch_size = 1000000` vectors, and optionally caches `index.faiss` on disk (`--save_or_load_index`).
   - Loads the text passages via `src.data.load_passages(self.args.passages)` and builds the dictionary `self.passage_id_map = {x["id"]: x for x in self.passages}`.
2. `search_document(self, query, top_n=10)` (lines 151-158):
   - Encodes `[query]` into a `[1, 768]` NumPy array `questions_embedding` via `self.embed_queries`.
   - Calls `top_ids_and_scores = self.index.search_knn(questions_embedding, self.args.n_docs)`.
   - Looks up each returned `doc_id` in `self.passage_id_map` via `self.add_passages` and returns the top `top_n` passage dictionaries `[{"id": ..., "title": ..., "text": ...}, ...]`.

For users who do not have $100\text{ GB}$ of RAM to load all $21\text{M}$ passages of `psgs_w100.tsv`, `download_demo_corpus.sh` and `setup_retriever_demo` (lines 169-195) provide a $9\text{ GB}$ demo subset (`enwiki_2020_intro_only`) containing only the introductory paragraphs of English Wikipedia articles.

Also note the Wikipedia dump distinction explained in Appendix B.2 of the paper:
- For most tasks, Self-RAG retrieves from the 2018 English Wikipedia dump (`psgs_w100.tsv`).
- However, because PopQA was constructed from 2022 WikiData and includes newer entities missing from the 2018 dump, PopQA evaluation uses the December 2020 preprocessed Wikipedia corpus from Atlas.
- For biography generation and both open-domain QA tasks (PopQA and TriviaQA), five additional documents come from Google Programmable Search; because that API returns only snippets, the authors retrieve the Wikipedia introductory paragraphs of the matching entities instead.

### 4.2 Retrieval Evaluation Metrics (`src/evaluation.py` and `src/beir_utils.py`)

Finally, let us examine how `src/evaluation.py` and `src/beir_utils.py` measure the quality of the retriever itself:

1. Answer Substring Recall@$k$ (`calculate_matches` and `has_answer` in `src/evaluation.py`, lines 43-112):
   In open-domain QA, a retrieved passage `doc['text']` is considered a "hit" if it contains at least one of the acceptable gold answer strings in `example['answers']`.
   Look at how `has_answer(answers, text, tokenizer)` (lines 95-105) tests this:

   ```python
   def has_answer(answers, text, tokenizer) -> bool:
       """Check if a document contains an answer string."""
       text = _normalize(text)
       text = tokenizer.tokenize(text, uncased=True)

       for answer in answers:
           answer = _normalize(answer)
           answer = tokenizer.tokenize(answer, uncased=True)
           for i in range(0, len(text) - len(answer) + 1):
               if answer == text[i: i + len(answer)]:
                   return True
       return False
   ```

   Why does `has_answer` tokenize `text` and `answer` with `SimpleTokenizer` instead of just checking `if answer.lower() in text.lower():`?
   Because simple character substring matching produces false positives on short words! For example, if the gold answer is `"art"`, character substring matching (`"art" in "party"`) would falsely match the word `"party"`. By splitting `text` into whole alphanumeric word tokens via `SimpleTokenizer` (`r'[\p{L}\p{N}\p{M}]+'`) and checking `answer == text[i : i + len(answer)]`, `"art"` only matches the standalone word token `"art"`.
   In `calculate_matches` (lines 43-77), `best_hit = next((i for i, x in enumerate(question_hits) if x), None)` finds the earliest rank `best_hit` $\in \{0, \dots, K-1\}$ where `has_answer` is `True`, and increments `top_k_hits[best_hit:]` by $1$ for all $k \ge \text{best\_hit}$, computing the cumulative Top-$k$ hit count across the dataset.

2. BEIR Benchmark Evaluation (`src/beir_utils.py`):
   `DenseEncoderModel` (lines 20-121) wraps `query_encoder` and `doc_encoder` in the interface expected by the BEIR library (`encode_queries` and `encode_corpus`), splitting work across GPUs via `np.array_split` and gathering variable-sized shards via `dist_utils.varsize_gather_nograd`. `evaluate_model` (lines 124-207) then computes standard IR metrics (nDCG@$k$, MAP@$k$, Recall@$k$, Precision@$k$, and MRR). Note a minor bug on line 191 of `src/beir_utils.py`: inside `elif dataset == "cqadupstack":`, `GenericDataLoader(data_folder=data_folder)` references an undefined variable `data_folder` instead of the loop variable `path`.
