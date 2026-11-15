# Segment-Level Beam Search Algorithm and Tree Trace

## 1. Plain-English Intuition: Why Long-Form Generation Requires a Segment-Level Tree Search

In Module 09 (`run_short_form.py`), we evaluated short-form tasks where the entire answer was a single sentence or phrase (`"October 23, 2016"`). The model only needed to retrieve once, score $K$ passages in parallel, and pick the best one.

Now consider a long-form generation task, such as writing a biography (`"Tell me a bio about Rory Tapner"`, evaluated by FactScore) or answering an ambiguous multi-aspect question (`"When did Sunday become the seventh day of the week?"` on ASQA, where every sentence must cite the specific passage `[1]`, `[2]` that supports it).

Why is single-step passage selection not enough for long-form generation?
1. Different sentences need different passages:
   Suppose the retriever returns $K = 3$ passages about Rory Tapner:
   - Passage $d_0$ covers his birth date (`"born 30 September 1959"`).
   - Passage $d_1$ covers his career as CEO of Coutts (`"appointed CEO of Coutts in September 2010; stepped down in February 2015"`).
   - Passage $d_2$ covers his university board roles (`"Council of the University of Buckingham"`).
   To write a complete 3-sentence biography, Sentence $1$ needs to condition on $d_0$, Sentence $2$ needs to condition on $d_1$, and Sentence $3$ needs to condition on $d_2$!
2. Why greedy sentence selection ($B = 1$) can get trapped:
   Suppose at Sentence $1$, Passage $d_0$ produces a sentence with score $2.80$, and Passage $d_2$ produces a sentence with score $2.20$. If we immediately threw away Passage $d_2$'s branch at Sentence $1$ (greedy search, beam width $B = 1$), we could never explore whether starting with Sentence $1$ from $d_2$ leads to a much higher-scoring Sentence $2$ and Sentence $3$!

Self-RAG solves both problems by running segment-level beam search (implemented in `call_model_beam_batch` inside `reference/code/retrieval_lm/run_long_form_static.py`). The released script is "static": it receives one fixed list of $K$ pre-retrieved passages per question (`ctxs`) and reuses that same list at every depth, without calling the retriever again. This differs from Algorithm 1 and Figure 1, where each retrieval step queries $\mathcal{R}$ with the current context; the README notes that only this pre-retrieved-passage version of long-form inference is released.
- Instead of searching over individual words, each node in the search tree is a complete sentence segment $y_t^{(k)}$ conditioned on one specific passage $d_k \in \{d_0, \dots, d_{K-1}\}$.
- At each sentence depth $t = 1, 2, \dots, T - 1$, the decoder keeps the top $B$ (`beam_width = 2`) highest-scoring partial trajectories, expands each of those $B$ trajectories across all $K$ passages ($B \times K$ candidate child nodes), scores them using the critique score from Module 09, prunes back to the top $B$ nodes, and repeats until generation terminates or reaches `max_depth`.

---

## 2. Token-Level Beam Search vs Self-RAG Segment-Level Beam Search

To keep the tree structure clear, compare traditional token-level beam search with Self-RAG's segment-level beam search:

| Property | Traditional Token-Level Beam Search | Self-RAG Segment-Level Beam Search (`run_long_form_static.py`) |
|---|---|---|
| What a single tree node represents | A single vocabulary token $y_j \in \mathcal{V}$ | A complete sentence segment $y_t^{(k)}$ + reflection tokens conditioned on passage $d_k$ |
| How tokens inside a segment are decoded | Explores $B$ token beams at every word | Greedy token decoding (`temperature = 0.0`) inside each passage branch |
| Branching factor per active node | Vocabulary size $V = 32016$ | Number of retrieved passages $K = \text{ndocs}$ (typically $K = 5$) |
| Total candidates evaluated at depth $t$ | $B \times V$ token candidates | $B \times K$ passage-conditioned segment candidates (e.g., $2 \times 5 = 10$) |
| Tree depth $t$ | Number of tokens $n$ (e.g., $100 \dots 300$) | Number of sentence segments $T$ (`curr_depth < max_depth`, e.g., $6$ or $7$) |
| Edge score for a single step | Token log-probability $\log p_\theta(y_j \mid \cdot)$ | Segment critique score $f(y_t^{(k)}, d_k, \text{Critique})$ (`result["score"]`) |
| Cumulative path score at depth $t$ | Sum of token log-probabilities | Product of segment scores across depths (`node_score = result["score"] * parent_score`) |

---

## 3. Why Multi-Sentence Generations Are Truncated at `[Retrieval]` (`processed_pred`)

Before tracing the search tree, let us understand a crucial string-slicing step in `call_model_beam_batch` (lines 238-242 of `run_long_form_static.py`):

```python
if "[Retrieval]" in pred:
    gen_result_index = pred.index("[Retrieval]")
    prev_generation = pred[:gen_result_index]
else:
    prev_generation = pred
prediction_tree[node_id]["processed_pred"] = prev_generation
```

Why does the code slice `pred[:pred.index("[Retrieval]")]` and store that truncated string in `"processed_pred"`, while keeping the un-truncated string `pred` in `"pred"`?

Consider what happens when we pass `prompt + prev_generation + [Retrieval]<paragraph>d_k</paragraph>` into vLLM with `max_new_tokens = 100`:
1. The Generator $\mathcal{M}$ reads `<paragraph>d_k</paragraph>`, outputs `[Relevant]`, writes the first sentence `"11 of 50 state names come from persons."`, and outputs `[Fully supported]`.
2. Because it has not yet reached `</s>` or `100` tokens, $\mathcal{M}$ keeps generating! It outputs `[Retrieval]` to ask for a new passage for the second sentence, and if vLLM is not stopped, $\mathcal{M}$ will continue generating tokens after `[Retrieval]` without any new passage inserted!
3. Therefore, everything generated after `[Retrieval]` must be thrown away so that at the next tree depth (`curr_depth + 1`), we can insert a real passage `<paragraph>d_i</paragraph>` right after `[Retrieval]`.

Storing both `"pred"` (un-truncated) and `"processed_pred"` (truncated right before `"[Retrieval]"`) in `prediction_tree[node_id]` serves two purposes:
- `"pred"` still contains the substring `"[Retrieval]"`, so at the next level of the `while curr_depth < max_depth:` loop, line 219 (`if "[Retrieval]" in pred:`) sees `"[Retrieval]"` and knows this node wants to expand another level of children!
- `"processed_pred"` contains only the clean single-sentence segment `"[Relevant]11 of 50 state names come from persons.[Fully supported]"` (without `"[Retrieval]"`), which is passed as `prev_generation` to the child nodes at the next depth.

---

## 4. Step-by-Step Numerical Trace of a Segment-Level Beam Search Tree ($B = 2, K = 3, \text{max\_depth} = 3$)

Let us trace `call_model_beam_batch` (lines 148-298 of `run_long_form_static.py`) step by step on a concrete example with beam width $B = 2$ (`beam_width = 2`), $K = 3$ candidate passages ($d_0, d_1, d_2$), and `max_depth = 3` (meaning the loop `while curr_depth < max_depth:` runs for `curr_depth = 1` and `curr_depth = 2`).

### 4.1 Tree Initialization (`curr_depth = 0`, Lines 199-206)

When `do_retrieve is True`, the tree is initialized with a single root node (`node_id = 0`) at level $0$:

```python
curr_depth = 1
terminated = False
node_id = 0
prediction_tree = {}
levels = {}
prediction_tree[node_id] = {
    "prompt": prompt, "pred": "[Retrieval]",
    "processed_pred": "", "score": None, "ctx": None, "parent": None
}
levels[0] = [0]
```

Here, `levels[0] = [0]` holds the active node IDs at depth $0$. Notice that `prediction_tree[0]["pred"]` is initialized to `"[Retrieval]"` so that the depth-1 loop immediately expands Node $0$, and `prediction_tree[0]["score"]` is `None`.

### 4.2 Depth 1 Expansion and Pruning (`curr_depth = 1`)

In Iteration 1 of `while curr_depth < max_depth:` (`curr_depth = 1 < 3`):
1. The loop iterates over `for node in levels[0]:` (which is only `node = 0`).
2. Because `"[Retrieval]" in prediction_tree[0]["pred"]` is `True`, it calls `run_step_generation_batch` with `prompt + ""` and the $K = 3$ passages `[d_0, d_1, d_2]`.
3. This creates $3$ new child nodes (`node_id = 1, 2, 3`) with `parent = 0`. Because parent Node $0$ has `score = None`, lines 231-232 (`node_score = result["score"] * score if score is not None else result["score"]`) set each child's `node_score` equal to its own step score `result["score"]`:

| `node_id` | `parent` | Passage `ctx` | Raw `pred` Generated | Truncated `processed_pred` | Step Score | Cumulative `node_score` | Kept in Top $B=2$? |
|---|---|---|---|---|---|---|---|
| $1$ | $0$ | $d_0$ | `"[Relevant]Sent 1A.[Fully supported][Retrieval]..."` | `"[Relevant]Sent 1A.[Fully supported]"` | $2.80$ | $2.80$ | Yes ($1$st) |
| $2$ | $0$ | $d_1$ | `"[Irrelevant]Sent 1B."` | `"[Irrelevant]Sent 1B."` | $0.60$ | $0.60$ | No (Pruned) |
| $3$ | $0$ | $d_2$ | `"[Relevant]Sent 1C.[Partially supported][Retrieval]..."` | `"[Relevant]Sent 1C.[Partially supported]"` | $2.20$ | $2.20$ | Yes ($2$nd) |

4. Beam pruning at Depth 1 (lines 246-252):
   `levels[1]` initially contains `[1, 2, 3]`. Sorting by `prediction_tree[node_id]["score"]` in descending order and slicing `[:beam_width]` ($B = 2$) prunes Node $2$ ($0.60$) and updates:

   $$\text{levels}[1] = [1, 3]$$

   Then `curr_depth` increments to $2$.

### 4.3 Depth 2 Expansion and Pruning (`curr_depth = 2`)

In Iteration 2 of `while curr_depth < max_depth:` (`curr_depth = 2 < 3`):
1. The loop iterates over `for node in levels[1]:`, which visits `node = 1` first and `node = 3` second.
2. Expanding `node = 1` (parent `score = 2.80`, prefix `P0 + "[Relevant]Sent 1A.[Fully supported]"`):
   Because `"[Retrieval]" in prediction_tree[1]["pred"]` is `True`, calling `run_step_generation_batch` across $d_0, d_1, d_2$ creates $3$ child nodes (`node_id = 4, 5, 6`) with `parent = 1`.
   On lines 231-232, each child's cumulative `node_score` is the product of its step score and its parent's score ($2.80$):

   $$\text{node\_score}(\text{child}) = \text{score}_{\text{step}}(\text{child}) \times \text{node\_score}(\text{parent})$$

3. Expanding `node = 3` (parent `score = 2.20`, intended prefix `P0 + "[Relevant]Sent 1C.[Partially supported]"`):
   Because `"[Retrieval]" in prediction_tree[3]["pred"]` is also `True`, calling `run_step_generation_batch` across $d_0, d_1, d_2$ creates $3$ more child nodes (`node_id = 7, 8, 9`) with `parent = 3`, each multiplied by parent score $2.20$. (This is the intended context. Because of a variable-reuse bug covered in Part 2, the code actually stores Node $3$'s prompt with its sibling Node $2$'s sentence appended, so the real prefix is `P0 + "[Irrelevant]Sent 1B." + "[Relevant]Sent 1C.[Partially supported]"`.)

| `node_id` | `parent` | Parent Score | Passage `ctx` | Truncated `processed_pred` | Step Score | Cumulative `node_score` (`step * parent`) | Rank Among All 6 Depth-2 Nodes |
|---|---|---|---|---|---|---|---|
| $4$ | $1$ | $2.80$ | $d_0$ | `"[Relevant]Sent 2A.[Partially supported]"` | $1.50$ | $1.50 \times 2.80 = 4.20$ | $4$th (Pruned) |
| $5$ | $1$ | $2.80$ | $d_1$ | `"[Relevant]Sent 2B.[Fully supported][Utility:5]</s>"` | $2.50$ | $2.50 \times 2.80 = 7.00$ | $1$st (Kept) |
| $6$ | $1$ | $2.80$ | $d_2$ | `"[Relevant]Sent 2C.[Partially supported]"` | $2.00$ | $2.00 \times 2.80 = 5.60$ | $3$rd (Pruned) |
| $7$ | $3$ | $2.20$ | $d_0$ | `"[Relevant]Sent 2D.[Fully supported][Utility:5]</s>"` | $2.90$ | $2.90 \times 2.20 = 6.38$ | $2$nd (Kept) |
| $8$ | $3$ | $2.20$ | $d_1$ | `"[Irrelevant]Sent 2E."` | $1.00$ | $1.00 \times 2.20 = 2.20$ | $6$th (Pruned) |
| $9$ | $3$ | $2.20$ | $d_2$ | `"[Relevant]Sent 2F.[Partially supported]"` | $1.80$ | $1.80 \times 2.20 = 3.96$ | $5$th (Pruned) |

4. Beam pruning at Depth 2 (lines 246-252):
   Out of all $6$ nodes in `levels[2] = [4, 5, 6, 7, 8, 9]`, the top $B = 2$ cumulative scores are Node $5$ ($7.00$) and Node $7$ ($6.38$). Therefore:

   $$\text{levels}[2] = [5, 7]$$

   Notice why maintaining a beam of width $B = 2$ at Depth 1 was valuable: Node $7$ (which branched off $2$nd-place Node $3$) achieved a higher 2-sentence cumulative score ($6.38$) than both Node $6$ ($5.60$) and Node $4$ ($4.20$) from $1$st-place Node $1$!
   Finally, `curr_depth` increments to $3$. Because `curr_depth < max_depth` ($3 < 3$) is now `False`, the `while` loop terminates.

### 4.4 Bottom-Up Path Reconstruction (`best_selections`, Lines 256-289)

Once the tree expansion loop finishes, how does `call_model_beam_batch` reconstruct the full multi-sentence sequence from the winning leaf nodes in the deepest level `levels[2] = [5, 7]`?

Lines 260-274 start at each leaf node in `levels[len(levels)]` (`levels[2] = [5, 7]`) and walk backward up the `"parent"` pointers until reaching the root (`parent = None`):

```python
    # Traverse from the bottom
    levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
    for path_i, node in enumerate(levels[len(levels)]):
        if node == 0:
            break
        best_selections[path_i] = [node]
        current_node = node
        current_level = curr_depth
        if current_node is None:
            continue
        while current_level > 0 and current_node is not None:
            parent = prediction_tree[current_node]["parent"]
            best_selections[path_i] = [parent] + best_selections[path_i]
            current_node = parent
            current_level += 1
```

Let us trace this backward pointer walk for our top leaf node (`path_i = 0`, `node = 5`):
1. Start: `best_selections[0] = [5]`, `current_node = 5`.
2. Step 1: `parent = prediction_tree[5]["parent"] = 1`. Prepend `1` to the front of the list:
   `best_selections[0] = [1, 5]`, `current_node = 1`.
3. Step 2: `parent = prediction_tree[1]["parent"] = 0`. Prepend `0` to the front of the list:
   `best_selections[0] = [0, 1, 5]`, `current_node = 0`.
4. Step 3: `parent = prediction_tree[0]["parent"] = None`. Prepend `None` to the front of the list:
   `best_selections[0] = [None, 0, 1, 5]`, `current_node = None`.
5. Now `current_node is not None` is `False`, so the `while` loop stops!

Finally, lines 280-289 iterate through the non-`None` nodes `[0, 1, 5]` in `best_selections[0]`:
- Joining `processed_pred` across nodes $0, 1, 5$ gives:
  `"" + " " + "[Relevant]Sent 1A.[Fully supported]" + " " + "[Relevant]Sent 2B.[Fully supported][Utility:5]</s>"`
- Collecting `ctx` across nodes $0, 1, 5$ gives `[None, d_0, d_1]`: the root contributes `None`, followed by the cited passages for Sentence $1$ and Sentence $2$.
