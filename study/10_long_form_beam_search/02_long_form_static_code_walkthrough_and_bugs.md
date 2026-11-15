# Long-Form Static Code Walkthrough and Bugs

## 1. Plain-English Intuition: Dynamic Retrieval Remapping, Hard Filtering, and Inline Citations

In Part 1, we traced how `call_model_beam_batch` in `reference/code/retrieval_lm/run_long_form_static.py` builds a segment-level beam search tree across depths $t = 1, \dots, T - 1$.

To complete our understanding of `run_long_form_static.py`, we need to look at three practical mechanisms in the code:
1. Mid-generation `[No Retrieval]` $\to$ `[Retrieval]` remapping (`run_step_generation_batch`, lines 105-141):
   In Part 1, we saw that a tree node only expands child nodes at the next depth if its `"pred"` string contains `"[Retrieval]"`.
   What happens if, after writing Sentence 1, greedy decoding outputs `"[No Retrieval]"` before Sentence 2, even though the model's soft retrieval probability at that position is $0.35 > \delta = 0.20$?
   If `"[No Retrieval]"` were left unchanged in `pred_text`, the beam search would not expand a new retrieval step for Sentence 2!
   Therefore, `run_step_generation_batch` checks the token probabilities at every `"[No Retrieval]"` position, and whenever the soft threshold $\delta$ is exceeded, rewrites `"[No Retrieval]"` to `"[Retrieval]"` in `pred_text`.
2. Hard constraint filtering (`--ignore_cont`, lines 280-289):
   In addition to soft critique scoring, Self-RAG supports a hard constraint flag `--ignore_cont` that deletes any generated sentence segment containing `"[No support / Contradictory]"` from the final output.
3. Inline citation formatting for ALCE-ASQA (lines 387-441):
   On ASQA, the official ALCE evaluation suite requires every sentence to end with bracketed citation numbers (such as `"Sentence one [1]. Sentence two [2]."`), accompanied by a `"docs"` list of the cited passages.

Below, we walk through each of these three mechanisms line by line, look at how the script makes the very first retrieval decision, and then diagnose the implementation bugs in `run_long_form_static.py`.

---

## 2. Step-by-Step Walkthrough: Mid-Generation `[No Retrieval]` Remapping (Lines 105-141)

Let us inspect lines 105-141 of `run_step_generation_batch` in `reference/code/retrieval_lm/run_long_form_static.py`:

```python
        # modify and add do retrieve tokens
        if "[No Retrieval]" in pred_text:
            ret_token_appear_indices = []
            substrings = pred_text.split("[No Retrieval]")

            for tok_idx, tok in enumerate(pred_token_ids):
                if tok == ret_tokens["[No Retrieval]"]:
                    ret_token_appear_indices.append(tok_idx)
                    substrings
                    print("retrieval_tokens")

            ret_token_score_dict = {}
            retrieval_remap = {}
            for order, idx in enumerate(ret_token_appear_indices):
                ret_token_score_dict.setdefault(order, {})
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[idx][tok_id] if tok_id in pred_log_probs[idx] else -100
                    ret_token_score_dict[order][tok] = np.exp(prob)
                if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                    do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                        ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > threshold
                else:
                    do_retrieve = 0.0
                if do_retrieve > threshold:
                    retrieval_remap[order] = True
                else:
                    retrieval_remap[order] = False
            processed_pred = ""
            for substr_i, substring in enumerate(substrings):
                if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                    processed_pred += substring + "[Retrieval]"
                else:
                    processed_pred += substring + "[No Retrieval]"
            pred_text = processed_pred
            final_preds.append(pred_text)
        else:
            final_preds.append(pred_text)
```

Let us trace how this block transforms a concrete `pred_text` when `threshold = 0.2`:
1. Suppose vLLM generates:
   `pred_text = "[Relevant]First sentence.[Fully supported][No Retrieval]Second sentence.[Utility:5]</s>"`
2. `substrings = pred_text.split("[No Retrieval]")` splits `pred_text` into $2$ pieces:
   - `substrings[0]` = `"[Relevant]First sentence.[Fully supported]"`
   - `substrings[1]` = `"Second sentence.[Utility:5]</s>"`
3. The loop over `pred_token_ids` finds the token index `idx` where `ret_tokens["[No Retrieval]"]` ($32000$) was generated (`order = 0`).
4. At position `idx`, lines 118-125 exponentiate the log-probabilities of `[Retrieval]`, `[No Retrieval]`, and `[Continue to Use Evidence]` via `np.exp(prob)` and check whether:

   $$\frac{p_{\text{idx}}(\text{[Retrieval]}) + p_{\text{idx}}(\text{[Continue to Use Evidence]})}{p_{\text{idx}}(\text{[Retrieval]}) + p_{\text{idx}}(\text{[No Retrieval]})} > \text{threshold}$$

5. If this ratio exceeds `threshold` (`retrieval_remap[0] = True`), lines 132-137 re-join `substrings[0]` with `"[Retrieval]"` instead of `"[No Retrieval]"`:
   `pred_text` becomes `"[Relevant]First sentence.[Fully supported][Retrieval]Second sentence.[Utility:5]</s>[No Retrieval]"`.
   Notice the stray `"[No Retrieval]"` at the very end: the loop appends a separator after every substring, including the last one. It is harmless here because everything after `"[Retrieval]"` is discarded in the next step, and `postprocess` strips `"[No Retrieval]"` anyway.
6. When `pred_text` returns to `call_model_beam_batch` (lines 238-242), `if "[Retrieval]" in pred:` is now `True`! The tree node truncates `processed_pred` right before `"[Retrieval]"` (`"[Relevant]First sentence.[Fully supported]"`) and, at the next tree depth, expands this node again over the same $K$ pre-retrieved passages to write `"Second sentence"` (no new retriever call is made).

### The Initial Retrieval Decision at the Root (Lines 159-198)

The remapping above handles retrieval decisions after the first sentence. The decision before any text is generated is made once per question in `call_model_beam_batch`:
- With `--mode always_retrieve` (the README's ASQA command), the tree is always built.
- With any other mode except `no_retrieval` (the README's FactScore command passes no `--mode`, so it uses the argparse default `"default"`), lines 172-191 generate up to $25$ tokens and retrieve only if the generated text already contains `"[Retrieval]"` and the normalized probability $p(\text{[Retrieval]}) / (p(\text{[Retrieval]}) + p(\text{[No Retrieval]}))$ at the first position exceeds the threshold. If `--threshold` is not given, it never retrieves.
- This is stricter than Appendix A.3, where the probability test alone triggers retrieval even when `[Retrieval]` is not the greedy token: here the threshold can only suppress retrieval, never add it. The probability is also read at position $0$ even if `"[Retrieval]"` appears later in the $25$ tokens.
- With `--mode no_retrieval`, lines 159-165 return `prediction_tree` before it is defined, which raises `UnboundLocalError`. (The paper does not report a long-form no-retrieval ablation; the ASQA cells for it in Figure 3a are empty.)

---

## 3. Hard Constraint Filtering (`--ignore_cont`) and Citation Formatting (Lines 280-441)

### 3.1 Hard Filtering of Contradictory Segments (`--ignore_cont`, Lines 280-289)

In Section 3.3 of the paper, the authors note that in addition to soft probability re-ranking, Self-RAG can enforce hard constraints by filtering out any segment where the model predicts an undesirable critique token such as `[No support / Contradictory]`.

In `call_model_beam_batch` (lines 280-289), this hard filter is controlled by the `--ignore_cont` flag during bottom-up path assembly:

```python
    for path_i, nodes in best_selections.items():
        final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
        splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
        original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
        ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
            ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
```

When `--ignore_cont` is passed (`ignore_cont is True`), any node along the winning path whose `"processed_pred"` contains `"[No support / Contradictory]"` is excluded from `final_prediction`, `splitted_sentences`, `original_splitted_sentences`, and `ctxs`.

### 3.2 FactScore vs ALCE-ASQA Postprocessing (`main`, Lines 372-441)

At the bottom of `run_long_form_static.py`, `main()` formats the winning beam path (`path_i = 0`) differently depending on `--task`:

1. Biography Generation (`--task factscore`, lines 372-385):
   FactScore evaluates plain-text biographies without bracketed citation numbers. Line 378 calls `postprocessed_result = fix_spacing(postprocess(result[0]))` to strip all reflection tokens from the top beam `result[0]` and fix period spacing.
   This path breaks whenever the root decides not to retrieve. In that case `call_model_beam_batch` returns a plain string and an empty `prediction_tree`, so `result[0]` is only the first character of the answer, and line 380 then raises `KeyError` on `intermediate["original_splitted_sentences"]`. Because the README's FactScore command uses the adaptive default mode, any biography prompt without retrieval stops the run.

2. Ambiguous Long-Form QA (`--task asqa` or `eli5`, lines 387-441):
   ALCE requires inline 1-indexed citations (`[1]`, `[2]`, ...) matching the `"docs"` list attached to each example. Let us trace lines 404-426 carefully:

```python
                if len(postprocess(final_pred[0])) == 0:
                    intermediate["splitted_sentences"][0], intermediate["ctxs"][
                        0] = intermediate["splitted_sentences"][1], intermediate["ctxs"][1]
                for idx, (sent, doc) in enumerate(zip(intermediate["splitted_sentences"][0], intermediate["ctxs"][0])):
                    if len(sent) == 0:
                        continue
                    postprocessed_result = postprocess(sent)
                    if postprocessed_result in prev_gen:
                        continue
                    else:
                        prev_gen.append(postprocessed_result)
                    final_output += postprocessed_result[:-
                                                         1] + " [{}]".format(idx) + ". "
                    docs.append(doc)
                if len(final_output) == 0:
                    item["output"] = fix_spacing(final_output)
                if len(final_output) > 0 and final_output[-1] == " ":
                    final_output = final_output[:-1]
                item["output"] = fix_spacing(final_output)
                item["output"] = item["output"].replace(
                    ".[Continue to Use Evidence]", " [1]. ")
                item["output"] = item["output"].replace(". [1] ", " [1]. ")
            item["docs"] = docs
```

Look at how the citation index `idx` works in this loop:
- Recall from Part 1 that `best_selections[0]` starts with Root Node `0` (`[None, 0, 1, 5]`), and for Root Node `0`, `prediction_tree[0]["processed_pred"]` is the empty string `""`!
- Therefore, in `intermediate["splitted_sentences"][0]`:
  - Element `idx = 0` is Root Node `0`, where `sent == ""`. Line 408 (`if len(sent) == 0: continue`) skips `idx = 0`!
  - Element `idx = 1` is the first generated sentence segment!
- Because `idx = 0` was skipped, the first real sentence has `idx = 1`! Line 415 slices off the final period (`postprocessed_result[:-1]`), appends `" [1]. "` (`" [{}]".format(idx) + ". "`), and appends its passage `doc` to `docs` (at index `0` of `docs`, which ALCE's 1-indexed citation evaluator maps to `[1]`).
- Line 411 (`if postprocessed_result in prev_gen: continue`) checks whether the exact same sentence was already generated at an earlier depth, skipping duplicate sentences.
- Citation numbering bug: the number written into the text is the path position `idx`, but `docs` grows only when a sentence is kept. Once a non-root sentence is skipped (empty or duplicate), every later `[idx]` points one or more entries past its own document in `docs` (for example, a path whose second sentence is a duplicate writes `[3]` for its third sentence while `docs` has only two entries). Only paths with no skipped sentences are cited correctly.
- Finally, lines 423-424 replace any unstripped `".[Continue to Use Evidence]"` tag with `" [1]. "`, which always cites the first document regardless of which passage the continued sentence relied on.

---

## 4. Detailed Diagnosis of the Implementation Bugs in `run_long_form_static.py`

Inspecting `run_long_form_static.py` line by line reveals the following implementation bugs in the long-form script (the root-decision and citation problems were covered above):

### 4.1 Bug 1: `grd_tokens` Loop Typo That Zeroes Out `utility_score` (Line 68)

Look closely at lines 61-87 of `run_long_form_static.py`:

```python
        utility_token_appear_indices = []
        if ut_tokens is not None:
            for tok_idx, tok in enumerate(pred_token_ids):
                if tok in list(ut_tokens.values()):
                    utility_token_appear_indices.append(tok_idx)
            if len(utility_token_appear_indices) > 0:
                idx = utility_token_appear_indices[0]
                for token, token_id in grd_tokens.items():
                    prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                    ut_score_dict[p_idx][token] = np.exp(prob)
...
        if len(ut_score_dict[p_idx]) == 5:
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum)
                                   if "[Utility:{}]".format(i+1) in ut_score_dict[p_idx] else 0.0 for i in range(0, 5)])
        else:
            utility_score = 0.0
```

Compare line 68 (`for token, token_id in grd_tokens.items():`) with `run_short_form.py` line 130 (`for token, token_id in ut_tokens.items():`):
1. In `run_long_form_static.py`, line 68 was copy-pasted from the `grd_tokens` block (line 57), so it loops over `grd_tokens.items()` ($3$ tokens: `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`) instead of `ut_tokens.items()` ($5$ tokens: `[Utility:1]`..`[Utility:5]`)!
2. Because `grd_tokens` has only $3$ items, `len(ut_score_dict[p_idx])` is at most $3$ (never $5$).
3. Therefore, the condition `if len(ut_score_dict[p_idx]) == 5:` on line 82 is always `False`, and `utility_score` always falls into the `else:` branch (`utility_score = 0.0`) across all long-form evaluations!

### 4.2 Bug 2: `use_seqscore` Not Forwarded to `run_step_generation_batch` (Line 224)

In `call_model_beam_batch` (lines 221-224):

```python
preds, scores, overall_score_dict = run_step_generation_batch(
    model, prompt + prev_generation, ctxs, max_new_tokens,
    rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
    threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use)
```

Even though `call_model_beam_batch` accepts `use_seqscore=False` (line 149) and `main()` passes `use_seqscore=args.use_seqscore` (line 362), the call on lines 221-224 omits `use_seqscore=use_seqscore` when calling `run_step_generation_batch`. As a result, `use_seqscore` inside `run_step_generation_batch` (line 13) always remains `False`, so `final_score` is computed purely from the critique scores (`w_rel * relevance_score + w_sup * ground_score + w_use * utility_score`).

### 4.3 Bug 3: Redundant Boolean Comparison `if do_retrieve > threshold:` (Lines 123-128)

Look at lines 123-130 inside `run_step_generation_batch`:

```python
                if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                    do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                        ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > threshold
                else:
                    do_retrieve = 0.0
                if do_retrieve > threshold:
                    retrieval_remap[order] = True
```

Line 125 already ends with `> threshold`, making `do_retrieve` a Python boolean (`True` or `False`). Line 128 then compares `if do_retrieve > threshold:` a second time! Because `True == 1.0` and `False == 0.0` in Python numeric comparisons, for any threshold $\delta \in [0, 1)$, `True > threshold` ($1.0 > \delta$) evaluates to `True` and `False > threshold` ($0.0 > \delta$) evaluates to `False`.

### 4.4 Bug 4: `current_level += 1` During Bottom-Up Tree Traversal (Line 274)

In the bottom-up traversal loop (lines 270-274):

```python
        while current_level > 0 and current_node is not None:
            parent = prediction_tree[current_node]["parent"]
            best_selections[path_i] = [parent] + best_selections[path_i]
            current_node = parent
            current_level += 1
```

Instead of decrementing `current_level -= 1` as it walks up the tree toward level $0$, line 274 increments `current_level += 1`, so `current_level > 0` stays `True` forever. Fortunately, the second condition `and current_node is not None` terminates the loop as soon as `current_node` reaches the root node (`node_id = 0`), whose `"parent"` is `None`.

### 4.5 Bug 5: Hardcoded `w_rel`, `w_sup`, `w_use` in `generate()` (Line 364)

In `main()` (lines 357-364):

```python
    def generate(prompt, ctxs, max_new_tokens):
        processed_prompt = PROMPT_DICT["prompt_no_input"].format_map(
            {"instruction": prompt})
        return call_model_beam_batch(processed_prompt, model=model, max_new_tokens=max_new_tokens, ctxs=ctxs, query=prompt,
                                     rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                     use_seqscore=args.use_seqscore, threshold=args.threshold, 
                                     beam_width=args.beam_width, max_depth=args.max_depth,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode=args.mode, ignore_cont=args.ignore_cont, )
```

Although `--w_rel`, `--w_sup`, and `--w_use` are registered as CLI flags on lines 329-334, line 364 passes hardcoded literals `w_rel=1.0, w_sup=1.0, w_use=0.5` instead of `w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use`.

### 4.6 Bug 6: Sibling Text Leaks into Stored Prompts (Lines 217-243)

Inside the depth loop, `prev_generation` holds the parent's text (line 217) and is used to build the prompt passed to `run_step_generation_batch`. The child loop then reuses the same variable:

```python
                        for i, result in retrieval_results.items():
                            node_id += 1
                            ...
                            prediction_tree[node_id] = {"prompt": prompt + prev_generation, "pred": pred,
                                                        "score": node_score, "ctx": ctxs[i], "parent": node,
                                                        "overall_score_dict": overall_score_dict}

                            if "[Retrieval]" in pred:
                                gen_result_index = pred.index("[Retrieval]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
```

After the first child is stored, `prev_generation` is overwritten with that child's own segment, so the second child's `"prompt"` (line 234) is the parent prompt plus the first child's segment, the third child's prompt contains the second child's segment, and so on. Only the first child of each node stores the correct context. When a later child survives pruning and is expanded at the next depth, the model is conditioned on a sentence from a sibling branch that never appears in the final output. The candidates scored at the current depth are generated correctly (the generation call happens before the loop); the error affects every deeper expansion of non-first children.

### 4.7 Bug 7: A Missing Relevance Log-Probability Is Used as a Probability (Lines 41-46)

```python
        for tok, id in rel_tokens.items():
            if id not in pred_log_probs[0]:
                prob = -100
            else:
                prob = np.exp(pred_log_probs[0][id])
            relevance_score_dict[p_idx][tok] = prob
```

When a relevance token is missing from the returned log-probabilities, the code stores `-100` directly as a probability (instead of $\exp(-100) \approx 0$), so `relevance_score` becomes a meaningless negative or out-of-range number. With `logprobs=32000` out of $32016$ tokens this is rare, but `run_short_form.py` handles the same case correctly.

### 4.8 Combined Effect on the Paper's Long-Form Results

Taken together, Bugs 1, 2, and 5 mean that every long-form run of the released script ranks segments by exactly $1.0 \cdot s_t(\text{IsRel}) + 1.0 \cdot s_t(\text{IsSup})$: the utility score, the sequence probability, and the command-line weights have no effect. In particular, Figure 3b (which varies $w_{\text{sup}}$) cannot be reproduced with this script without editing it, and the paper's stated default $w_{\text{use}} = 0.5$ plays no role in long-form decoding.
