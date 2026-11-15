# Passage Selection Cascade and Postprocessing

## 1. Plain-English Intuition: How Do We Pick One Passage out of Ten for Each Training Sentence?

By the end of Part 2 (`run_reward_vllm.py`), we have all five sets of Critic predictions for our seed examples:
1. Initial retrieval decision ($t = 1$): `[Retrieval]` or `[No Retrieval]`.
2. Multi-sentence retrieval decisions ($t > 1$): `[Retrieval]`, `[No Retrieval]`, or `[Continue to Use Evidence]`.
3. Overall response utility (`IsUse`): `[Utility:1]` through `[Utility:5]`.
4. Passage relevance (`IsRel`) for each of the $K = 10$ retrieved passages per sentence: `[Relevant]` or `[Irrelevant]`.
5. Factual support (`IsSup`) for each of the $K = 10$ retrieved passages per sentence: `[Fully supported]`, `[Partially supported]`, or `[No support / Contradictory]`.

Now, in the final step of Stage 3 (`reference/code/data_creation/generator/postprocess_data.py`), we must combine all of these predictions and build one single target string `output` for each training example in $\mathcal{D}_{\text{gen}}$.

Whenever a sentence $y_t$ triggers `[Retrieval]`, we have $K = 10$ candidate passages `ctxs[0] .. ctxs[9]`, but we can only insert one passage `<paragraph>d</paragraph>` into the training sequence at step $t$. Which passage should we choose?
1. Prioritizing best support: Whenever at least one retrieved passage is both `[Relevant]` and `[Fully supported]`, we prefer the highest-ranked `[Relevant]` + `[Fully supported]` passage (`fully_supported_p_indices[0]`). If none is `[Fully supported]`, we fall back to `[Relevant]` + `[Partially supported]`, and then `[Relevant]` + `[No support / Contradictory]`.
2. Why we also include `[Irrelevant]` and unsupported passages during training:
   Suppose every single training example in $\mathcal{D}_{\text{gen}}$ contained only `[Relevant]` and `[Fully supported]` passages. Then during training, the Generator $\mathcal{M}$ would see `[Relevant]` and `[Fully supported]` $100\%$ of the time, and would never learn when to predict `[Irrelevant]` or `[No support / Contradictory]`!
   At test time, however, the retriever $\mathcal{R}$ will frequently return off-topic or unsupported passages. Including a controlled fraction of `[Partially supported]`, `[No support / Contradictory]`, and `[Irrelevant]` passages in $\mathcal{D}_{\text{gen}}$ teaches $\mathcal{M}$ to recognize bad passages and down-weight them during beam search.

---

## 2. Normalizing Raw Critic Predictions (`postprocess_data.py`, Lines 66-135)

Because the Critic $\mathcal{C}$ is a generative language model, its raw decoded string `pred` occasionally lacks a leading bracket `[` or outputs `"Yes"` / `"No"` instead of `"[Retrieval]"` / `"[No Retrieval]"`. Lines 66-135 of `postprocess_data.py` define four normalization functions to map `pred` into exact special token strings:

1. `convert_score_to_utility_token(pred)` (from line 66):
   Checks if any digit `"1"` through `"5"` appears in `pred` and returns `"[Utility:{}]"`, or prepends `"["` if missing and validates that the result is in `["[Utility:1]", ..., "[Utility:5]"]`.
2. `convert_score_to_retrieval_token(pred)` (from line 88):
   Maps `"Yes"` or `"[Yes]"` to `"[Retrieval]"`, maps `"No"` or `"[No]"` to `"[No Retrieval]"`, and preserves valid tokens in `["[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"]` (defaulting any other string to `"[No Retrieval]"`).
3. `convert_score_to_groudness(pred)` (from line 105):
   Prepends `"["` if missing, keeps valid tokens in `["[No support / Contradictory]", "[Partially supported]", "[Fully supported]"]`, and maps legacy numeric scores `"4"` or `"5"` to `"[Fully supported]"`.
4. `postprocess_relevance_reward_token(pred)` (lines 126-135):
   ```python
   def postprocess_relevance_reward_token(pred):
       if len(pred) == 0:
           print("relevance token empty")
           return None
       if "Relevant" in pred:
           return "[Relevant]"
       elif "Irrelevant" in pred:
           return "[Irrelevant]"
       else:
           return None
   ```
   Notice why checking `"Relevant" in pred` before `"Irrelevant" in pred` works without false positives: in `"Irrelevant"`, the `"r"` after `"I"` is lowercase (`"rrelevant"`), whereas `"Relevant"` has an uppercase `"R"`, so `"[Irrelevant]"` does not match `"Relevant" in pred` and reaches the `elif` branch.

---

## 3. Step-by-Step Walkthrough of Sequence Assembly (`postprocess_data.py`, Lines 193-368)

Let us trace the main loop over `for q_id, instance in tqdm(input_data.items()):` (lines 193-354) in four stages.

### 3.1 Stage 1: FEVER Utility Fix and Branch 1 (`[No Retrieval]` Downsampling, Lines 205-226)

First, lines 209-212 normalize the initial retrieval token `need_retrieval_i_token` and the overall response utility token `utility_i_token`:

```python
        need_retrieval_i_token = convert_score_to_retrieval_token(need_retrieval_initial_i)
        utility_i_token = convert_score_to_utility_token(reward_utility_i)
        if instance[0]["output"] in ["true", "false"] and utility_i_token == "[Utility:1]":
            utility_i_token = "[Utility:5]"
```

Why do lines 211-212 override `[Utility:1]` to `[Utility:5]` when `output in ["true", "false"]`?
In the FEVER dataset, the ground-truth output is a single word (`"true"` or `"false"`). When the Critic evaluates `IsUse` on a 1-word answer without seeing the passage, it sometimes rates the 1-word answer as uninformative (`[Utility:1]`). Because `"true"` or `"false"` is the exact required answer for FEVER, lines 211-212 correct `[Utility:1]` to `[Utility:5]`. (This rule is not mentioned in the paper.)

Next comes Branch 1 (lines 222-226), taken when the initial retrieval decision for `q_id` is `"[No Retrieval]"`. Line 222 compares the raw Critic string `need_retrieval_initial_i`, not the normalized `need_retrieval_i_token`, so a raw prediction such as `"No"` falls through to the retrieval branch:

```python
        if need_retrieval_initial_i == "[No Retrieval]":
            processed_output = need_retrieval_i_token + output + utility_i_token
            if random.random() > 0.6:
                need_retrieval_multi_stats.append(need_retrieval_initial_i)
                processed_data.append({"instruction": instruction, "output": processed_output, "input": "", "id": q_id, "dataset_name": dataset_name})
```

Here:
- The entire target string is assembled without sentence splitting or passages: `"[No Retrieval]" + output + "[Utility:k]"`.
- Because $63.2\%$ of the seed examples come from Open-Instruct (where `[No Retrieval]` is frequent), `if random.random() > 0.6:` keeps only $40\%$ of these retrieval-free examples so they do not overwhelm the retrieval-augmented examples during Generator training.

### 3.2 Stage 2: Partitioning Candidate Passages into Four Buckets (Lines 242-287)

When `need_retrieval_initial_i` is anything other than `"[No Retrieval]"` (Branch 2, the `else:` on line 227), the script loops over each sentence `sent_idx, sentence in enumerate(splitted_output_sentences)`.
- If `skipped_sentences[str(sent_idx)] is True` (a short sentence $< 30$ characters skipped in Part 1), lines 243-245 append `output += sentence` directly without any reflection tokens and move to the next sentence.
- For `sent_idx == 0` the decision is the initial retrieval token; for later sentences it is the multi-sentence Critic prediction (lines 249-252).
- Otherwise, if the sentence-level retrieval decision `need_retrieval_decision_token` is `"[Retrieval]"`, lines 260-287 inspect all $K$ candidate passages `p_idx` $\in \{0, \dots, K-1\}$ for `sent_idx` and sort their indices into four buckets:

```python
                    fully_supported_p_indices = []
                    partially_supported_p_indices = []
                    no_support_p_indices = []
                    irrelevant_p_indices = []
                    for p_idx in range(len(relevance_p_pidx)):
                        relevance_token = postprocess_relevance_reward_token(relevance_p_pidx[p_idx]["pred"])
                        if p_idx not in groundness_p_pidx:
                            continue
                        groundness_token = convert_score_to_groudness(groundness_p_pidx[p_idx]["pred"])
                        if relevance_token is None or groundness_token is None:
                            continue
                        if relevance_token == "[Relevant]" and groundness_token == "[Fully supported]":
                            fully_supported_p_indices.append(p_idx)
                        elif relevance_token == "[Relevant]" and groundness_token == "[Partially supported]":
                            partially_supported_p_indices.append(p_idx)
                        elif relevance_token == "[Relevant]" and groundness_token == "[No support / Contradictory]":
                            no_support_p_indices.append(p_idx)
                        elif relevance_token == "[Irrelevant]":
                            irrelevant_p_indices.append(p_idx)
```

Notice that because `range(len(relevance_p_pidx))` iterates in ascending order $0, 1, \dots, K-1$ (which is descending order of Contriever similarity score), the first element `[0]` of each bucket list is always the passage with the highest retrieval similarity score in that bucket!

### 3.3 Stage 3: The Passage Selection Cascade and Code-Level Overrides (Lines 289-326)

Once the four bucket lists are populated, lines 289-326 choose a single passage index `p_idx` for sentence `sent_idx`:

```python
                    # Form the output
                    if len(fully_supported_p_indices) > 0:
                        p_idx = fully_supported_p_indices[0]
                        if dataset_name == "nq" and random.random() > 0.75 and len(irrelevant_p_indices) > 0:
                            p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]
                    elif len(fully_supported_p_indices) == 0 and len(partially_supported_p_indices) > 0 and random.random() > 0.3:
                        p_idx = partially_supported_p_indices[0]
                    elif len(fully_supported_p_indices) == 0 and len(partially_supported_p_indices) == 0 and len(no_support_p_indices) > 0 and random.random() > 0.75:
                        p_idx = no_support_p_indices[0]
                    else:
                        if len(irrelevant_p_indices) == 0:
                            continue
                        else:
                            p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]

                    if args.negative_samples is True:
                        if len(irrelevant_p_indices) == 0:
                               continue
                        p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]

                    if dataset_name in ["nq", "fever"] and len(irrelevant_p_indices) > 0:
                        p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]
...
                    relevance_token = postprocess_relevance_reward_token(relevance_p_pidx[p_idx]["pred"])
                    groundness_token = convert_score_to_groudness(groundness_p_pidx[p_idx]["pred"])

                    if args.negative_samples is True and groundness_token == "[No support / Contradictory]":
                        relevance_token = "[Irrelevant]"
                    
                    if dataset_name in ["nq", "fever", "wow"] and groundness_token == "[No support / Contradictory]" and relevance_token == "[Relevant]":
                        relevance_token = "[Irrelevant]"
```

Let us trace both the priority cascade and three subtle code behaviors in this block:

1. The priority cascade (lines 290-302):
   - Priority 1: If `fully_supported_p_indices` is non-empty, pick `fully_supported_p_indices[0]` (the top-ranked `[Relevant]` + `[Fully supported]` passage).
   - Priority 2: Else if `partially_supported_p_indices` is non-empty, pick `partially_supported_p_indices[0]` with $70\%$ probability (`random.random() > 0.3`).
   - Priority 3: Else if `no_support_p_indices` is non-empty, pick `no_support_p_indices[0]` with $25\%$ probability (`random.random() > 0.75`).
   - Priority 4: Otherwise (including the $30\%$ and $75\%$ fallthroughs from Priorities 2 and 3), sample an index from `irrelevant_p_indices`. If `irrelevant_p_indices` is empty, the `continue` on line 300 skips to the next sentence without appending anything, so this sentence disappears from the training target entirely. The same happens at the other `continue` statements inside the sentence loop (lines 246-248, 257-259, 304-306, 312-317, and 328-329), so some examples in $\mathcal{D}_{\text{gen}}$ are missing sentences of their original answer.

2. Subtle bug in `random.sample(range(len(irrelevant_p_indices)), k=1)[0]` (lines 293, 302, 307, 310):
   Look closely at how the script samples from `irrelevant_p_indices`:
   Suppose `irrelevant_p_indices = [7, 8]` (meaning passages $7$ and $8$ are `[Irrelevant]`, while passages $0 \dots 6$ are `[Relevant]`).
   Here `len(irrelevant_p_indices)` is $2$, so `range(len(irrelevant_p_indices))` is `[0, 1]`.
   Instead of indexing `irrelevant_p_indices[sampled_i]` (or writing `random.choice(irrelevant_p_indices)` to get $7$ or $8$), the code sets `p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]`, which sets `p_idx` to $0$ or $1$! Then on lines 318-319, it looks up `relevance_p_pidx[p_idx]` and `groundness_p_pidx[p_idx]` for that `p_idx` ($0$ or $1$), so the inserted passage and its reflection tokens still match each other, even though `p_idx` was drawn from `0 .. len(irrelevant_p_indices) - 1`. The practical effect is that the "irrelevant" sample is really a random pick among the top-ranked passages, so $\mathcal{D}_{\text{gen}}$ contains fewer `[Irrelevant]` examples than the cascade intends.

3. Dataset-specific overrides on `nq`, `fever`, and `wow` (lines 309-310 and 325-326):
   - On lines 309-310, whenever `dataset_name in ["nq", "fever"] and len(irrelevant_p_indices) > 0`, `p_idx` is overwritten with `random.sample(range(len(irrelevant_p_indices)), k=1)[0]`. Because this runs after the cascade, it replaces even a `[Fully supported]` choice and makes the `nq` rule on lines 292-293 redundant.
   - On lines 325-326, for `nq`, `fever`, and `wow`, if a selected passage was labeled `[Relevant]` + `[No support / Contradictory]`, its `relevance_token` is relabeled to `"[Irrelevant]"`.
   - None of these dataset-specific rules, nor the downsampling rules in this file, are described in the paper.

### 3.4 Stage 4: Assembling the Segment String and Final Filtering (Lines 328-368)

Once `p_idx`, `relevance_token`, and `groundness_token` are finalized for sentence `sent_idx`, lines 330-344 append the formatted segment to `output`:

```python
                    ctx = retrieved_results[sent_idx]["ctxs"][p_idx]
                    processed_paragraph = ctx["title"]+"\n" + ctx["text"]
                    if relevance_token == "[Relevant]":
                        output += "[Retrieval]" + "<paragraph>{}</paragraph>".format(processed_paragraph) + "[Relevant]" + sentence + groundness_token
                        groundness_stats.append(groundness_token)
                    else:
                        output += "[Retrieval]" + "<paragraph>{}</paragraph>".format(processed_paragraph) + "[Irrelevant]" + sentence
                    relevance_stats.append(relevance_token)
                elif need_retrieval_decision_token == "[Continue to Use Evidence]":
                    need_retrieval_multi_stats.append(need_retrieval_decision_token)
                    output += "[Continue to Use Evidence]" + sentence
                elif need_retrieval_decision_token == "[No Retrieval]":
                    need_retrieval_multi_stats.append("[No Retrieval]")
                    output += "[No Retrieval]" + sentence
```

Look at how these lines produce the exact four grammar patterns we studied in Module 04 Part 1:
- When `need_retrieval_decision_token == "[Retrieval]"` and `relevance_token == "[Relevant]"`:
  `[Retrieval]<paragraph>{title}\n{text}</paragraph>[Relevant]{sentence}{groundness_token}`
- When `need_retrieval_decision_token == "[Retrieval]"` and `relevance_token == "[Irrelevant]"`:
  `[Retrieval]<paragraph>{title}\n{text}</paragraph>[Irrelevant]{sentence}` (with `groundness_token` omitted).
- When `need_retrieval_decision_token == "[Continue to Use Evidence]"`:
  `[Continue to Use Evidence]{sentence}`
- When `need_retrieval_decision_token == "[No Retrieval]"`:
  `[No Retrieval]{sentence}`

Finally, after all sentences in `splitted_output_sentences` are processed:
- Line 348 appends `output += utility_i_token` (`[Utility:k]`) at the end of the response.
- Lines 351-352 drop half of the `nq` examples whose output contains any `[No Retrieval]` (`random.random() > 0.5`), another downsampling rule that the paper does not describe.
- Lines 356-360 convert any remaining `"REFUTES"` / `"SUPPORTS"` strings to `"false"` / `"true"`.
- Line 368 (`processed_data = [item for item in processed_data if item["output"].find("[Continue to Use Evidence]") != 0]`) removes any malformed sequence where `"[Continue to Use Evidence]"` appeared at index $0$ (the very start of the response before any passage was ever retrieved), and the script saves the final Generator training dataset $\mathcal{D}_{\text{gen}}$ to `.json` and `.jsonl`.

Because Branch 1 keeps only $40\%$ of retrieval-free examples, `nq` loses half of its examples that contain `[No Retrieval]`, and several `continue` statements drop examples or sentences, the final $\mathcal{D}_{\text{gen}}$ is smaller than the set of examples that entered this script.
