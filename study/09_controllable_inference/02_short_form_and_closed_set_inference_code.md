# Short-Form and Closed-Set Inference Code

## 1. Plain-English Intuition: Open-Ended Best-Passage Selection vs Closed-Set Vote Aggregation

In Part 1, we derived how Self-RAG scores each candidate passage $d_k \in \{d_1, \dots, d_K\}$ by combining its length-normalized sequence probability $\exp(\text{seq\_score})$ with the weighted critique scores $s_t(\text{IsRel})$, $s_t(\text{IsSup})$, and $s_t(\text{IsUse})$ into a single number `final_score`.

In `reference/code/retrieval_lm/run_short_form.py`, this single-step candidate scoring loop evaluates four benchmarks from Table 2 of the paper, which fall into two distinct task types:

1. Open-Ended Short-Form QA (`closed = False`, used for PopQA and TriviaQA):
   Here, the question asks for an open-ended factual entity or phrase (such as `"Who is the author of The Lie?"` or `"When did Season 7 of The Walking Dead premiere?"`). Each of the $K$ retrieved passages may produce a different phrasing or entity. Therefore, when `closed = False`, the decoder simply selects the candidate response from the single passage $k^*$ with the highest `final_score`:

   $$k^* = \arg\max_{k \in \{1, \dots, K\}} \text{score}(k)$$

2. Closed-Set Classification (`closed = True`, used for PubHealth fact verification and ARC-Challenge multiple-choice reasoning):
   In PubHealth (`--task fever`), the only valid answers are `"true"` or `"false"`. In ARC-Challenge (`--task arc_c`), the only valid answers are `"A"`, `"B"`, `"C"`, or `"D"`.
   Suppose we retrieve $K = 5$ passages for an ARC-Challenge question:
   - Passage $1$ is an outlier that predicts `"A"` with score $2.20$.
   - Passages $2, 3, 4,$ and $5$ all predict `"B"` with scores $2.00, 1.90, 1.85,$ and $1.75$.

   If we only looked at the single highest-scoring passage, Passage $1$ (`"A"`, score $2.20$) would win, ignoring the fact that four other well-supported passages agreed on `"B"`!
   Therefore, when `closed = True`, `run_short_form.py` strips all reflection tokens from each passage's prediction to extract the bare answer option $a \in \mathcal{A}$ (`"A"`, `"B"`, `"C"`, `"D"` or `"true"`, `"false"`), and sums `final_score` across all passages that predicted the same answer option $a$:

   $$\text{TotalScore}(a) = \sum_{k \,:\, \text{ans}(y^{(k)}) = a} \text{score}(k), \quad \hat{a} = \arg\max_{a \in \mathcal{A}} \text{TotalScore}(a)$$

   In our example, `"B"` receives $2.00 + 1.90 + 1.85 + 1.75 = 7.50 > 2.20$, so `"B"` wins by combining evidence across all supporting passages.

---

## 2. Step-by-Step Walkthrough of `run_short_form.py`

Let us trace `reference/code/retrieval_lm/run_short_form.py` from input formatting to final prediction.

### 2.1 Stage 1: Task-Specific Input Formatting (`preprocess_input_data`, Lines 210-250)

Before running inference, `preprocess_input_data(dataset, task=args.task)` (lines 210-250) formats each item's `"instruction"` field according to `args.task`:

```python
def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            prompt = instruction + "\n\n## Input:\n\n" + \
                item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data
```

Comparing the three formatting cases:
- When `--task arc_c` (ARC-Challenge): normalizes numeric labels `"1".."4"` to `"A".."D"`, formats the four choices as `"\nA: ...\nB: ...\nC: ...\nD: ..."`, and joins `TASK_INST["arc_c"] + "\n\n### Input:\n" + item["question"] + choices`.
- When `--task fever` (PubHealth): `instruction` is `TASK_INST["fever"]`, and line 244 joins `instruction + "\n\n## Input:\n\n" + item["question"]`.
- When `--task` is not passed (PopQA and TriviaQA): `instruction` is `None`, so `item["instruction"]` is simply the raw question `item["question"]`.
- Both closed-set formats differ from the training data, where the preprocessing scripts of Module 04 Part 3 joined the instruction and question as `TASK_INST[...] + "## Input:\n\n" + question` (no leading blank line; ARC questions used `"\nA: ..."` options after the stem). The short-form `TASK_INST` also omits the formatting sentences shown in the paper's Table 5.

### 2.2 Stage 2: Initial Retrieval Decision and the Log-Probability Bug (Lines 51-83)

Inside `call_model_rerank_w_scores_batch` (lines 51-83), the script first decides whether to trigger retrieval (`do_retrieve`):

```python
    results = {}
    if mode != "always_retrieve":
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32016)
        preds = model.generate([prompt], sampling_params)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_text = preds[0].outputs[0].text
        pred_log_probs = preds[0].outputs[0].logprobs
        results["no_retrieval"] = pred_text

    # save relevance token scores
    if mode == "always_retrieve":
        do_retrieve = True

    elif mode == "no_retrieval":
        do_retrieve = False

    else:
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob)
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred
```

Inspecting lines 73-83 reveals a critical mathematical bug in `run_short_form.py` (`run_long_form_static.py` exponentiates correctly before dividing, on lines 122 and 186):

1. Missing `np.exp` on lines 78-79:
   In vLLM, `pred_log_probs[0][id]` is a log-probability $\log p_0(\text{id}) \in (-\infty, 0]$ (zero or negative), not a probability in $[0, 1]$!
   On line 109 of the very same file (`relevance_score_dict[p_idx][tok] = np.exp(float(prob))`) and in `run_long_form_static.py` line 186 (`ret_token_score_dict[tok] = np.exp(prob)`), the code properly applies `np.exp(...)` to convert the log-probability into a probability before dividing.
   On line 79 of `run_short_form.py`, however, `score_dict[tok] = float(prob)` stores the raw log-probability without `np.exp`!

   Let us see with concrete numbers how dividing two negative log-probabilities without `np.exp` inverts the threshold test:
   - Case 1 (Model is $90\%$ sure it needs retrieval):
     Suppose $p_0(\text{[Retrieval]}) = 0.90$ ($\log p = -0.1054$) and $p_0(\text{[No Retrieval]}) = 0.01$ ($\log p = -4.6052$).
     With `np.exp` (the intended formula):
     $$\frac{0.90}{0.90 + 0.01} = \frac{0.90}{0.91} \approx 0.989 > 0.2 \quad (\text{Retrieves!})$$
     Without `np.exp` (what lines 80-81 actually compute):
     $$\frac{-0.1054}{-0.1054 + (-4.6052)} = \frac{-0.1054}{-4.7106} \approx +0.0224 < 0.2 \quad (\text{Skips retrieval!})$$
   - Case 2 (Model is $90\%$ sure it does NOT need retrieval):
     Suppose $p_0(\text{[Retrieval]}) = 0.01$ ($\log p = -4.6052$) and $p_0(\text{[No Retrieval]}) = 0.90$ ($\log p = -0.1054$).
     Without `np.exp`, lines 80-81 compute:
     $$\frac{-4.6052}{-4.6052 + (-0.1054)} = \frac{-4.6052}{-4.7106} \approx +0.9776 > 0.2 \quad (\text{Retrieves!})$$
   Because higher probabilities have log-probabilities closer to zero, omitting `np.exp(float(prob))` on line 79 reverses the ratio! (Note that when `--mode always_retrieve` is used, lines 73-83 are bypassed because `do_retrieve = True` on line 67.)

   Which commands hit this bug? The README's ARC-Challenge and PubHealth commands pass `--threshold 0.2` but no `--mode`. The argparse default for `--mode` is the string `"default"` (argparse does not check defaults against `choices`), which is neither `"always_retrieve"` nor `"no_retrieval"`, so those commands run this inverted threshold branch. The README's QA command leaves the mode as a `MODE` placeholder. The repository therefore does not show which mode produced the paper's short-form numbers.

2. Two minor syntax bugs in lines 75-83:
   - On lines 76-78, `if id not in pred_log_probs[0]: score_dict[tok] = -100` lacks an `else:` before `prob = pred_log_probs[0][id]` (it does not crash only because `logprobs=32016` on line 58 returns all $32016$ vocabulary tokens, so `id` is always in `pred_log_probs[0]`).
   - On line 83 (`else: do_retrieve = "[Retrieval]" in pred`), the variable is named `pred_text` (defined on line 61), not `pred`. Because `pred` is assigned later in the same function, Python raises `UnboundLocalError` (a subclass of `NameError`) whenever this line runs, that is, in adaptive mode without `--threshold`.

### 2.3 Stage 3: Parallel Passage Generation and Critique Scoring (Lines 85-169)

When `do_retrieve is True`, lines 85-169 construct $K$ (`args.ndocs`) parallel prompts and compute `final_score` for each passage `p_idx` $\in \{0, \dots, K-1\}$ (note `logprobs=5000`: a reflection token outside the $5000$ most probable tokens at a position falls back to log-probability $-100$, that is, probability $\approx 0$):

```python
    if do_retrieve is True:
        evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
            para["title"], para["text"]) for para in evidences]
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params)
...
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)
            ...
            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))

            if ut_tokens is not None:
                utility_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))
```

Notice how lines 107-157 implement the exact equations we derived in Part 1:
- At index $0$ (`pred_log_probs[0]`), `relevance_score_dict[p_idx]` exponentiates the log-probabilities of `[Relevant]` and `[Irrelevant]`, and lines 134-135 compute `relevance_score`.
- At the first token index `idx` where an `IsSup` token appears in `pred_token_ids`, `grd_score_dict[p_idx]` exponentiates the log-probabilities of all three `IsSup` tokens, and lines 137-142 compute `ground_score` with weights $1.0$ and $0.5$.
- At the first token index `idx` where an `IsUse` token appears in `pred_token_ids`, `ut_score_dict[p_idx]` exponentiates the log-probabilities of all five `IsUse` tokens, and lines 144-150 compute `utility_score` with `ut_scores = [-1, -0.5, 0, 0.5, 1]`.
- The weights come from the command line. Note that `--w_use` defaults to `1.0` in this script's argparse (line 285), while the paper's default configuration uses $w_{\text{use}} = 0.5$; the README's short-form commands do not override it, so they run with $w_{\text{use}} = 1.0$.

### 2.4 Stage 4: Answer Aggregation and Three Script Bugs (Lines 177-200 and 312-367)

Lines 177-200 select the final prediction `best_option`:

```python
    # Aggregating answers
    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]
        return best_option, results, do_retrieve
```

- When `len(results) == 1` (meaning `do_retrieve` was `False`, so only `"no_retrieval"` is in `results`), it strips reflection tokens from `pred` and returns it. The test `len(results) == 1` is only a proxy: with `--mode always_retrieve` and `--ndocs 1` (the "Retrieve top 1" setting), `results` also has exactly one entry, `pred` is then the vLLM output object left over from the scoring loop rather than a string, and `postprocess_answer_option_conditioned(pred)` fails.
- When `do_retrieve` was `True` and `closed is True` (`args.task in ["fever", "arc_c"]`), it strips all reflection tokens, `"</s>"`, and `"\n"` via `postprocess_answer_option_conditioned(result["pred"])` to get the clean option string `answer` (such as `"true"` or `"B"`), sums `answer2score[answer] += score` across all $K$ passages, and returns the highest-scoring option `sorted_answers[0][0]`.
- When `do_retrieve` was `True` and `closed is False` (PopQA, TriviaQA), it returns `results[best_path]["pred"]` from the single highest-scoring passage.

Also note two more bugs inside `main()` of `run_short_form.py`:
1. Unexpected keyword argument `max_depth` on line 315:
   In `main()`, `generate` (lines 312-316) calls `call_model_rerank_w_scores_batch(..., max_depth=args.max_depth, ...)`, but `call_model_rerank_w_scores_batch` (defined on lines 51-54) does not have `max_depth` in its parameter list! Passing `max_depth` raises a `TypeError` on the first example unless `max_depth=args.max_depth` is removed from line 315 (or `**kwargs` is added to `call_model_rerank_w_scores_batch`), so every short-form command in the README needs this fix to run.
2. Retrieval frequency denominator on line 367:
   At the very end of `main()` (line 367), the script prints:
   ```python
   print("Retrieval Frequencies: {0}".format(count / len(final_results)))
   ```
   Because `final_results` is a dictionary with $8$ metadata keys (`"preds"`, `"prompts"`, `"metric_results"`, `"all_results"`, `"golds"`, `"metric"`, `"metric_mean"`, `"scores"`), `len(final_results)` is always $8$ rather than `len(input_data)` (or `len(preds)`).

---

## 3. Line-by-Line Walkthrough of `retrieval_lm/metrics.py`

`reference/code/retrieval_lm/metrics.py` implements the evaluation metrics used by `run_short_form.py` and `run_baseline_lm.py`:

### 3.1 Text Normalization (`normalize_answer`, Lines 51-64)

```python
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
```

Following the standard SQuAD evaluation protocol, `normalize_answer(s)` applies four transformations in sequence:
1. `lower(s)`: converts `"The Eiffel Tower!"` to `"the eiffel tower!"`.
2. `remove_punc(...)`: removes all ASCII punctuation characters, giving `"the eiffel tower"`.
3. `remove_articles(...)`: replaces standalone words `"a"`, `"an"`, or `"the"` (`\b(a|an|the)\b`) with spaces, giving `"  eiffel tower"`.
4. `white_space_fix(...)`: collapses multiple spaces and trims ends, giving `"eiffel tower"`.

### 3.2 Substring Inclusion `match`, `accuracy`, `exact_match_score`, and `qa_f1_score`

1. Substring Inclusion (`match`, lines 81-85):
   ```python
   def match(prediction, ground_truth):
       for gt in ground_truth:
           if gt in prediction:
               return 1
       return 0
   ```
   Used as the primary metric (`--metric match`) across PopQA, TriviaQA, PubHealth, and ARC-Challenge in `run_short_form.py`. Notice that in `metrics.py`, `match` checks raw case-sensitive substring inclusion `gt in prediction` (whereas in `run_short_form.py` lines 345-348, `"SUPPORTS"` and `"REFUTES"` are first mapped to `"true"` and `"false"` before calling `match`).

2. Exact Match (`exact_match_score`, lines 8-9) and `accuracy` (lines 18-25):
   `exact_match_score` checks whether `normalize_answer(prediction) == normalize_answer(ground_truth)`. `accuracy(preds, labels)` computes `100 * (match_count / len(preds))` over a list of predictions. (Note that in `run_short_form.py` line 342, `accuracy(pred, row["output"])` is called on a single example rather than a batch list; the README's short-form commands all pass `--metric match` instead of `--metric accuracy`.)

3. Token-Level Precision, Recall, and $F_1$ (`qa_f1_score`, lines 39-49):
   ```python
   def qa_f1_score(prediction, ground_truth):
       prediction_tokens = normalize_answer(prediction).split()
       ground_truth_tokens = normalize_answer(ground_truth).split()
       common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
       num_same = sum(common.values())
       if num_same == 0:
           return 0
       precision = 1.0 * num_same / len(prediction_tokens)
       recall = 1.0 * num_same / len(ground_truth_tokens)
       f1 = (2 * precision * recall) / (precision + recall)
       return f1
   ```
   Here `Counter(prediction_tokens) & Counter(ground_truth_tokens)` computes the multiset intersection (counting each shared word up to the minimum number of times it appears in both strings), and computes the harmonic mean $F_1 = \frac{2 PR}{P + R}$.
