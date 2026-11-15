# Candidate Pairing and vLLM Critic Inference

## 1. Plain-English Intuition: Scoring Millions of Sentence-Passage Pairs with the Critic $\mathcal{C}$

In Part 1 (`create_retrieval_data.py`), we split each multi-sentence response $y$ into individual sentences $y_1, \dots, y_T$ and ran Contriever to retrieve the top $K = 10$ Wikipedia passages for each sentence $y_t$.

Now consider the scale of the grading task facing the Critic $\mathcal{C}$:
- Suppose $100,000$ of our seed examples require retrieval, with an average of $3$ sentences per example ($300,000$ sentences).
- Each sentence has $10$ candidate passages retrieved from Wikipedia ($3,000,000$ sentence-passage pairs).
- For every one of those $3,000,000$ sentence-passage pairs, the Critic $\mathcal{C}$ must predict both passage relevance (`IsRel`: `[Relevant]` vs `[Irrelevant]`) and factual support (`IsSup`: `[Fully supported]`, `[Partially supported]`, or `[No support / Contradictory]`).

To execute millions of Critic evaluations cleanly and in parallel across multiple GPUs, the repository uses two scripts in `reference/code/data_creation/generator/`:
1. `create_prompt_data.py`: Unpacks the $10$ retrieved passages `item["ctxs"][:top_n]` for each sentence into $10$ individual evaluation dictionaries (indexed by passage rank `p_idx` $\in \{0, \dots, 9\}$), skips any sentence where the Critic already predicted `[No Retrieval]`, and splits the workload into `num_jobs` batch files (`prompt_data_batch_0.jsonl`, `prompt_data_batch_1.jsonl`, and so on).
2. `run_reward_vllm.py`: Loads the trained Llama-2 7B Critic $\mathcal{C}$ into vLLM and runs batched greedy decoding (`temperature=0.0, max_tokens=15`) to predict the single reflection token for each item.

---

## 2. Line-by-Line Walkthrough of `create_prompt_data.py`

Let us trace `reference/code/data_creation/generator/create_prompt_data.py` from top to bottom.

### 2.1 Loading Multi-Sentence Retrieval Predictions (`qid2need_retrieval`, Lines 48-58)

Before creating passage-evaluation prompts for `IsRel` and `IsSup`, the script loads the Critic's multi-sentence retrieval predictions (`MULTI_RETRIEVAL_TOKEN_OUTPUT` from Step 4b in Part 1):

```python
    qid2need_retrieval = {}
    if args.multi_need_retrieval_pred_files is not None:
        for multi_need_retrieval_pred_file in args.multi_need_retrieval_pred_files:
            need_retrieval_data = json.load(open(multi_need_retrieval_pred_file))
            for item in need_retrieval_data:
                if "q_id" in item and "id" not in item:
                    item["id"] = item["q_id"]
                if "id" in item and "q_id" not in item:
                    item["q_id"] = item["id"]
                qid2need_retrieval.setdefault(item["q_id"], {})
                qid2need_retrieval[item["q_id"]][item["sent_idx"]] = item["pred"]
```

This builds a two-level dictionary `qid2need_retrieval[q_id][sent_idx]` mapping each example ID `q_id` and sentence index `sent_idx` to the Critic's predicted retrieval token (`"[Retrieval]"`, `"[No Retrieval]"`, or `"[Continue to Use Evidence]"`).

### 2.2 Unpacking Top-$N$ Passages and Skipping `"No Retrieval"` Sentences (Lines 60-98)

Next, lines 60-98 divide `dpr_data` (`MULTI_RETRIEVAL_OUTPUT`, where each item is a sentence with a list of retrieved passages `"ctxs"`) into `args.num_jobs` batch files:

```python
    batch_size = len(dpr_data) // args.num_jobs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(args.num_jobs):
        processed_data = []
        dpr_data_i = dpr_data[batch_size*i:batch_size*(i+1)]
        for item in dpr_data_i:
            dataset_name = item["dataset_name"]
            if dataset_name in TASK_DATA:
                instruction = TASK_INST[dataset_name] + "## Input:\n\n" + item["instruction"]
            else:
                instruction = item["instruction"]
            preceding_sentences = item["preceding_sentences"]
            target_output = item["target_output"]
            q_id = item["q_id"]
            output = item["output"]
            sent_idx = item["sent_idx"]
            if sent_idx == 0:
                for p_idx, ctx in enumerate(item["ctxs"][:args.top_n]):
                    evidence = ctx["title"] + "\n" + ctx["text"]
                    processed_data.append({"id": q_id, "instruction": instruction, "output": output, "evidence": evidence, "p_idx": p_idx, 
                                        "target_output": target_output, "preceding_sentences": preceding_sentences, "sent_idx": sent_idx})

            else:
                if q_id in qid2need_retrieval and sent_idx in qid2need_retrieval[q_id] and "No Retrieval" in qid2need_retrieval[q_id][sent_idx]:
                    continue
                for p_idx, ctx in enumerate(item["ctxs"][:args.top_n]):
                    evidence = ctx["title"] + "\n" + ctx["text"]
                    processed_data.append({"id": q_id, "instruction": instruction, "output": output, "evidence": evidence, "p_idx": p_idx, 
                                        "target_output": target_output, "preceding_sentences": preceding_sentences, "sent_idx": sent_idx})

        save_file_jsonl(processed_data, os.path.join(
            args.output_dir, "prompt_data_batch_{}.jsonl".format(i)))
```

Look at three details in this loop:
1. Re-attaching `TASK_INST[dataset_name] + "## Input:\n\n"` (lines 69-72):
   In Part 1 (`create_retrieval_data.py` line 64), the generic task instruction was stripped off so it would not interfere with Contriever passage search. Now that Contriever has already retrieved `item["ctxs"]`, lines 69-70 put `TASK_INST[dataset_name] + "## Input:\n\n"` back onto `instruction` so the Critic $\mathcal{C}$ sees the full task instruction when judging `IsRel` and `IsSup`.
2. Skipping `sent_idx > 0` when `"No Retrieval"` was predicted (lines 85-86):
   For `sent_idx == 0` (the first sentence of an example whose initial retrieval decision was `[Retrieval]`), all `args.top_n` passages (`p_idx` $= 0 \dots \text{top\_n}-1$) are unpacked.
   For `sent_idx > 0`, if the Critic predicted `"No Retrieval"` for that sentence (`"No Retrieval" in qid2need_retrieval[q_id][sent_idx]`), line 86 executes `continue`, skipping all `top_n` passages for that sentence because no passage will be inserted at `sent_idx`.
3. Remainder slice in `dpr_data_i` (line 66):
   Notice that `dpr_data_i = dpr_data[batch_size*i:batch_size*(i+1)]` does not extend the upper bound to `len(dpr_data)` on the last job `i == args.num_jobs - 1`. If `len(dpr_data)` is not an exact multiple of `args.num_jobs`, up to `args.num_jobs - 1` items at the tail of `dpr_data` are omitted unless `len(dpr_data)` is divisible by `args.num_jobs`.

---

## 3. Line-by-Line Walkthrough of `run_reward_vllm.py`

`reference/code/data_creation/generator/run_reward_vllm.py` is the workhorse inference script that runs the trained Critic $\mathcal{C}$ in vLLM across all five reflection evaluation passes.

### 3.1 The Five CLI Invocations of `run_reward_vllm.py`

In `reference/code/data_creation/generator/README.md`, `run_reward_vllm.py` is called five times during Stage 3 with different `--task`, `--inst_mode`, and `--input_mode` arguments:

| Pass | `--task` | `--inst_mode` (Rubric Key in `PROMPT_DICT`) | `--input_mode` (Instance Key in `PROMPT_DICT`) | Reflection Token Predicted |
|---|---|---|---|---|
| 1. Initial Retrieve ($t=1$) | `'retrieval'` | `retrieval_instruction` | `retrieval_input` | `[Retrieval]`, `[No Retrieval]` |
| 2. Multi-Sentence Retrieve ($t>1$) | `'multi_retrieval'` | `retrieval_multi_instruction` (or `multi_retrieval_three_way_instruction`) | `retrieval_multi_input` (or `multi_retrieval_three_way_input`) | `[Continue to Use Evidence]`, `[No Retrieval]`, `[Retrieval]` |
| 3. Perceived Utility | `'utility'` | `utility_instruction` | `utility_input` | `[Utility:1]` .. `[Utility:5]` |
| 4. Passage Relevance | `'relevance'` | `relevance_instruction` | `relevance_input` | `[Relevant]`, `[Irrelevant]` |
| 5. Factual Support | `'groudness'` | `ground_multi_instruction` | `ground_multi_input` | `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]` |

For Pass 2, the README command uses the two-way pair `retrieval_multi_instruction` / `retrieval_multi_input`, whose instruction offers only `[No Retrieval]` and `[Retrieval]` (and maps "verifiable from the evidence" to `[No Retrieval]`). The Critic, however, was trained on multi-sentence retrieval only with the three-way prompt (`combine_chat_gpt_reward.py` lines 124-126), so the documented command does not match the Critic's training format; the three-way pair is needed to obtain `[Continue to Use Evidence]`.

### 3.2 Two Execution Modes: `split == "train"` vs `split == "test"` (`process_data`, Lines 171-190)

Look at `process_data` in `run_reward_vllm.py` (lines 171-190):

```python
def process_data(input_data, inst_mode, input_mode, split="train", multi_retrieval=False):
    if split == "train":
        prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(input_data)
        output = str(input_data["output"])
        return prompt, output
    else:
        instruction = PROMPT_DICT[inst_mode]
        if multi_retrieval is True and (input_data["sent_idx"] == 0 or "preceding_sentences" not in input_data or type(input_data["preceding_sentences"]) is not str or len(input_data["preceding_sentences"]) == 0):
            input = PROMPT_DICT[input_mode +
                                "_no_preceding"].format_map(input_data)
        else:
            input = PROMPT_DICT[input_mode].format_map(input_data)
        prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(
            {"instruction": instruction, "input": input})
        output = "None"
        return prompt, output
```

Why does `process_data` have a `split == "train"` branch and an `else:` (`split == "test"`) branch?
1. Evaluating Critic accuracy on held-out GPT-4 data (`--split train`):
   When the authors wanted to measure how well the trained Critic $\mathcal{C}$ agrees with GPT-4 on the validation file `_dev.json` (producing the accuracy numbers in Appendix Figure 5 from Module 06 Part 3), each item in `_dev.json` was already formatted by `combine_chat_gpt_reward.py` with `"instruction"`, `"input"`, `"output"`, and `"task"`.
   In `main()` (line 239), `if args.split == "train": input_data = [item for item in input_data if item["task"] == args.task]` selects the validation examples for that specific task, formats `ALPACA_PROMPT_DICT["prompt_input"]`, compares the Critic's prediction `item["pred"]` against the GPT-4 ground truth `item["output"]` (lines 269-276), and prints the mean accuracy `np.mean([item["correct"] ...])` on line 302. This scoring is lenient: a prediction counts as correct when either string contains the other, and the printed mean skips empty predictions. Whether the paper's Figure 5 numbers came from this path is not stated.
2. Annotating unlabeled Generator training data (`--split test`):
   When annotating the $145,619$ seed examples in Stage 3, there is no GPT-4 ground-truth label (`output = "None"`). The `else:` branch looks up `instruction = PROMPT_DICT[inst_mode]` and `input = PROMPT_DICT[input_mode].format_map(input_data)`, wraps them into `ALPACA_PROMPT_DICT["prompt_input"]`, and saves the Critic's predicted token in `item["pred"]`.

### 3.3 Greedy vLLM Generation (`call_model` and `posprocess_output`, Lines 123-136)

Lines 123-136 run deterministic greedy decoding in vLLM and strip any trailing boundary tokens:

```python
def posprocess_output(answer, return_score=False):
    answer = answer.replace("</s>", "")
    answer = answer.replace("<unk>", "")
    answer = answer.replace("[PAD]", "")
    return answer


def call_model(prompts, model, max_new_tokens=50):
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    preds = model.generate(prompts, sampling_params)
    preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [posprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds
```

The downstream steps need `pred.outputs[0].text` to contain the predicted special token string (such as `"[Relevant]"` or `"[Fully supported]"`), after which `posprocess_output` removes `"</s>"`, `"<unk>"`, and `"[PAD]"`. Note, however, that `call_model` does not pass `skip_special_tokens=False`. In vLLM $0.2.6$ (the version pinned in `requirements.txt`), `SamplingParams` defaults to `skip_special_tokens=True`, which removes registered special tokens (including the $15$ reflection tokens) from the decoded text; the main README's Quick start passes `skip_special_tokens=False` for exactly this reason. Reproducing this step with the pinned version therefore requires adding that flag; the authors' original runs presumably used an environment where the tokens survived decoding.

---

## 4. Diagnosing Three Implementation Bugs in `run_reward_vllm.py`

Inspecting `run_reward_vllm.py` line by line reveals three implementation bugs:

### 4.1 Bug 1: Duplicate Function Definition of `process_data` (Lines 153-169 and 171-190)

In `run_reward_vllm.py`, `def process_data(input_data, inst_mode, input_mode, split="train", multi_retrieval=False):` is defined on lines 153-169 and then immediately defined a second time on lines 171-190 with identical active code. Python simply overwrites the first definition with the second.

### 4.2 Bug 2: Key Suffix Mismatch `"_no_preceding"` vs `"_wo_preceding"` (Line 183)

Look at line 183 inside `process_data`:

```python
        if multi_retrieval is True and (input_data["sent_idx"] == 0 or "preceding_sentences" not in input_data or type(input_data["preceding_sentences"]) is not str or len(input_data["preceding_sentences"]) == 0):
            input = PROMPT_DICT[input_mode +
                                "_no_preceding"].format_map(input_data)
```

Now check the keys actually defined inside `PROMPT_DICT` on lines 52 and 84 of `run_reward_vllm.py`:
- Line 52: `"ground_multi_input_wo_preceding"`
- Line 84: `"multi_retrieval_three_way_input_wo_preceding"`

Both dictionary keys end in `"_wo_preceding"` ("without preceding"), whereas line 183 looks up `input_mode + "_no_preceding"`!
Why did line 183 not crash with a `KeyError` during Stage 3?
Look at how `process_data` is called inside `main()` on lines 255-256 and 283-284:

```python
        processed_batch = [process_data(
            item, inst_mode=args.inst_mode, input_mode=args.input_mode, split=args.split)[0] for item in batch]
```

Because `main()` never passes `multi_retrieval=True` to `process_data`, the parameter `multi_retrieval` always stays at its default value `False`! Consequently, the `if multi_retrieval is True ...` branch on line 181 is never entered, and `process_data` always executes line 185 (`input = PROMPT_DICT[input_mode].format_map(input_data)`).

This has a side effect on first sentences. For `sent_idx == 0`, `preceding_sentences` is `None`, so the `IsSup` pass formats `ground_multi_input` (which contains a `Preceding sentences: {preceding_sentences}` line) and the Critic reads the literal text `"Preceding sentences: None"`. During Critic training, first-sentence `IsSup` examples used `ground_multi_input_wo_preceding`, which omits that line, so the annotation prompt differs from the training prompt.

### 4.3 Bug 3: Dropped Remainder Batch in `predicted_results` (Lines 281-299)

Compare the main batch loop (lines 251-279) with the remainder batch loop (`if len(input_data) % args.batch_size > 0:`, lines 281-299):

```python
    # Main batch loop (lines 251-279):
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        ...
        for j, item in enumerate(batch):
            ...
            pred = preds[j]
            item["pred"] = pred
            ...
            predicted_results.append(copy.deepcopy(item))
        with open(args.result_fp + "_tmp", "w") as outfile:
            json.dump(predicted_results, outfile)

    # Remainder batch loop (lines 281-299):
    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        ...
        for j, item in enumerate(batch):
            pred = preds[j]
            item["pred"] = pred
            if args.split == "train":
                ...
```

In the remainder batch loop (lines 281-299), the line `predicted_results.append(copy.deepcopy(item))` is missing!
- When `args.split == "train"`, mutating `item["correct"]` in-place inside `batch` still updates `input_data`, so the accuracy printed on line 302 (`np.mean([item["correct"] for item in input_data ...])`) includes the remainder batch.
- However, when `json.dump(predicted_results, outfile)` saves the output file on line 306, the last `len(input_data) % args.batch_size` items (up to $9$ items for the default `batch_size = 10`) are not in `predicted_results` and are silently dropped from the saved JSON file. Downstream, `postprocess_data.py` treats the dropped sentence-passage pairs as missing, which can remove candidate passages or whole sentences from $\mathcal{D}_{\text{gen}}$.
