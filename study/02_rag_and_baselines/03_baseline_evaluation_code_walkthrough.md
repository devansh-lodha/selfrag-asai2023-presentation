# Baseline Evaluation Code Walkthrough

## 1. Plain-English Intuition: How the Codebase Runs Standalone and Conventional RAG Baselines

In Part 1 and Part 2, we saw the two categories of baseline models evaluated in Table 2 of the Self-RAG paper:
1. Standalone models (`mode="vanilla"`): The model receives only the task instruction and question $x$, relying entirely on its parametric weights $\theta$.
2. Conventional RAG models (`mode="retrieval"`): The script takes the top $K$ (`--top_n`) pre-retrieved Wikipedia passages $d_{(1)}, \dots, d_{(K)}$, formats them as a numbered list `[1] ... [2] ...`, pastes that list right above the question $x$, and asks the model to generate an answer $y$.

Both baseline modes are implemented in a single script, `reference/code/retrieval_lm/run_baseline_lm.py`, using prompt templates and helper functions from `reference/code/retrieval_lm/utils.py`. Walking through these two files line by line shows exactly how conventional RAG prompts are constructed in practice and reveals two subtle implementation bugs in the baseline script.

---

## 2. Prompt Templates Across Model Families (`retrieval_lm/utils.py`)

### 2.1 Why Different Baseline Models Need Different Prompt Wrappers

When a language model is instruction-tuned (such as Alpaca, Open-Instruct, or Llama-2-chat), it is trained to recognize specific header strings (delimiters) that separate the user's instruction from the model's response. If you test an instruction-tuned model at inference time without the exact header format it saw during training, its accuracy drops sharply.

To support all baseline model families fairly, `reference/code/retrieval_lm/utils.py` (lines 6-33) defines `PROMPT_DICT` with paired non-retrieval and retrieval-augmented templates:

```python
PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
}
```

Let us compare the three template pairs in `PROMPT_DICT`:

| Model Family | Vanilla Template (`--mode vanilla`) | Conventional RAG Template (`--mode retrieval`) | Where Retrieved Passages `{paragraph}` Are Placed |
|---|---|---|---|
| Alpaca, Llama-2, Llama2-FT | `prompt_no_input` | `prompt_no_input_retrieval` | Under a `### Paragraph:\n` header before `### Instruction:\n` |
| Open-Instruct (Tulu) | `prompt_open_instruct` | `prompt_open_instruct_retrieval` | After `<user>\nReference:` and before `{instruction}` |
| Llama-2-chat | `llama_chat_prompt` | `llama_chat_prompt_retrieval` | Inside `[INST]...[/INST]` on the line before `{instruction}` |

Notice that in all three retrieval templates, `{paragraph}` is a single string containing all $K$ retrieved documents concatenated together, placed directly before `{instruction}`.

### 2.2 Task-Specific Zero-Shot Instructions (`TASK_INST`)

For open-domain question answering (PopQA and TriviaQA), the input query is already a self-contained question (such as `"Who is the author of The Lie?"`), so no extra task description is needed.

For closed-set and structured tasks (such as PubHealth fact verification or ARC-Challenge multiple-choice exams), the model needs an explicit instruction telling it what format to output (such as `"true"` or `"false"`, or `"A"`, `"B"`, `"C"`, `"D"`). Lines 35-42 of `reference/code/retrieval_lm/utils.py` define `TASK_INST`:

```python
TASK_INST = {
    "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
    "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
    "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
    "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
    "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
}
```

As in the PubHealth commands of `reference/code/README.md`, when evaluating on the PubHealth dataset, the command-line flag `--task fever` is passed so that the script looks up `TASK_INST["fever"]` (`"Is the following statement correct or not? Say true if it's correct; otherwise say false."`).

Note that Table 5 of the paper prints longer versions of two of these instructions: PubHealth adds `"Don't capitalize or add periods, just say "true" or "false"."`, and ARC-Challenge adds `"Please answer with the capitalized alphabet only, without adding any extra phrase or period."`. The released `TASK_INST` in `retrieval_lm/utils.py` (lines 34-41) omits both sentences, which fits the remark in Appendix B.2 that instructions were adjusted per model to avoid format violations.

---

## 3. Step-by-Step Walkthrough of `run_baseline_lm.py`

Let us trace `reference/code/retrieval_lm/run_baseline_lm.py` from top to bottom in five logical stages.

### 3.1 Stage 1: Model Initialization (vLLM vs OpenAI API, Lines 60-106)

After parsing command-line arguments (`--model_name`, `--input_file`, `--mode`, `--top_n`, `--prompt_name`, `--metric`, and so on), lines 93-105 initialize either an OpenAI API client or a local vLLM engine:

```python
    isOpenAI = True if args.model_name in ["text-davinci-003", "gpt-3.5-turbo-0301", "gpt-3.5-turbo"] else False
    if isOpenAI is False:
        if args.dtype is not None:
            model = LLM(model=args.model_name, download_dir=args.download_dir, dtype=args.dtype,
                        tensor_parallel_size=args.world_size,)
        else:
            model = LLM(model=args.model_name, download_dir=args.download_dir,
                        tensor_parallel_size=args.world_size,)

    input_data = load_file(args.input_file)
    if isOpenAI is True and args.api_key is not None:
        with open(args.api_key) as f:
            openai.api_key = f.read()[:-1]
        openai.organization = ORG_KEY
```

Here:
- If `--model_name` is a Hugging Face model path (such as `meta-llama/Llama-2-7b-hf`), `isOpenAI` is `False`, and `vllm.LLM` loads the weights onto `args.world_size` GPUs.
- If `--model_name` is an OpenAI model (`text-davinci-003` for InstructGPT or `gpt-3.5-turbo-0301` for ChatGPT), `isOpenAI` is `True`, and the script reads the secret key from the text file `args.api_key` (slicing `[:-1]` to remove the trailing newline character `\n` at the end of the file).
- `load_file(args.input_file)` (defined in `utils.py` lines 88-93) loads the evaluation dataset as a Python list of dictionaries `input_data` from either a `.json` or `.jsonl` file.

### 3.2 Stage 2: Formatting Retrieved Passages into `item["paragraph"]` (Lines 107-127)

When `--mode retrieval` is passed, the script builds the `{paragraph}` string for each evaluation example by taking the top `args.top_n` ($K$) passages from the `"ctxs"` list:

```python
    # For baseline scripts, we simply load pre-retrieved documents from `retrieval_file` option.
    if args.mode == "retrieval":
        if args.retrieval_file is not None:
            retrieval_data = load_file(args.retrieval_file)
            id2retrieval = {}
            for id, item in enumerate(retrieval_data):
                if "id" not in item:
                    id2retrieval[id] = item["ctxs"][:args.top_n]
                else:
                    id2retrieval[item["id"]] = item["ctxs"][:args.top_n]
            for id, item in enumerate(input_data):
                retrieval_result = id2retrieval[id if "id" not in item else item["id"]]
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        else:
            for id, item in enumerate(input_data):
                retrieval_result = item["ctxs"][:args.top_n]
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
```

Let us trace a concrete example of what this code produces when `args.top_n = 2` ($K = 2$). Suppose `item["ctxs"]` contains:

```python
[
    {"title": "US State Names", "text": "Eleven states are named after an individual person."},
    {"title": "Baby Names", "text": "In Texas, Emma is a popular baby name."},
    {"title": "California", "text": "California was named after a fictional island."}
]
```

1. `retrieval_result = item["ctxs"][:2]` slices the first $2$ dictionaries (indices $0$ and $1$).
2. The list comprehension formats index $i = 0$ with `"[1] "` and index $i = 1$ with `"[2] "`, placing a newline between `ctx["title"]` and `ctx["text"]`:
   - `evidences[0]` = `"[1] US State Names\nEleven states are named after an individual person."`
   - `evidences[1]` = `"[2] Baby Names\nIn Texas, Emma is a popular baby name."`
3. `item["paragraph"] = "\n".join(evidences)` joins them into a single multi-line string stored in `item["paragraph"]`. Later, when `PROMPT_DICT["prompt_no_input_retrieval"].format_map(item)` runs, Python substitutes `item["paragraph"]` directly into `{paragraph}`.

### 3.3 Stage 3: Normalizing Gold Answers and Task Instructions (Lines 129-148)

Different benchmark files store their questions and ground-truth answers under different dictionary keys. Lines 129-148 standardize every item in `input_data` so that it has an `"instruction"` field (for the prompt) and a `"golds"` field (a list of acceptable ground-truth answer strings for grading):

```python
    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
        if args.task == "fever" or args.task == "arc_c":
            item["instruction"] = TASK_INST[args.task] + \
                "\n\n### Input:\n" + item["instruction"]
```

Notice how each dataset maps to `item["golds"]`:
- PubHealth stores gold labels (`["true"]` or `["false"]`) under `"output"`.
- TriviaQA stores acceptable answer aliases under `"answers"`.
- PopQA stores acceptable answer aliases as a stringified Python list inside `"possible_answers"` (for example, `'["Sam Harris", "Samuel Benjamin Harris"]'`). Calling `ast.literal_eval(item["possible_answers"])` safely parses that string into an actual Python list of strings.
- ARC-Challenge stores the single correct letter (`"A"`, `"B"`, `"C"`, or `"D"`) as a string under `"answerKey"`, so `[item["answerKey"]]` wraps it in a 1-element list.

### 3.4 Stage 4: Batched Generation and Postprocessing (Lines 14-59 and 150-191)

Lines 150-191 iterate over `input_data` in batches of size `args.batch_size` (default $5$), format each dictionary using `PROMPT_DICT[args.prompt_name].format_map(item)`, and call one of three generation functions:
1. `call_model` (for local vLLM models, lines 47-53):
   Runs nucleus sampling (`temperature=0.8, top_p=0.95, max_tokens=max_new_tokens`), keeps only the text before the first double newline (`.split("\n\n")[0]`), and strips `"</s>"` and any leading space.
2. `call_model_instructgpt` (for `text-davinci-003`, lines 38-44):
   Calls `openai.Completion.create` with greedy decoding (`temperature=0.0, top_p=1`).
3. `call_model_chatgpt` (for `gpt-3.5-turbo`, lines 21-35):
   Calls `openai.ChatCompletion.create` with exponential backoff (`@backoff.on_exception`) when rate-limited.

### 3.5 Stage 5: Diagnosing Two Bugs in `run_baseline_lm.py` and Grading Metrics

Inspecting the error handling and postprocessing lines in `run_baseline_lm.py` reveals two bugs in the baseline script:

1. Bug 1: Variable name typo `results` vs `result` in `call_model_instructgpt` (lines 38-45):
   Compare `call_model_chatgpt` (lines 21-35) with `call_model_instructgpt` (lines 38-45):

   ```python
   def call_model_instructgpt(prompt, model, max_tokens=50):
       try:
           results = completions_instructgpt_backoff(model=model, prompt=prompt, temperature=0.0,
                                                     max_tokens=max_tokens, logprobs=5, top_p=1, frequency_penalty=0.0, presence_penalty=0.0)
           result = results["choices"][0]["text"]
       except (APIError, Timeout, APIConnectionError):
           results = "ERROR: API error outputs"
       return result
   ```

   If the OpenAI API raises a `Timeout` or `APIError`, the `except` block sets `results = "ERROR: API error outputs"` (with an `s`), leaving `result` (without an `s`) undefined! The next line `return result` then crashes with `UnboundLocalError: local variable 'result' referenced before assignment`.

2. Bug 2: Overwriting `postprocess_answers_closed` in the main batch loop (lines 165-170 vs 185-190):
   For closed-set tasks (`fever` and `arc_c`), `postprocess_answers_closed` in `utils.py` (lines 177-190) cleans up verbose model outputs (for example, extracting `"A"` if the model outputs `"The answer is A"`, or lowercasing `"True"` to `"true"`).
   Now look at how `run_baseline_lm.py` stores `item["output"]` in the main batch loop (lines 165-170) versus the remainder batch loop (lines 185-190):

   ```python
       # Main batch loop (lines 165-170):
       for j, item in enumerate(batch):
           pred = preds[j]
           item["output"] = postprocess_answers_closed(
               pred, args.task, args.choices)
           item["output"] = pred
           final_results.append(item)

       # Remainder batch loop (lines 185-190):
       for j, item in enumerate(batch):
           pred = preds[j]
           item["output"] = postprocess_answers_closed(
               pred, args.task, args.choices)
           final_results.append(item)
   ```

   In the main batch loop, line 167 computes the cleaned answer `postprocess_answers_closed(...)`, and line 169 (`item["output"] = pred`) immediately overwrites it with the raw, uncleaned prediction `pred`! Only the last `len(input_data) % args.batch_size` items in the remainder loop keep the cleaned output.

Finally, lines 193-205 score each item's `item["output"]` against `item["golds"]`:

```python
    for item in input_data:
        if args.metric == "em":
            metric_result = metric_max_over_ground_truths(
                exact_match_score, item["output"], item["golds"])
        elif args.metric == "accuracy":
            metric_result = 1.0 if item["golds"][0] in item["output"] else 0.0
        elif args.metric == "match":
            metric_result = match(item["output"], item["golds"])
        else:
            raise NotImplementedError
        item["metric_result"] = metric_result
```

Notice that both `--metric accuracy` (`item["golds"][0] in item["output"]`) and `--metric match` (`any(gt in item["output"] for gt in item["golds"])`) check whether the gold answer string is a substring of `item["output"]`, rather than requiring exact string equality `==`. That substring check partially masks Bug 2 whenever the model outputs a sentence like `"The statement is true."` (since `"true"` is still a substring), but fails if the model outputs capitalized `"True"` on a full batch where `postprocess_answers_closed`'s lowercasing was overwritten by `item["output"] = pred`.
