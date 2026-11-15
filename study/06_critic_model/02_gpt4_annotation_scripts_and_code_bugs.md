# GPT-4 Annotation Scripts and Code Bugs

## 1. Plain-English Intuition: How the Codebase Collects GPT-4 Supervision

In Part 1, we studied the five grading rubrics (Tables 8-12) used to prompt GPT-4 for reflection token labels.

In the repository, those prompts are executed by the scripts inside `reference/code/data_creation/critic/gpt4_reward/`:
1. `chatgpt_need_retrieval.py`: collects initial binary `Retrieve` labels (`[Yes]` / `[No]`) and multi-sentence `Retrieve` labels (`[Continue to Use Evidence]`, `[No Retrieval]`, `[Retrieval]`).
2. `chatgpt_relevance.py`: collects `IsRel` passage relevance labels (`[Relevant]`, `[Irrelevant]`).
3. `chatgpt_groundness.py`: collects `IsSup` factual support labels (`[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`).
4. `chatgpt_utility.py`: collects `IsUse` perceived utility ratings (`1` through `5`).
5. `chatgpt_inference.py`: a standalone baseline script that prompts `gpt-3.5-turbo` with `example["instruction"]` and evaluates the response against `example["answers"]` using SacreBLEU (`BLEU().corpus_score`).

Why are the scripts named `chatgpt_*.py` if the Critic was distilled from GPT-4?
`reference/code/data_creation/critic/gpt4_reward/README.md` (line 3) says "We use GPT-4 by default", but the scripts' `--model_name` argument actually defaults to `gpt-3.5-turbo` (for example, `chatgpt_need_retrieval.py` line 153), so GPT-4 has to be requested explicitly on the command line (for example, `--model_name gpt-4`). The paper states that the reflection-token labels come from GPT-4.

Walking through these scripts line by line shows how the inputs are formatted, how interrupted API runs are resumed, and how four bugs in these scripts are handled downstream in Part 3 (`combine_chat_gpt_reward.py`).

---

## 2. Shared Execution Architecture Across the Annotation Scripts

### 2.1 Input Entry Schema (`README.md`)

Each annotation script reads one or more `.json` or `.jsonl` files (`--input_files`), where each dictionary entry represents a full example or a single sentence-passage pair:

```python
{
    "instruction": str,          # task instruction or question x
    "target_output": str,        # current sentence segment y_t
    "evidence": str,             # retrieved Wikipedia passage d (title + "\n" + text)
    "preceding_sentences": str,  # previously generated sentences y_{<t}
    "output": str,               # complete response y (used for utility)
    "q_id": str,                 # unique instance ID (mapped to "id")
    "sent_idx": int,             # 0-indexed sentence position t - 1
    "p_idx": int                 # 0-indexed retrieved passage rank
}
```

### 2.2 Checkpointing and Resuming (`--previous_results`)

Because querying the OpenAI API on $10,000$ to $20,000$ examples per aspect takes hours and can be interrupted by network timeouts, `chatgpt_need_retrieval.py` (lines 163-181) and `chatgpt_groundness.py` (lines 142-158) save intermediate progress every $20$ or $100$ steps to `args.output_file_name + "_tmp"`, and support resuming via `--previous_results`:

```python
    for item in examples:
        if "id" not in item and "q_id" in item:
            item["id"] = item["q_id"]

    previous_result = []
    for prev_file in args.previous_results:
        if prev_file.endswith(".json") or prev_file.endswith(".json_tmp"):
            previous_result += json.load(open(prev_file))
        else:
            previous_result += load_jsonlines(prev_file)
...
    previous_ids = [item["input"]["id"] for item in previous_result]
    examples = [item for item in examples if item["id"] not in previous_ids]
```

By extracting `item["input"]["id"]` from all previously saved dictionaries and filtering out any example whose `"id"` is in `previous_ids`, a restarted job only queries OpenAI on the remaining unfinished examples.

### 2.3 Task Prefix Injection and Refusal Filtering (`chatgpt_relevance.py` and `chatgpt_groundness.py`)

In `chatgpt_relevance.py` (lines 130-143) and `chatgpt_groundness.py` (lines 167-180), before formatting the prompt for GPT-4, the loop normalizes raw QA examples and filters out AI refusal responses:

```python
    KNOWLEDGE_INSTRUCTIONS = {
        "nq": "Please answer the following questions using the shortest possible response. For example, if the question asks 'What is the capital of France?'', you can simply reply with 'Paris'.",
        "fever": "Determine whether the following statement is true or false.",
        "wow": "You have been provided with a chat history between two agents, separated by new lines. Generate a response that is informative and engaging based on the latest message."
    }
...
    for idx, example in tqdm(enumerate(examples)):
        if "output" not in example and "answers" in example:
            example["output"] = example["answers"][0] if type(
                example["answers"]) is list else example["answers"]
        if "target_output" not in example and "output" in example:
            example["target_output"] = example["output"]
        if "instruction" not in example and "question" in example:
            data_type = example["q_id"].split("_")[0]
            if data_type in KNOWLEDGE_INSTRUCTIONS:
                example["instruction"] = KNOWLEDGE_INSTRUCTIONS[data_type] + \
                    example["question"]
            else:
                example["instruction"] = example["question"]
        if "As a language model, I cannot" in example["output"]:
            continue
```

Notice line 142 (`if "As a language model, I cannot" in example["output"]: continue`): instruction-following datasets like ShareGPT and Alpaca contain refusal responses where the assistant says `"As a language model, I cannot..."`. Skipping those refusals prevents wasting GPT-4 API calls on uninformative boilerplate.

---

## 3. Prompt Variants in `chatgpt_need_retrieval.py` and `chatgpt_groundness.py`

### 3.1 The Five Prompt Templates in `chatgpt_need_retrieval.py`

Inspecting `PROMPT_DICT` in `reference/code/data_creation/critic/gpt4_reward/chatgpt_need_retrieval.py` (lines 18-100) reveals five prompt templates corresponding to three different modes controlled by `--multi_retrieval` and `--three_way`:

| Template Key in `PROMPT_DICT` | When Intended to Be Used | Output Labels Requested | Fields Formatted into Prompt |
|---|---|---|---|
| `"context"` | `--multi_retrieval` is `False` (initial retrieval at $t = 1$) | `[Yes]`, `[No]` | `{instruction}` |
| `"multi_retrieval"` | `--multi_retrieval` is `True`, `--three_way` is `False`, with preceding sentences | 2-way: `[No Retrieval]`, `[Retrieval]` | `{instruction}`, `{preceding_sentences}`, `{evidence}`, `{target_output}` |
| `"multi_retrieval_no_preceding"` | `--multi_retrieval` is `True`, `--three_way` is `False`, without preceding sentences | 2-way: `[No Retrieval]`, `[Retrieval]` | `{instruction}`, `{evidence}`, `{target_output}` |
| `"multi_retrieval_three_way"` | `--multi_retrieval` is `True`, `--three_way` is `True`, with preceding sentences | 3-way: `[Continue to Use Evidence]`, `[No Retrieval]`, `[Retrieval]` | `{instruction}`, `{preceding_sentences}`, `{evidence}`, `{target_output}` |
| `"multi_retrieval_three_way_no_preceding"` | `--multi_retrieval` is `True`, `--three_way` is `True`, without preceding sentences | 3-way: `[Continue to Use Evidence]`, `[No Retrieval]`, `[Retrieval]` | `{instruction}`, `{evidence}`, `{target_output}` |

Also note lines 183-185 of `chatgpt_need_retrieval.py`:

```python
    if args.multi_retrieval is True:
        examples = [item for item in examples if "sent_idx" in item]
        examples = [item for item in examples if item["sent_idx"] > 0]
```

When `--multi_retrieval` is passed, the script keeps only sentences with `sent_idx > 0` (the second sentence onward, $t > 1$), because the first sentence (`sent_idx == 0`) is governed by the initial retrieval decision (`"context"`).

### 3.2 Five-Scale vs Three-Way Groundedness in `chatgpt_groundness.py`

Similarly, `PROMPT_DICT` in `chatgpt_groundness.py` (lines 22-82) contains both an earlier 5-point numerical entailment prompt (`"context"`, rating support from `1: Ignore / Contradictory` to `5: Fully supported` on whole outputs) and the final 3-way segment-level entailment prompts (`"multi"` and `"multi_no_preceding"`, outputting `[Fully supported]`, `[Partially supported]`, or `[No support / Contradictory]` on `{target_output}`).

---

## 4. Detailed Diagnosis of Four Code Bugs in `gpt4_reward/`

Inspecting every line of the `gpt4_reward/` scripts reveals four implementation bugs and shows why the downstream combiner script (`combine_chat_gpt_reward.py`, covered in Part 3) was written with specific recovery logic.

### 4.1 Bug 1: Swapped Three-Way Templates in `chatgpt_need_retrieval.py` (Lines 125-137)

Look at `process_input` in `reference/code/data_creation/critic/gpt4_reward/chatgpt_need_retrieval.py` (lines 125-137):

```python
def process_input(example, multi_retrieval=False, three_way=False):
    if multi_retrieval is False:
        return PROMPT_DICT["context"].format_map(example)
    else:
        if example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            if three_way is False:
                return PROMPT_DICT["multi_retrieval_no_preceding"].format_map(example)
            else:
                return PROMPT_DICT["multi_retrieval_three_way"].format_map(example)
        else:
            if three_way is False:
                return PROMPT_DICT["multi_retrieval"].format_map(example)
            else:
                return PROMPT_DICT["multi_retrieval_three_way_no_preceding"].format_map(example)
```

Compare the `three_way is False` branches against the `three_way is True` branches:
- When `example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0` (there are no preceding sentences):
  - `three_way is False` correctly selects `"multi_retrieval_no_preceding"`.
  - `three_way is True` mistakenly selects `"multi_retrieval_three_way"` (which formats `Preceding sentences: {preceding_sentences}\n`).
- When `example["sent_idx"] > 0` and `len(example["preceding_sentences"]) > 0` (there are preceding sentences):
  - `three_way is False` correctly selects `"multi_retrieval"`.
  - `three_way is True` mistakenly selects `"multi_retrieval_three_way_no_preceding"` (which omits `{preceding_sentences}`)!

Because lines 183-185 filter for `item["sent_idx"] > 0` when `--multi_retrieval` is enabled, almost all multi-sentence examples enter the `else:` branch (line 133) and therefore use `"multi_retrieval_three_way_no_preceding"` (omitting `preceding_sentences` from the prompt sent to GPT-4).

### 4.2 Bug 2: Digit-Only `postprocess` Parser Returning `""` on Bracketed Tokens (`chatgpt_groundness.py` and `chatgpt_relevance.py`)

In `chatgpt_utility.py` (lines 46-59), GPT-4 outputs a digit in `1..5` (such as `"Perceived utility: 3\nExplanation: ..."`), so `postprocess` searches for digits `str(i)` for `i in range(1, 6)` inside `score_string`.

However, that exact same digit-searching `postprocess` function was copied unchanged into `chatgpt_groundness.py` (lines 91-106) and `chatgpt_relevance.py` (lines 71-86):

```python
def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    print(raw_output)
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1]
        if explanation[0] == " ":
            explanation = explanation[1:]
        score_string = raw_output.split("\nExplanation:")[0]
        score = None
        for i in range(1, 6):
            if str(i) in score_string:
                score = int(i)
        if score is None:
            return "", explanation
        else:
            return score, explanation
    else:
        return "", ""
```

Trace what happens when GPT-4 follows the `multi` prompt in `chatgpt_groundness.py` or `chatgpt_relevance.py` and outputs:

`"[Fully supported]\nExplanation: The evidence mentions all applications."`

1. `score_string = raw_output.split("\nExplanation:")[0]` extracts `"[Fully supported]"`.
2. The loop `for i in range(1, 6): if str(i) in score_string:` checks whether `"1"`, `"2"`, `"3"`, `"4"`, or `"5"` is in `"[Fully supported]"`.
3. Because there are no digits in `"[Fully supported]"` (or in `"[Relevant]"` / `"[Irrelevant]"`), `score` stays `None`!
4. `if score is None: return "", explanation` returns an empty string `score = ""`!

How did the pipeline survive `score` being saved as `""` for every bracketed token in `chatgpt_groundness.py` and `chatgpt_relevance.py`?
Look at what `main()` appends to `result_list` on line 205 of `chatgpt_groundness.py` (and line 159 of `chatgpt_relevance.py`):

```python
result_list.append({"input": example, "score": score, "explanation": score,
                   "raw_output": results["choices"][0]["message"]["content"]})
```

Because `result_list` also saved the unparsed string `"raw_output"`, in Part 3 (`combine_chat_gpt_reward.py` lines 162-163 and 187-188) the authors added a recovery line:

```python
if item["score"] == "":
    item["score"] = raw_output.split("\n")[0]
```

which extracts `"[Fully supported]"` or `"[Relevant]"` directly from the first line of `"raw_output"`!

### 4.3 Bug 3: Saving `"explanation": score` Instead of `"explanation": explanation`

Look again at line 205 of `chatgpt_groundness.py`, line 159 of `chatgpt_relevance.py`, and line 104 of `chatgpt_utility.py`: all three scripts write `"explanation": score` instead of `"explanation": explanation` inside `result_list.append(...)` (only `chatgpt_need_retrieval.py` line 221 correctly writes `"explanation": explanation`). Because `combine_chat_gpt_reward.py` discards explanations when building the Critic training data, this typo does not affect Critic training.

### 4.4 Bug 4: Missing `"context"` Key in `chatgpt_relevance.py` (Line 93)

In `chatgpt_relevance.py` (lines 91-97):

```python
def process_input(example, multi_retrieval=False):
    if multi_retrieval is False:
        return PROMPT_DICT["context"].format_map(example)
    else:
        if "sent_idx" not in example or example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            return PROMPT_DICT["multi_no_preceding"].format_map(example)
        else:
            return PROMPT_DICT["multi"].format_map(example)
```

In `chatgpt_relevance.py`, `PROMPT_DICT` (lines 22-62) only defines `"multi"` and `"multi_no_preceding"`; there is no `"context"` key. Calling `chatgpt_relevance.py` without `--multi_retrieval` therefore raises `KeyError: 'context'`, meaning `chatgpt_relevance.py` must always be invoked with `--multi_retrieval`.
