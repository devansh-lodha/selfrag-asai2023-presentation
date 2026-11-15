# Two-Stage Retrieval and Sentence Splitting

## 1. Plain-English Intuition: How We Annotate 145,619 Seed Examples Offline

By the end of Module 06, we have all three ingredients required to build the Generator training dataset $\mathcal{D}_{\text{gen}}$:
1. The $145,619$ plain `(instruction, output)` seed examples from Module 04.
2. The Contriever dense retriever $\mathcal{R}$ and FAISS index over Wikipedia from Module 05.
3. The trained Llama-2 7B Critic $\mathcal{C}$ from Module 06.

Now, in Stage 3 of the Self-RAG pipeline (`reference/code/data_creation/generator/`), we run $\mathcal{R}$ and $\mathcal{C}$ offline across the $145,619$ seed examples to split multi-sentence outputs into segments $y = (y_1, \dots, y_T)$, retrieve candidate Wikipedia passages for each segment, score them with the Critic, and insert `<paragraph>...</paragraph>` blocks and reflection tokens.

Before reading the code in `create_retrieval_data.py`, let us answer two key design questions:

1. Why does offline data creation include the ground-truth answer $y$ (or sentence $y_t$) inside the search query sent to Contriever (`question = instruction + " " + output`)?
   - At test time, the model has not generated the answer yet, so it searches Wikipedia using only the prompt $x$ (or $x$ plus the preceding sentence).
   - During offline training data creation, however, the ground-truth sentence $y_t$ is already known! Suppose the instruction $x$ is `"How did US states get their names?"` and the second sentence $y_2$ of the ground-truth answer says `"For instance, Louisiana was named in honor of King Louis XIV of France..."`.
   - If we searched Wikipedia using only `"How did US states get their names?"`, the top retrieved passages might discuss Washington or California and never mention Louisiana!
   - By concatenating the instruction and the target sentence (`instruction + " " + y_t`), a technique called hindsight retrieval, we guide Contriever to fetch the specific Wikipedia passage about Louisiana that actually supports the known target sentence $y_t$.

2. Why does the pipeline run Contriever in two separate passes (Initial Retrieval on the full output, followed by Continuous Retrieval on each sentence)?
   - Recall from Module 06 Part 1 (Table 9) that when a response has multiple sentences ($t > 1$), the Critic $\mathcal{C}$ must decide among three retrieval tokens: `[Continue to Use Evidence]`, `[No Retrieval]`, or `[Retrieval]`.
   - To judge whether the model can `[Continue to Use Evidence]` at sentence $t > 1$, the Critic needs to see the initial passage (`evidence`) that was retrieved at the start of the response!
   - Therefore, Pass 1 (Initial Retrieval) retrieves passages for the whole response so its top-1 passage can serve as `evidence` when the Critic evaluates multi-sentence retrieval necessity. Then Pass 2 (Continuous Retrieval) splits the output into individual sentences $y_1, \dots, y_T$ and retrieves top-$10$ candidate passages specifically for each sentence $y_t$.

---

## 2. Overview of the Complete 8-Step Generator Data Creation Pipeline (`README.md`)

`reference/code/data_creation/generator/README.md` documents the exact sequence of scripts that implement Algorithm 3 of the paper:

| Pipeline Step | Script Executed | Task / Mode | Input Files | Output File Produced |
|---|---|---|---|---|
| Step 1 | `run_reward_vllm.py` | Critic evaluates initial `Retrieve` ($t=1$) given `instruction` only | Seed `INPUT_FILE` | `INITIAL_RETRIEVAL_TOKEN_OUTPUT` |
| Step 2 | `run_reward_vllm.py` | Critic evaluates `IsUse` (`[Utility:1..5]`) given `instruction` and full `output` | Seed `INPUT_FILE` | `UTILITY_OUTPUT` |
| Step 3a | `create_retrieval_data.py` | Builds Pass 1 queries (`instruction + " " + output`) | Seed `INPUT_FILE` | `INITIAL_RETRIEVAL_INPUT` |
| Step 3b | `passage_retrieval.py` | Contriever retrieves top-10 passages for Pass 1 | `INITIAL_RETRIEVAL_INPUT` | `INITIAL_RETRIEVAL_OUTPUT` |
| Step 4a | `create_retrieval_data.py` (`--multiple_sent`) | Splits sentences via spaCy, attaches top-1 passage from Step 3b as `evidence`, builds Pass 2 queries (`instruction + " " + sent_t`) | `INPUT_FILE`, `INITIAL_RETRIEVAL_TOKEN_OUTPUT`, `INITIAL_RETRIEVAL_OUTPUT` | `MULTI_RETRIEVAL_INPUT` and `_splitted` |
| Step 4b | `run_reward_vllm.py` | Critic evaluates 3-way `Retrieve` ($t>1$) given `instruction`, `preceding_sentences`, `evidence`, and `target_output` | `MULTI_RETRIEVAL_INPUT` | `MULTI_RETRIEVAL_TOKEN_OUTPUT` |
| Step 4c | `passage_retrieval.py` | Contriever retrieves top-10 passages per sentence for Pass 2 | `MULTI_RETRIEVAL_INPUT` | `MULTI_RETRIEVAL_OUTPUT` |
| Step 5 | `create_prompt_data.py` | Pairs each sentence $y_t$ with its top-10 passages from Step 4c into batched files | `MULTI_RETRIEVAL_OUTPUT`, `MULTI_RETRIEVAL_TOKEN_OUTPUT` | `prompt_data_batch_{i}.jsonl` |
| Step 6 | `run_reward_vllm.py` | Critic evaluates `IsRel` and `IsSup` on every sentence-passage pair | `prompt_data_batch_{i}.jsonl` | `REL_OUTPUT_FILE_{i}`, `SUP_OUTPUT_FILE_{i}` |
| Step 7 | `postprocess_data.py` | Selects the best passage per sentence and assembles the final interleaved sequence | All outputs from Steps 1-6 | `FINAL_OUTPUT_PATH` ($\mathcal{D}_{\text{gen}}$) |

One mismatch in the documented commands: for Step 4b, the README runs the Critic with the two-way prompt (`--inst_mode retrieval_multi_instruction --input_mode retrieval_multi_input`, which offers only `[No Retrieval]` and `[Retrieval]`), whereas the Critic was trained on multi-sentence retrieval only with the three-way prompt (`multi_retrieval_three_way_instruction`). Producing `[Continue to Use Evidence]` labels as described in the paper requires the three-way prompt.

---

## 3. Line-by-Line Walkthrough of `create_retrieval_data.py`

Let us examine `reference/code/data_creation/generator/create_retrieval_data.py` in detail.

### 3.1 Sentence Segmentation with spaCy (`split_sentences`, Lines 6-14)

At the top of `create_retrieval_data.py` (lines 6-14), the script loads the spaCy English pipeline `"en_core_web_sm"` to segment paragraph responses into individual sentences:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
TASK_DATA = ["nq", "wow", "fever", "tqa", "arc_easy", "arc_hard", "obqa", "qrecc", "race", "asqa"]
def split_sentences(paragraph):
    doc = nlp(paragraph)
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)
    return sentences
```

Why use `nlp(paragraph).sents` instead of splitting on periods (`paragraph.split(".")`)?
Because English text frequently uses periods inside abbreviations (`"U.S."`, `"Dr."`, `"e.g."`, `"King Louis XIV."`) and decimal numbers (`"3.14"`). A naive period split would slice a single sentence in half at `"U.S."`, whereas spaCy's dependency parser (`en_core_web_sm`) identifies true grammatical sentence boundaries.

### 3.2 Mode 1: Initial Retrieval Query Construction (`--multiple_sent` False, Lines 107-117)

When `create_retrieval_data.py` is run without `--multiple_sent` (Step 3a in the table above), it executes the `else:` branch on lines 107-117:

```python
        else:
            for idx, item in tqdm(enumerate(data)):
                q_id = item["id"]
                instruction = item["instruction"]
                if item["dataset_name"] in ["nq", "fever", "wow"] and "\n\nInput: " in instruction:
                    instruction = instruction.split("\n\nInput: ")[1]
                output = item["output"]
                # add output for retrieval
                processed_entry = {"question": instruction +
                                   " " + output, "answers": [output], "q_id": q_id, }
                processed_data.append(processed_entry)
```

For each example, this creates a dictionary with `"question": instruction + " " + output` so Contriever can retrieve passages relevant to the complete response.

Notice that this branch tries to strip a task prefix with the delimiter `"\n\nInput: "` (and only for `nq`, `fever`, and `wow`), but the preprocessing scripts from Module 04 Part 3 join task instructions with `"## Input:\n\n"`. The condition therefore never matches, so for `fever`, `wow`, `arc_easy`, `obqa`, and `asqa` the Pass 1 queries still contain the generic task instruction (`nq` has no instruction prefix to begin with). Pass 1 only supplies the `evidence` passage used by the multi-sentence retrieval Critic, so this affects that input rather than the passages inserted into $\mathcal{D}_{\text{gen}}$.

### 3.3 Mode 2: Sentence-Level Splitting and Query Construction (`--multiple_sent` True, Lines 38-105)

When `--multiple_sent` is passed (Step 4a in the table above), the script loads the initial retrieval necessity predictions and the top-1 passage from Mode 1, and then splits each response into sentences:

```python
    if args.need_retrieval_files is not None:
        retrieval_necessity = {}
        for f in args.need_retrieval_files:
            retrieval_necessity_f = json.load(open(f))
            retrieval_necessity.update({item["id"]: item["pred"] for item in retrieval_necessity_f})
    else:
        retrieval_necessity = None

    id2evidence = None
    if args.initial_retrieval_file is not None:
        if args.initial_retrieval_file.endswith(".json"):
            data = json.load(open(args.initial_retrieval_file))
        else:
            data = load_jsonlines(args.initial_retrieval_file)
        if "id" in data[0]:
            id2evidence = {item["id"]: item["ctxs"][0] for item in data if ("sent_idx" not in item or item["sent_idx"] == 0)}
        elif "q_id" in data[0]:
            id2evidence = {item["q_id"]: item["ctxs"][0] for item in data if ("sent_idx" not in item or item["sent_idx"] == 0)}
```

The README says continuous retrieval runs only for queries whose initial retrieval decision is true, and line 74 below tries to enforce this with `retrieval_necessity[q_id] is False`. However, `run_reward_vllm.py` stores each Critic prediction in `"pred"` as decoded text (such as `"[No Retrieval]"`), never as the boolean `False`, so the check never fires: every example is split and sent to Pass 2, and the initial decision is applied later, in `postprocess_data.py`.

Now trace the sentence loop on lines 63-105:

```python
        if args.multiple_sent is True:
            for idx, item in tqdm(enumerate(data)):
                q_id = item["id"]
                instruction = item["instruction"]
                dataset_name = item["dataset_name"]
                if dataset_name in TASK_DATA and "## Input:\n\n" in instruction:
                    # For task data, we remove the task-specific instruction
                    instruction = instruction.split("## Input:\n\n")[1]
                if len(item["input"]) > 0 and item["input"] not in instruction:
                    instruction = instruction + " " + item["input"]
                output = item["output"]
                if retrieval_necessity is not None and q_id in retrieval_necessity and retrieval_necessity[q_id] is False:
                    continue

                splitted_output = split_sentences(output)
                skipped = {}
                for sent_idx in range(len(splitted_output)):
                    if len(splitted_output) > 2 and len(splitted_output[sent_idx]) < 30:
                        skipped[sent_idx] = True
                        continue
                    else:
                        skipped[sent_idx] = False

                    question = instruction + " " + splitted_output[sent_idx]
                    output = splitted_output[sent_idx]
                    if sent_idx > 0:
                        preceding_sentences = " ".join(
                            splitted_output[:sent_idx])
                    else:
                        preceding_sentences = None
                    if "evidence" in item:
                        evidence = item["evidence"]
                    else:
                        if id2evidence is not None and q_id in id2evidence:
                            evidence = id2evidence[q_id]["title"] + "\n" + id2evidence[q_id]["text"]
                        else:
                            evidence = None
                    processed_entry = {"question": question, "answers": [output], "output": item["output"], "target_output": output,
                                       "instruction": instruction, "preceding_sentences": preceding_sentences,
                                       "q_id": q_id, "sent_idx": sent_idx, "evidence": evidence, "dataset_name": dataset_name}
                    splitted_data.append({"q_id": q_id, "dataset_name": dataset_name,
                                     "instruction": instruction, "output": output, "splitted_output": splitted_output, "skipped": skipped})
                    processed_data.append(processed_entry)
```

Let us examine four important mechanisms in this loop:

1. Stripping `"## Input:\n\n"` on task datasets (lines 68-70):
   As we saw in Module 04 Part 3, task datasets like `arc_easy`, `obqa`, `fever`, `wow`, and `asqa` had `TASK_INST[dataset_name] + "## Input:\n\n"` prepended to their questions. Line 70 strips away everything before `"## Input:\n\n"` so the generic task instructions do not pollute the search query `question`.

2. The Short-Sentence Skipping Rule (`len(splitted_output) > 2 and len(splitted_output[sent_idx]) < 30`, lines 80-84):
   Why does the script skip any sentence shorter than $30$ characters when a response has more than $2$ sentences?
   In instruction-following datasets (such as GPT-4 Alpaca), multi-sentence responses often begin with short conversational openers like `"Sure!"` ($5$ characters) or `"Here are three examples:"` ($24$ characters), or contain short list headers like `"1. Memory:"` ($10$ characters).
   Retrieving $10$ Wikipedia passages and inserting `<paragraph>...</paragraph>` after `"Sure!"` would be noisy and unnatural. By marking `skipped[sent_idx] = True` and skipping retrieval for that short sentence, `postprocess_data.py` (lines 243-245) simply keeps the short sentence in the output text as-is without attaching any retrieval or critique tokens to it.

3. Constructing `processed_entry` for each non-skipped sentence (lines 86-105):
   Let us trace a concrete 3-sentence response for `q_id = "alpaca_10"` where:
   - `instruction` $=$ `"Explain computer memory."`
   - `splitted_output`:
     - `sent_idx = 0`: `"Sure!"` ($5$ chars $< 30 \implies$ `skipped[0] = True`, skipped!)
     - `sent_idx = 1`: `"Computer memory stores data for immediate use."` ($46$ chars $\ge 30 \implies$ `skipped[1] = False`)
     - `sent_idx = 2`: `"RAM is a fast form of volatile memory."` ($38$ chars $\ge 30 \implies$ `skipped[2] = False`)

   | `sent_idx` | `skipped[sent_idx]` | `question` (Sent to Contriever) | `target_output` | `preceding_sentences` | `evidence` (From Pass 1 Top-1) |
   |---|---|---|---|---|---|
   | $0$ | `True` | (Skipped) | (Skipped) | (Skipped) | (Skipped) |
   | $1$ | `False` | `"Explain computer memory. Computer memory stores data for immediate use."` | `"Computer memory stores data for immediate use."` | `"Sure!"` | `id2evidence["alpaca_10"]` (`title + "\n" + text`) |
   | $2$ | `False` | `"Explain computer memory. RAM is a fast form of volatile memory."` | `"RAM is a fast form of volatile memory."` | `"Sure! Computer memory stores data for immediate use."` | `id2evidence["alpaca_10"]` (`title + "\n" + text`) |

4. Saving both `args.output_file` and `args.output_file + "_splitted"` (lines 103-120):
   - `processed_data` (saved to `args.output_file`) contains one entry per non-skipped sentence, ready for Pass 2 Contriever retrieval and 3-way Critic retrieval evaluation.
   - `splitted_data` (saved to `args.output_file + "_splitted"`) records the full list `splitted_output` and the `skipped` dictionary for each `q_id` so `postprocess_data.py` can reassemble all sentences (including the skipped short sentences) in order. Notice that because `splitted_data.append(...)` (lines 103-104) is indented inside the `for sent_idx` loop, an example with $M$ non-skipped sentences appends $M$ copies of its `splitted_data` entry; `postprocess_data.py` (lines 234-235) simply reads the first copy `splitted_output_data[q_id][0]`.
