# Seed Datasets and Preprocessing Scripts

## 1. Plain-English Intuition: Why Self-RAG Trains on a Hybrid Mixture of 11 Datasets

In Part 1 and Part 2, we saw the $15$ special tokens and how they are registered in the tokenizer. Before we can insert those special tokens into training examples, we need a raw seed collection of input-output pairs $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N_{\text{seed}}}$.

How should we choose which datasets go into $\mathcal{D}$?
- Suppose we only included Wikipedia question-answering datasets (like Natural Questions or FEVER). Then almost every training example would require retrieval (`[Retrieval]`), and almost every target output $y_i$ would be a single short phrase or `"true"` / `"false"`. A model trained only on QA datasets would forget how to write essays, poems, or multi-step explanations, and would never learn when to predict `[No Retrieval]`.
- Conversely, suppose we only included general chat datasets (like ShareGPT or Alpaca). Then the model would see plenty of open-ended writing, but would rarely practice strict fact verification, scientific multiple-choice reasoning, or synthesizing multi-faceted long-form answers from evidence.

To ensure the Generator $\mathcal{M}$ masters both open-ended versatility and factual grounding, Self-RAG constructs a seed corpus of $145,619$ examples (referred to in rounded terms as $150\text{k}$ in Section 4.3 of the paper) by combining two categories of data (Appendix A.2 Table 3). The paper titles Table 3 "The generator LM M training data statistics", yet the postprocessing step in Module 07 drops and downsamples examples, so these counts cannot describe both the raw seed pool and the final training set; the paper does not say which stage they count. This module treats them as the seed mixture:
1. Instruction-following datasets ($92,008$ examples, $63.2\%$ of the corpus): Five subsets from the Open-Instruct (Tulu) collection covering creative writing, coding, brainstorming, and general assistance.
2. Knowledge-intensive datasets ($53,611$ examples, $36.8\%$ of the corpus): Six benchmarks covering open-domain QA, fact verification, knowledge-grounded dialogue, multiple-choice science reasoning, and ambiguous long-form QA.

---

## 2. Composition of the 145,619 Seed Corpus and Target JSONL Schema

### 2.1 Breakdown of the 11 Seed Datasets (Appendix Table 3)

The table below lists all $11$ seed datasets from Appendix Table 3, their category, their preprocessing script in `reference/code/data_creation/process_data/`, and their exact instance counts:

| Dataset Name | Category | Source Collection | Preprocessor Script | Instance Count | Share of Total ($145,619$) |
|---|---|---|---|---|---|
| GPT-4 Alpaca | Instruction-following | Open-Instruct | `tulu.py` | $26,168$ | $18.0\%$ |
| Stanford Alpaca | Instruction-following | Open-Instruct | `tulu.py` | $25,153$ | $17.3\%$ |
| FLAN-V2 | Instruction-following | Open-Instruct | `tulu.py` | $17,817$ | $12.2\%$ |
| ShareGPT | Instruction-following | Open-Instruct | `tulu.py` | $13,406$ | $9.2\%$ |
| Open Assistant 1 (`oasst1`) | Instruction-following | Open-Instruct | `tulu.py` | $9,464$ | $6.5\%$ |
| Wizard of Wikipedia (`wow`) | Knowledge-intensive | KILT | `kilt_dpr.py` | $17,367$ | $11.9\%$ |
| Natural Questions (`nq`) | Knowledge-intensive | KILT | `kilt_dpr.py` | $15,535$ | $10.7\%$ |
| FEVER (`fever`) | Knowledge-intensive | KILT | `kilt_dpr.py` | $9,966$ | $6.8\%$ |
| OpenBookQA (`obqa`) | Knowledge-intensive | Hugging Face `datasets` | `obqa.py` | $4,699$ | $3.2\%$ |
| ASQA (`asqa`) | Knowledge-intensive | ASQA official release | `asqa.py` | $3,897$ | $2.7\%$ |
| ARC-Easy (`arc_easy`) | Knowledge-intensive | Hugging Face `datasets` | `arc.py` | $2,147$ | $1.5\%$ |
| Total | | | | $145,619$ | $100.0\%$ |

### 2.2 The Unified Target Schema

In their raw downloaded forms, these $11$ datasets have completely different JSON structures. The preprocessing scripts in `reference/code/data_creation/process_data/` convert every dataset into a single standardized JSONL format required by the downstream generator data creation pipeline (`reference/code/data_creation/generator/README.md` lines 24-31):

```json
{
    "instruction": "string (the task instruction and input query)",
    "output": "string (the ground-truth response)",
    "input": "",
    "topic": "string (Wikipedia entity title if available, else empty)",
    "id": "string (unique identifier, e.g., fever_42)",
    "dataset_name": "string (dataset prefix, e.g., fever)"
}
```

---

## 3. Line-by-Line Walkthrough: Processing Open-Instruct Datasets (`tulu.py`)

Let us inspect `reference/code/data_creation/process_data/tulu.py`, which processes the five Open-Instruct instruction-following datasets (`gpt4_alpaca`, `stanford_alpaca`, `flan_v2`, `sharegpt`, and `oasst1`):

```python
def process_tulu_dataset(fp, single_turn_only=True, max_n=1000, dataset_name=False):
    data = load_jsonlines(fp)
    processed_data = []
    for item in tqdm(data):
        messages = item["messages"]
        if single_turn_only is True and len(messages) > 2:
            continue
        # currently only support single turn
        input = messages[0]["content"].replace("\nOutput:", "")
        if len(input.split(" ")) > 500:
            continue 
        if input[-2:] == "\n\n":
            input = input[:-2]
        if input[-1:] == "\n":
            input = input[:-1]
        output = messages[1]["content"]
        if len(output) == 0:
            continue
        if output[0] == "\n":
            output = output[1:]
        if len(output.split(" ")) > 500:
            continue
        if dataset_name == "sharegpt" or "oasst1": 
            try:
                if detect(input) != "en" or detect(output) != "en":
                    # print("multilingual input")
                    continue
            except:
                print("non text input")
        id = item["id"]
        processed_data.append({"input": "", "instruction": input, "output": output, "id": id, "dataset_name": dataset_name})
    if max_n is not None and max_n < len(processed_data):
        processed_data = random.sample(processed_data, k=max_n)
    return processed_data
```

### 3.1 Filtering Rules in `process_tulu_dataset`

Each raw item in an Open-Instruct file contains a `"messages"` list of alternating user and assistant turns:
`[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]`.

`process_tulu_dataset` applies four sequential filters to keep training sequences clean and within the GPU context window:
1. Single-turn filter (lines 18-20): `if single_turn_only is True and len(messages) > 2: continue` discards multi-turn conversations that have more than one user message and one assistant response. Notice that in `main()` (line 60), `single_turn_only=True` is hardcoded regardless of the `--single_turn_only` CLI flag.
2. Input word-count filter and cleanup (lines 22-28): extracts `messages[0]["content"]`, strips any trailing `"\nOutput:"` marker left over from FLAN templates, discards prompts longer than $500$ space-separated words (`len(input.split(" ")) > 500`), and strips trailing newlines.
3. Output word-count filter and cleanup (lines 29-35): extracts `messages[1]["content"]`, discards empty responses (`len(output) == 0`), strips a leading newline, and discards responses longer than $500$ words (`len(output.split(" ")) > 500`). Limiting raw inputs and outputs to $\le 500$ words each is critical because Stage 3 will insert multiple $100$-word Wikipedia paragraphs into the output, and the total sequence must fit inside the $2048$-token (7B) or $1536$-token (13B) training limit (the 13B script sets $1536$; Appendix B.1 of the paper reports $1,524$).
4. Random subsampling (lines 44-45): if `--n` (`max_n`) is specified and smaller than the filtered dataset size, `random.sample(processed_data, k=max_n)` randomly samples `max_n` examples.

### 3.2 Diagnosing the Boolean Precedence Bug on Line 38 of `tulu.py`

Look closely at line 38 of `tulu.py`:

```python
        if dataset_name == "sharegpt" or "oasst1": 
            try:
                if detect(input) != "en" or detect(output) != "en":
                    # print("multilingual input")
                    continue
            except:
                print("non text input")
```

The author's intent was to run the `langdetect.detect` language detector only when `dataset_name` is `"sharegpt"` or `"oasst1"` (since crowd-sourced ShareGPT and OpenAssistant logs contain many non-English conversations).

However, in Python, the equality operator `==` has higher precedence than `or`. Python evaluates line 38 as:

$$\left(\text{dataset\_name} == \text{``sharegpt''}\right) \;\text{or}\; \left(\text{``oasst1''}\right)$$

In Python, any non-empty string (such as `"oasst1"`) is truthy (`bool("oasst1") == True`). Therefore:
- Even when `dataset_name` is `"gpt4_alpaca"`, `dataset_name == "sharegpt"` is `False`, and `False or "oasst1"` evaluates to `"oasst1"` (which is truthy!).
- Consequently, the `if` condition evaluates to `True` on every single dataset, running `detect(input) != "en" or detect(output) != "en"` across all five Open-Instruct subsets. Also note that if `detect()` raises a `LangDetectException` (for example, on a purely numeric or code-only string), the `except:` block prints `"non text input"` without a `continue`, allowing that item to be kept.

---

## 4. Line-by-Line Walkthrough: Processing Knowledge-Intensive Datasets

### 4.1 Task Instructions and the `"## Input:\n\n"` Separator (`task_instructions.py`)

`reference/code/data_creation/process_data/task_instructions.py` defines the instruction dictionary imported by `kilt_dpr.py`, `arc.py`, `obqa.py`, and `asqa.py`:

```python
TASK_INST = {
    "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
    "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
    "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.", 
    "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.", 
    "arc_hard": "Given four answer candidates, A, B, C and D, choose the best answer choice.", 
    "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.", 
    "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
}
```

In all four knowledge-intensive scripts below, whenever `args.data_prefix` is in `TASK_INST`, the task instruction is joined to the question using the exact delimiter `"## Input:\n\n"`:

```python
instruction = TASK_INST[args.data_prefix] + "## Input:\n\n" + raw_question
```

Why do these scripts insert `"## Input:\n\n"` between the task instruction and the question?
Because later, when we run the dense retriever $\mathcal{R}$ (Contriever) in `reference/code/data_creation/generator/create_retrieval_data.py` (lines 68-70), we do not want the generic sentence `"Given four answer candidates, A, B, C and D, choose the best answer choice."` to pollute the Wikipedia search query! Having `"## Input:\n\n"` allows `create_retrieval_data.py` to strip the generic task prefix via `instruction = instruction.split("## Input:\n\n")[1]` so only the actual question is sent to the search engine. This stripping happens only in the sentence-level pass (`--multiple_sent`). The initial whole-output pass (lines 111-112) looks for a different delimiter, `"\n\nInput: "`, which these scripts never write, so its queries keep the task instruction.

### 4.2 KILT Datasets: Natural Questions, Wizard of Wikipedia, and FEVER (`kilt_dpr.py`)

`reference/code/data_creation/process_data/kilt_dpr.py` processes KILT datasets that have been formatted in Dense Passage Retrieval (DPR) JSON structure:

```python
    for idx, item in enumerate(input_data):
        q_id = "{0}_{1}".format(args.data_prefix, idx)
        instruction = item["question"]
        if args.data_prefix in TASK_INST:
            instruction = TASK_INST[args.data_prefix] + "## Input:\n\n" + instruction
        output = item["answers"][0]
        topic = item["positive_ctxs"][0]["title"]
        if args.data_prefix == "fever":
            if output not in ["REFUTES", "SUPPORTS"]:
                print(output)
            output = "false" if output == "REFUTES" else "true"
        new_data.append({"instruction": instruction, "output": output, "input": "", "topic": topic, "id": q_id, "dataset_name": args.data_prefix})
```

Tracing how the three KILT datasets (`nq`, `wow`, `fever`) pass through this loop:
- `nq` (Natural Questions): `"nq"` is not in `TASK_INST`, so `instruction` remains the raw question `item["question"]` (such as `"when did walking dead season 7 come out"`), and `output` is the short gold answer `item["answers"][0]` (`"October 23 , 2016"`).
- `wow` (Wizard of Wikipedia): `"wow"` is in `TASK_INST`, so `instruction` gets the dialogue instruction prefix followed by `"## Input:\n\n"` and the newline-separated chat history.
- `fever` (FEVER fact verification): `"fever"` is in `TASK_INST`, and lines 25-28 convert the original KILT label `"REFUTES"` to `"false"` and `"SUPPORTS"` to `"true"`, matching the instruction `"Say true if it's correct; otherwise say false."`.
- In all three datasets, `topic` stores the Wikipedia article title of the first gold passage (`item["positive_ctxs"][0]["title"]`).

### 4.3 Multiple-Choice Science Datasets (`arc.py` and `obqa.py`)

`reference/code/data_creation/process_data/arc.py` (for `ARC-Easy`) and `obqa.py` (for `OpenBookQA`) load multiple-choice science questions directly from Hugging Face `datasets`:

```python
    # From arc.py (lines 15-40):
    data = datasets.load_dataset("ai2_arc", "ARC-Easy")["train"]
    new_data = []
    for item in data:
        question_stem = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]
        if answer_key == "1":
            answer_key = "A"
        if answer_key == "2":
            answer_key = "B"
        if answer_key == "3":
            answer_key = "C"
        if answer_key == "4":
            answer_key = "D"
        answer_labels = {}
        choices["label"] = ["A", "B", "C", "D"]
        if len(choices["text"]) < 4:
            continue
        for text, choice in zip(choices["text"], choices["label"]):
            answer_labels[choice] = text
        q_id = "{0}_{1}".format(args.data_prefix, item["id"])
        input_question = "{0}\nA: {1}\nB: {2}\nC: {3}\nD: {4}".format(question_stem, answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
        instruction = TASK_INST[args.data_prefix] + "## Input:\n\n" + input_question
        output = answer_key
        new_data.append({"instruction": instruction, "output": output, "input": "", "id": q_id, "dataset_name": args.data_prefix})
```

Look at what `arc.py` and `obqa.py` do to each multiple-choice question:
1. In `ARC-Easy`, some items label their choices `"1", "2", "3", "4"` instead of `"A", "B", "C", "D"`, and a few items have only $3$ options. `arc.py` maps `"1".."4"` to `"A".."D"`, overwrites `choices["label"] = ["A", "B", "C", "D"]`, and skips any question with fewer than $4$ choice strings (`if len(choices["text"]) < 4: continue`). Questions with more than four options are not filtered: only the first four are shown, so an answer key of `"E"` (or `"5"`) would point to an option that is not in the prompt.
2. Both scripts format the stem and the four options into a single multi-line block:
   `"{question_stem}\nA: {choice_A}\nB: {choice_B}\nC: {choice_C}\nD: {choice_D}"`
3. Both scripts prepend `TASK_INST[args.data_prefix] + "## Input:\n\n"` and set `"output"` to the single capital letter `"A"`, `"B"`, `"C"`, or `"D"`.

### 4.4 Ambiguous Long-Form QA (`asqa.py`)

Finally, `reference/code/data_creation/process_data/asqa.py` processes the training split of ASQA:

```python
    input_data = json.load(open(args.input_file))["train"]

    new_data = []
    for sample_id, item in input_data.items():
        question = item["ambiguous_question"]
        q_id = "{0}_{1}".format(args.data_prefix, sample_id)
        dataset_name = args.data_prefix
        instruction = TASK_INST[args.data_prefix] + "## Input:\n\n" + question
        output = item["annotations"][0]["long_answer"]
        new_data.append({"instruction": instruction, "output": output, "input": "",
                        "topic": "", "id": q_id, "dataset_name": dataset_name})
```

In ASQA, each question (`item["ambiguous_question"]`) has multiple valid factual interpretations depending on context (for example, `"When did Sunday become the seventh day of the week?"`), and `item["annotations"][0]["long_answer"]` is a multi-sentence paragraph written by human annotators that synthesizes all disambiguated answers. Including these $3,897$ multi-sentence ASQA examples in the seed corpus provides rich multi-sentence targets for learning segment-by-segment retrieval and citation in Stage 3.
