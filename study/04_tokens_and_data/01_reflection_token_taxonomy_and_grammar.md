# Reflection Token Taxonomy and Grammar

## 1. Plain-English Intuition: Why Self-RAG Needs a Structured Grammar of Special Tokens

In Module 03, we saw that Self-RAG trains a single language model $\mathcal{M}$ to decide when to search Wikipedia, judge whether a retrieved passage is relevant, generate a grounded sentence, verify whether that sentence is supported by the passage, and rate the overall usefulness of the response.

Why does Self-RAG represent these judgments using $15$ dedicated special tokens (such as `[Retrieval]`, `[Relevant]`, and `[Fully supported]`) rather than letting the model write ordinary English sentences like `"I think I should search Wikipedia now"` or `"This sentence is mostly supported"`?

There are three reasons:
1. Unambiguous parsing and control: Because `[Retrieval]` is a dedicated special token that never appears in normal English prose, the inference loop can watch for `[Retrieval]` at every step and immediately pause generation to call the search engine $\mathcal{R}$.
2. Single-step probability scoring: Because `[Relevant]` and `[Irrelevant]` are single tokens in the vocabulary $\mathcal{V}_{\text{exp}}$, the model outputs their exact probabilities in a single forward pass at the first token position after `</paragraph>`. If the critique were a 10-word English sentence, we could not read a clean single-token probability to score and rank passages.
3. Clean removal before showing the answer to the user: Once generation finishes, a simple string replacement (`pred.replace(token, "")`) strips each listed special token, leaving only the clean natural-language answer.

To understand how Self-RAG sequences are structured during both training and inference, we will examine the exact definitions of all $15$ added special tokens, the formal sequence grammar governing where each token can appear, and a token-by-token walkthrough of the five training examples in Appendix Table 4 of the paper.

---

## 2. Complete Taxonomy of the 15 Special Tokens (Table 1 & Appendix A.1)

In the paper's figures and tables, the authors use compact shorthand labels (such as `[Retrieve=Yes]` or `<p>`) to save space. In the actual Python codebase (`reference/code/data_creation/train_special_tokens.py` line 306 and `reference/code/retrieval_lm/finetune.py` line 380), the tokenizer registers $15$ exact string literals under `additional_special_tokens`.

The table below lists all $15$ special tokens, grouped into their five functional categories:

| Group | Paper Shorthand | Exact String Literal in Codebase | Level of Application | Plain-English Definition and Condition |
|---|---|---|---|---|
| `Retrieve` | `[Retrieve=Yes]` | `[Retrieval]` | Before segment $y_t$ | Factual grounding is needed for the next sentence; calls retriever $\mathcal{R}$ to insert `<paragraph>d</paragraph>`. |
| `Retrieve` | `[Retrieve=No]` | `[No Retrieval]` | Before segment $y_t$ | Factual grounding is unnecessary (for example, creative writing, personal experience, or common sense) or would not improve the sentence. |
| `Retrieve` | `[Retrieve=Continue]` | `[Continue to Use Evidence]` | Before segment $y_t$ ($t > 1$) | Factual grounding is needed, and the passage retrieved in a previous segment already contains the facts required to verify the next sentence. |
| Context Markup | `<p>` | `<paragraph>` | Start of passage $d$ | Marks the beginning of an externally retrieved passage $d$. |
| Context Markup | `</p>` | `</paragraph>` | End of passage $d$ | Marks the end of an externally retrieved passage $d$. |
| `IsRel` | `[IsREL=Relevant]` | `[Relevant]` | Immediately after `</paragraph>` | The retrieved passage $d$ provides useful information for solving the input instruction $x$ given preceding context $y_{<t}$. |
| `IsRel` | `[IsREL=Irrelevant]` | `[Irrelevant]` | Immediately after `</paragraph>` | The retrieved passage $d$ is off-topic or does not provide useful information for $x$. |
| `IsSup` | `[IsSUP=Fully Supported]` | `[Fully supported]` | Immediately after segment $y_t$ | Every verification-worthy factual statement in segment $y_t$ is entailed by or extracted from the preceding `[Relevant]` passage $d$. |
| `IsSup` | `[IsSUP=Partially Supported]` | `[Partially supported]` | Immediately after segment $y_t$ | Segment $y_t$ is partially supported by passage $d$, but contains major factual details that are not discussed in $d$. |
| `IsSup` | `[IsSUP=No Support]` / `[Contradictory]` | `[No support / Contradictory]` | Immediately after segment $y_t$ | Segment $y_t$ completely ignores passage $d$, is unrelated to $d$, or directly contradicts facts stated in $d$. |
| `IsUse` | `[IsUSE=5]` | `[Utility:5]` | End of full response $y$ | The complete response $y$ is complete, highly detailed, and fully satisfies the query $x$ (perceived plausibility/helpfulness). |
| `IsUse` | `[IsUSE=4]` | `[Utility:4]` | End of full response $y$ | The response mostly fulfills the need in $x$, with only minor room for improvement in detail or structure. |
| `IsUse` | `[IsUSE=3]` | `[Utility:3]` | End of full response $y$ | The response is acceptable, but requires major additions or improvements to fully satisfy $x$. |
| `IsUse` | `[IsUSE=2]` | `[Utility:2]` | End of full response $y$ | The response addresses the main topic of $x$, but is incomplete or misses the specific request. |
| `IsUse` | `[IsUSE=1]` | `[Utility:1]` | End of full response $y$ | The response is barely on-topic or completely irrelevant to $x$. |

---

## 3. Formal Sequence Grammar of a Self-RAG Output

A complete Self-RAG target output $y = (y_1, y_2, \dots, y_T)$ consists of $T$ sentence segments followed by a single utility token `IsUse` and the end-of-sequence token `</s>`.

Depending on the retrieval and relevance decisions at step $t \in \{1, \dots, T\}$, each segment $y_t$ follows one of four exact structural patterns (implemented in `reference/code/data_creation/generator/postprocess_data.py` lines 330-344). Let $s_t$ denote the natural-language tokens of the $t$-th sentence, and let $d_t$ denote the natural-language tokens of the passage retrieved at step $t$:

### 3.1 Pattern 1: Retrieval Skipped (`[No Retrieval]`)

$$\text{[No Retrieval]} \; s_t$$

When `[No Retrieval]` is predicted before sentence $s_t$:
- No `<paragraph>d_t</paragraph>` block is inserted.
- No `IsRel` token (`[Relevant]` or `[Irrelevant]`) is generated, because there is no passage to evaluate for relevance.
- No `IsSup` token (`[Fully supported]`, `[Partially supported]`, or `[No support / Contradictory]`) is generated after $s_t$, because there is no passage against which to measure entailment.

### 3.2 Pattern 2: Retrieval Triggered and Passage Is Relevant (`[Retrieval]` + `[Relevant]`)

$$\text{[Retrieval]} \; \text{<paragraph>} \, d_t \, \text{</paragraph>} \; \text{[Relevant]} \; s_t \; \text{IsSup}_t$$

where $\text{IsSup}_t \in \{\text{[Fully supported]},\; \text{[Partially supported]},\; \text{[No support / Contradictory]}\}$.

When `[Retrieval]` is predicted and the inserted passage $d_t$ is judged `[Relevant]`:
- `[Relevant]` appears immediately after `</paragraph>` and immediately before the first word of sentence $s_t$.
- Immediately after the last word (or period) of sentence $s_t$, the model must output one of the three `IsSup` tokens grading how well $d_t$ supports $s_t$.

### 3.3 Pattern 3: Retrieval Triggered and Passage Is Irrelevant (`[Retrieval]` + `[Irrelevant]`)

$$\text{[Retrieval]} \; \text{<paragraph>} \, d_t \, \text{</paragraph>} \; \text{[Irrelevant]} \; s_t$$

Look closely at the end of Pattern 3 compared to Pattern 2: when the inserted passage $d_t$ is judged `[Irrelevant]`, no `IsSup` token is placed after $s_t$!

Why is `IsSup` omitted after `[Irrelevant]`?
In `reference/code/data_creation/generator/postprocess_data.py` (lines 332-336), the training sequence is assembled as:

```python
if relevance_token == "[Relevant]":
    output += "[Retrieval]" + "<paragraph>{}</paragraph>".format(processed_paragraph) + "[Relevant]" + sentence + groundness_token
else:
    output += "[Retrieval]" + "<paragraph>{}</paragraph>".format(processed_paragraph) + "[Irrelevant]" + sentence
```

If a retrieved passage $d_t$ is completely irrelevant to the user's prompt, it cannot provide factual support for a good answer sentence $s_t$. By omitting `groundness_token` (`IsSup`) whenever `relevance_token == "[Irrelevant]"`, the training data teaches the Generator $\mathcal{M}$ two things when it encounters an irrelevant passage:
1. Flag the passage immediately as `[Irrelevant]`.
2. Still generate a helpful continuation sentence $s_t$ using parametric knowledge, without falsely claiming that the irrelevant passage supported $s_t$.

### 3.4 Pattern 4: Continuing to Use Previous Evidence (`[Continue to Use Evidence]`)

$$\text{[Continue to Use Evidence]} \; s_t$$

(At inference time, when a single passage generates multiple sentences, `IsSup` may also follow the final sentence derived from that passage, as seen in Appendix Table 7.) During generator training data creation (`postprocess_data.py` line 341), `[Continue to Use Evidence]` is prepended directly to $s_t$ without inserting a duplicate `<paragraph>` block, signaling that $s_t$ relies on the passage already present in the preceding context.

### 3.5 End-of-Response Termination

After all $T$ sentence segments $(y_1, \dots, y_T)$ have been generated, the response concludes with exactly one utility critique token from `IsUse` followed by `</s>`:

$$y = y_1 \; y_2 \; \dots \; y_T \; \text{IsUse} \; \text{</s>}, \quad \text{where } \text{IsUse} \in \{\text{[Utility:1]}, \dots, \text{[Utility:5]}\}$$

---

## 4. Token-by-Token Walkthrough of the 5 Real Training Examples in Appendix Table 4

Appendix Table 4 in the paper displays five real training examples from $\mathcal{D}_{\text{gen}}$. Walking through all five examples shows how Patterns 1, 2, and 3 combine in practice across different tasks.

### 4.1 Example 1: Multi-Passage Technical Explanation (`gpt4_alpaca`)

- Input $x$: `"Identify the two types of computer memory."`
- Structure of Augmented Output:
  1. Segment 1 (Pattern 2: Relevant + Fully supported):
     - `[Retrieval]`
     - `<paragraph>Computer memory ... refers to the computer hardware integrated circuits that store information for immediate use in a computer; it is synonymous with the term "primary storage" ... "main memory" ...</paragraph>`
     - `[Relevant]`
     - Sentence $s_1$: `"Primary memory, also known as main memory, is the memory that the computer can directly access and manipulate."`
     - `[Fully supported]` (The passage explicitly defines computer memory as primary storage / main memory for immediate use.)
  2. Segment 2 (Pattern 2: Relevant + Partially supported):
     - `[Retrieval]`
     - `<paragraph>Computer memory - often associated with addressable semiconductor memory ... There are two main kinds of semiconductor memory, volatile and non-volatile ... Examples of volatile memory are primary storage ...</paragraph>`
     - `[Relevant]`
     - Sentence $s_2$: `"Primary memory can be accessed quickly, but it is volatile, meaning that the data is lost if the power is turned off."`
     - `[Partially supported]` (Why partial instead of full? Because the retrieved passage states that primary storage is volatile, but the passage gets cut off before defining `"volatile"` as `"data is lost if the power is turned off"`. That definition came from parametric knowledge, so the Critic accurately labels it `[Partially supported]`!)
  3. End of response: `... [Utility:5]`

### 4.2 Example 2: Hybrid Open-Ended and Grounded Response (`gpt4_alpaca`)

- Input $x$: `"Give an example of a situation where you have to delegate tasks"`

This example is particularly instructive because its $6$ sentences use Pattern 1 (`[No Retrieval]`), Pattern 2 (`[Retrieval]` + `[Relevant]`), and Pattern 3 (`[Retrieval]` + `[Irrelevant]`):

| Segment $t$ | Pattern Used | Reflection & Context Tokens Before $s_t$ | Generated Sentence $s_t$ | Token After $s_t$ | Why This Pattern Occurred |
|---|---|---|---|---|---|
| $t = 1$ | Pattern 1 | `[No Retrieval]` | `"The manager realizes that there is an extensive list of tasks that need to be completed in order to make the event a success."` | None | Sets up a fictional story scenario; needs no Wikipedia lookup. |
| $t = 2$ | Pattern 2 | `[Retrieval] <p>Delegation is the assignment of any responsibility or authority to another person ...</p> [Relevant]` | `"This includes tasks such as creating a guest list, designing and sending invitations, booking a venue, coordinating with vendors, planning the events program, and monitoring the budget."` | `[Partially supported]` | Passage defines management delegation (`[Relevant]`), but the specific party-planning examples come from the story, so `[Partially supported]`. |
| $t = 3$ | Pattern 1 | `[No Retrieval]` | `"Since it is not possible for the manager to handle everything alone, they must delegate tasks to their team members."` | None | Common-sense narrative transition; needs no retrieval. |
| $t = 4$ | Pattern 1 | `[No Retrieval]` | `"The manager would then assign each team member specific responsibilities based on their individual skillsets and strengths."` | None | Narrative continuation; needs no retrieval. |
| $t = 5$ | Pattern 3 | `[Retrieval] <p>Joint application design - Subject Matter Experts: These are the business users ...</p> [Irrelevant]` | `"For instance, the team's graphic designer might be tasked with designing the invitations, while the event coordinator would be in charge of planning the program and working with vendors..."` | None (`IsSup` omitted!) | Retriever returned an off-topic Wikipedia article about `"Joint application design"` (`[Irrelevant]`). Model flags `[Irrelevant]`, writes $s_5$ from parametric memory, and omits `IsSup`. |
| $t = 6$ | Pattern 1 | `[No Retrieval]` | `"By delegating tasks, the manager can ensure that all necessary tasks are completed in a timely manner, while also empowering their team members to take on responsibilities and contribute to the success of the event."` | `[Utility:5]` | Concluding summary sentence (`[No Retrieval]`), followed by `[Utility:5]` at the end of the response. |

### 4.3 Example 3: Pure Creative Writing (`stanford_alpaca`)

- Input $x$: `"Synthesize a poem based on this prompt: 'The sun is setting'"`
- Augmented Output:
  `"[No Retrieval] The stars begin to twinkle bright the trees, scattered throughout the night; [No Retrieval] ... [No Retrieval] They flicker like a silent dance, A peaceful moment in its trance. [No Retrieval] ... [No Retrieval] A chorus of crickets sing a lullaby, To every creature that roams the sky [Utility:5]"`
- Because writing a poem requires zero factual verification, every single verse uses Pattern 1 (`[No Retrieval]`), and the response ends with `[Utility:5]`.

### 4.4 Example 4: Single-Segment Factual Open-Domain QA (`nq`)

- Input $x$: `"when did walking dead season 7 come out"`
- Augmented Output:
  `"[Retrieval] <p>The Walking Dead (season 7) - The seventh season of 'The Walking Dead' ... premiered on October 23, 2016, and concluded on April 2, 2017 ...</p> [Relevant] October 23 , 2016 [Fully supported] [Utility:5]"`
- Here, the entire answer is a single short date phrase (`"October 23 , 2016"`). It uses Pattern 2 (`[Retrieval]` $\to$ `<p>...</p>` $\to$ `[Relevant]` $\to$ answer $\to$ `[Fully supported]`) followed immediately by `[Utility:5]`.

### 4.5 Example 5: Grounded Yet Low-Utility Dialogue Response (`wow`)

- Input $x$: `"Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response: Hello, I love German shepherd dogs and you? - My favorite dog, they are officially known as German Shepherd Dogs? - Yes, I have a German shepherd in my house and she is pregnant"`
- Augmented Output:
  `"They also are easily trained, obedient, and intelligent. [Retrieval] <p>German Shepherd - ... They are known for being easy to train and good for performing tasks and following instructions. They are especially well known for their police work ...</p> [Relevant] Because of this, they are often used for disability assistance and for police work. [Partially supported] [Utility:2]"`

(Table 4 prints this output starting mid-word, as `"hey also are easily trained..."`, and with no retrieval token before the first sentence, so the beginning of the output appears to be cut off in the table.)

Look at why this fifth example ends with `[Utility:2]` even though the passage is `[Relevant]` and the sentence is `[Partially supported]`:
- In the chat history, the user just said: `"Yes, I have a German shepherd in my house and she is pregnant"`.
- Instead of responding to the user's pregnant dog, the target response ignores the pregnancy and recites generic facts about German Shepherds doing police work.
- Therefore, while the factual claim about police work is grounded in the Wikipedia passage (`[Partially supported]`), its conversational helpfulness (`IsUse`) is poor, so the Critic labels it `[Utility:2]`. Training the Generator on examples with `[Utility:1]` and `[Utility:2]` (as well as `[Utility:5]`) teaches the Generator to distinguish high-utility responses from low-utility responses at inference time.
