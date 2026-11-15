# The Self-RAG Core Idea and Inference Cycle

## 1. Plain-English Intuition: Turning a Passive Generator into an Active Researcher and Self-Critic

In Module 02, we saw why conventional Retrieval-Augmented Generation (RAG) breaks down:
1. It retrieves passages blindly on every prompt, even when the user asks for a creative personal story (`"Write an essay of your best summer vacation"`).
2. When it retrieves $K$ passages, it pastes all $K$ passages together into one long prompt, allowing irrelevant keyword matches (`"In Texas, Emma is a popular baby name"`) to dilute self-attention and trigger hallucinations.
3. It retrieves only once at the start of generation and never checks whether each sentence it writes is actually supported by a passage.

To see how Self-RAG fixes all three problems, think about how a careful human researcher writes an answer to `"How did US states get their names?"`:
1. Decide when to search: Before writing each sentence, the researcher asks, *"Do I need external facts for this next sentence?"* If the prompt is `"Write an essay of your best summer vacation"`, the researcher says *"No retrieval needed"* and simply writes from imagination. If the prompt is about state name history, the researcher says *"Yes, let me search."*
2. Read each search result separately: When the search engine returns $3$ passages, the researcher does not blend all $3$ passages into one confusing paragraph. Instead, they evaluate each passage on its own:
   - Reading Passage 1 alone: *"Is this passage relevant to my question? Yes (`[Relevant]`). Using Passage 1, I can write: '11 of 50 state names come from persons.' Does Passage 1 completely prove that sentence? Yes (`[Fully supported]`)."*
   - Reading Passage 2 alone (Texas baby names): *"Is this passage relevant to how states got their names? No (`[Irrelevant]`)."*
   - Reading Passage 3 alone (California Spanish book): *"Is this passage relevant? Yes (`[Relevant]`). Using Passage 3, I write: 'California's name has its origins in a 16th-century novel Las Sergas de Esplandian.' Does Passage 3 prove every detail in that sentence? Only partially (`[Partially supported]`), because Passage 3 only said 'a Spanish book' and did not mention the 16th century or the title Las Sergas de Esplandian."*
3. Pick the best-supported sentence and repeat: Comparing the three candidate sentences and their self-evaluations, the researcher picks the sentence from Passage 1 (because it is both `Relevant` and `Fully supported`), cites Passage 1 (`[1]`), and then asks again whether to run a new search for the next sentence!

Self-RAG builds this exact three-step researcher workflow directly into a single language model $\mathcal{M}$. Instead of using external classifier models or complicated rules, Self-RAG adds special reflection tokens (such as `[Retrieval]`, `[No Retrieval]`, `[Relevant]`, `[Irrelevant]`, `[Fully supported]`, `[Partially supported]`, and `[Utility:5]`) directly into the model's vocabulary $\mathcal{V}$. Because these reflection tokens are just extra token IDs in the vocabulary, the language model predicts them using the exact same next-token softmax mechanism we learned in Module 01.

---

## 2. Step-by-Step Formulation: Segment-Level Generation and Reflection Tokens

### 2.1 Expanding the Vocabulary and Splitting Outputs into Segments

Let $\mathcal{V}$ denote the original vocabulary of $V = 32000$ subword tokens from Module 01, and let $\mathcal{V}_{\text{ref}}$ denote the set of newly added special tokens (the reflection tokens plus the `<paragraph>` and `</paragraph>` delimiters). The expanded vocabulary of the Self-RAG generator $\mathcal{M}$ is:

$$\mathcal{V}_{\text{exp}} = \mathcal{V} \cup \mathcal{V}_{\text{ref}}$$

Given an input prompt $x$, instead of treating the target response $y$ as one undivided block of text, Self-RAG decomposes $y$ into a sequence of $T$ segments:

$$y = (y_1, y_2, \dots, y_T)$$

where:
- $T \in \mathbb{Z}^+$ is the number of segments in the complete response.
- $y_t$ is the sequence of tokens for the $t$-th segment (in all experiments in the paper and codebase, one segment $y_t$ is one sentence, plus its accompanying reflection tokens).
- $y_{<t} = (y_1, \dots, y_{t-1})$ denotes all segments generated before step $t$, with $y_{<1} = \emptyset$.

### 2.2 Overview of the Four Reflection Token Groups (Table 1)

Table 1 in Section 3 of the paper organizes the reflection tokens into one retrieval token group (`Retrieve`) and three critique token groups (`IsRel`, `IsSup`, `IsUse`):

| Token Group | Category | What the Model Reads Before Predicting This Token | Possible Output Token Values (Most Desirable First) | Plain-English Question Answered by This Token |
|---|---|---|---|---|
| `Retrieve` | Retrieval token | Input prompt $x$ and preceding segments $y_{<t}$ | `yes` (`[Retrieval]`), `no` (`[No Retrieval]`), `continue` (`[Continue to Use Evidence]`) | Do I need to call the retriever $\mathcal{R}$ to ground the next sentence, skip retrieval, or keep using the previous passage? |
| `IsRel` | Critique token | Input prompt $x$ and retrieved passage $d$ | `relevant` (`[Relevant]`), `irrelevant` (`[Irrelevant]`) | Does retrieved passage $d$ provide useful information to answer $x$? |
| `IsSup` | Critique token | Input prompt $x$, passage $d$, and generated segment $y_t$ | `fully supported` (`[Fully supported]`), `partially supported` (`[Partially supported]`), `no support` (`[No support / Contradictory]`) | Are all verification-worthy factual claims in my generated sentence $y_t$ entailed by passage $d$? |
| `IsUse` | Critique token | Input prompt $x$ and complete response $y$ | `5` (`[Utility:5]`), `4`, `3`, `2`, `1` (`[Utility:1]`) | Overall, how helpful, informative, and complete is the response $y$ for answering $x$ (from $1$ lowest to $5$ highest)? |

---

## 3. Step-by-Step Walkthrough of the Self-RAG Inference Cycle (Algorithm 1 & Figure 1 Right)

Let us trace Algorithm 1 from Section 3.1 of the paper alongside the two visual examples on the right side of Figure 1.

At segment step $t$, the generator language model $\mathcal{M}$ has already read the user prompt $x$ and generated $t - 1$ preceding sentence segments $y_{<t} = (y_1, \dots, y_{t-1})$. To produce the $t$-th segment $y_t$, Self-RAG executes three steps:

### 3.1 Step 1: Retrieve on Demand (Algorithm 1, Lines 2-4 and 8)

First, $\mathcal{M}$ predicts a `Retrieve` decision token conditioned on $(x, y_{<t})$ (or compares the probability of `[Retrieval]` against a threshold $\delta$, which we will derive in Module 09):

1. Branch A: Retrieval is not needed (`Retrieve == No`, Algorithm 1 Lines 8-10):
   Look at the bottom-right example in Figure 1:
   - Prompt $x$: `"Write an essay of your best summer vacation"`
   - $\mathcal{M}$ predicts `[No Retrieval]`.
   - Because `[No Retrieval]` was predicted, the retriever $\mathcal{R}$ is never called. $\mathcal{M}$ immediately generates the response segment $y_t$ (`"My best summer vacation is when my family and I embarked on a road trip along ..."`), and at the end of the response predicts the overall utility token `IsUse` (for example, `[Utility:5]`).

2. Branch B: Retrieval is needed (`Retrieve == Yes`, Algorithm 1 Lines 3-4):
   Look at the top-right example in Figure 1:
   - Prompt $x$: `"How did US states get their names?"`
   - For the first sentence $y_1$, $\mathcal{M}$ writes a general opening sentence `"US states got their names from a variety of sources."` and then outputs the token `[Retrieval]`.
   - As soon as $\mathcal{M}$ outputs `[Retrieval]`, the inference script calls the retriever $\mathcal{R}$ to fetch $K$ candidate passages $D = \{d_1, d_2, \dots, d_K\}$ from the corpus $\mathcal{C}_{\text{corpus}}$.

### 3.2 Step 2: Generate Segments in Parallel per Passage (Algorithm 1, Line 5)

Why did conventional RAG fail in Module 02 Part 2 when the retriever returned $D = \{d_1, d_2, d_3\}$? Because conventional RAG concatenated all three passages into one prompt (`Prompt + d_1 + d_2 + d_3`), allowing the irrelevant baby-name passage $d_2$ to steal attention away from $d_1$ and $d_3$.

Self-RAG completely eliminates cross-passage distraction by processing the $K$ retrieved passages in parallel as $K$ separate batch items rather than concatenating them.

For $K = 3$ retrieved passages $d_1, d_2, d_3$, Self-RAG creates a batch of $3$ independent prompts, where each prompt contains only one passage $d_k$ wrapped inside `<paragraph>` and `</paragraph>` tags:
- Parallel Branch $k = 1$: `Prompt + y_{<t} + [Retrieval]<paragraph>d_1</paragraph>`
- Parallel Branch $k = 2$: `Prompt + y_{<t} + [Retrieval]<paragraph>d_2</paragraph>`
- Parallel Branch $k = 3$: `Prompt + y_{<t} + [Retrieval]<paragraph>d_3</paragraph>`

Because these three branches are separate rows in a GPU batch tensor of shape $[K=3, L, d]$, tokens in Branch $1$ can only attend to Passage $d_1$; they have zero visibility into Passage $d_2$ or Passage $d_3$.

In each branch $k \in \{1, \dots, K\}$ in parallel, $\mathcal{M}$ generates:
1. The relevance critique token `IsRel` (`[Relevant]` or `[Irrelevant]`) immediately after `</paragraph>`.
2. The candidate sentence segment $y_t^{(k)}$ conditioned on $(x, y_{<t}, d_k)$.

### 3.3 Step 3: Critique Outputs and Select Best Segment (Algorithm 1, Lines 6-7)

Immediately after generating the candidate sentence $y_t^{(k)}$ in Branch $k$, $\mathcal{M}$ generates its self-critique token `IsSup` (`[Fully supported]`, `[Partially supported]`, or `[No support / Contradictory]`) and `IsUse` (`[Utility:1]`..`[Utility:5]`). Algorithm 1 (line 6) lists `IsUse` at this step, while Appendix A.1 says the utility aspect is given only once per output. In the training data and in the model's own outputs, the `IsUse` token appears once, just before `</s>`.

Let us compare the exact outputs of all three parallel branches in Figure 1 (top right) for `"How did US states get their names?"`:

| Parallel Branch $k$ | Passage $d_k$ Provided to Branch $k$ | Predicted `IsRel` Token | Candidate Sentence $y_t^{(k)}$ Generated by $\mathcal{M}$ | Predicted `IsSup` Token | Why $\mathcal{M}$ Assigned These Critique Tokens | Rank |
|---|---|---|---|---|---|---|
| Branch $1$ (`Prompt + d_1`) | `"Of the fifty states, eleven are named after an individual person."` | `[Relevant]` (Green) | `"11 of 50 state names come from persons."` | `[Supported]` (`[Fully supported]`, Green) | Passage $d_1$ directly answers the prompt, and every claim in $y_t^{(1)}$ is entailed by $d_1$. | $1$st (Selected) |
| Branch $3$ (`Prompt + d_3`) | `"California was named after a fictional island in a Spanish book."` | `[Relevant]` (Green) | `"California's name has its origins in a 16th-century novel Las Sergas de Esplandian."` | `[Partially]` (`[Partially supported]`, Orange) | Passage $d_3$ is relevant, and supports that California was named after a Spanish book, but $d_3$ does not mention `"16th-century"` or `"Las Sergas de Esplandian"` (which came from parametric memory). | $2$nd (Dropped) |
| Branch $2$ (`Prompt + d_2`) | `"Popular names by states. In Texas, Emma is a popular baby name."` | `[Irrelevant]` (Red) | `"Texas is named after a Native American tribe."` | None / Unverified | Passage $d_2$ is about baby names, not state name origins, so $\mathcal{M}$ flags it as `[Irrelevant]`. | $3$rd (Dropped) |

Look at what happens in Step 3 when Self-RAG ranks the three branches using their critique tokens:
1. Branch $2$ is immediately penalized because its passage was flagged as `[Irrelevant]`. The unverified claim `"Texas is named after a Native American tribe"` is discarded.
2. Branch $3$ is recognized as `[Relevant]`, but because the model included extra parametric details (`"16th-century novel Las Sergas de Esplandian"`) not present in the short passage $d_3$, it honestly grades itself as `[Partially supported]`.
3. Branch $1$ is both `[Relevant]` and `[Fully supported]`, so it achieves the highest segment score. Self-RAG selects `"11 of 50 state names come from persons."` as the winning segment $y_2$ and cites Passage $d_1$ (`[1]`).
4. Next, to continue answering the question, $\mathcal{M}$ emits `[Retrieval]` again, retrieves a new passage $d_4$ that actually discusses Native American state names, and generates the supported third sentence: `"26 states are named after Native Americans, including Utah. [4]"`.

---

## 4. Paper and Codebase Grounding: Quick-Start Execution Trace

In the Quick start section of `reference/code/README.md`, the authors provide a minimal Python script demonstrating how the trained Self-RAG model (`selfrag/selfrag_llama2_7b`) executes Step 1, Step 2, and Step 3 using vLLM:

```python
from vllm import LLM, SamplingParams

model = LLM("selfrag/selfrag_llama2_7b", download_dir="/gscratch/h2lab/akari/model_cache", dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt

query_1 = "Leave odd one out: twitter, instagram, whatsapp."
query_2 = "Can you tell me the difference between llamas and alpacas?"
queries = [query_1, query_2]

preds = model.generate([format_prompt(query) for query in queries], sampling_params)
for pred in preds:
  print("Model prediction: {0}".format(pred.outputs[0].text))
```

Look at the two outputs printed by this first pass (when `paragraph=None`):

```text
Model prediction: Twitter, Instagram, and WhatsApp are all social media platforms. [No Retrieval]WhatsApp is the odd one out because it is a messaging app, while Twitter and Instagram are primarily used for sharing photos and videos.[Utility:5]</s>
Model prediction: Sure![Retrieval]<paragraph><paragraph>
```

Tracing both outputs token by token:
- On `query_1` (`"Leave odd one out: twitter, instagram, whatsapp."`), which is a common-sense reasoning question:
  1. The model writes the first sentence `"Twitter, Instagram, and WhatsApp are all social media platforms."`
  2. Before writing the second sentence, it predicts `[No Retrieval]`, continues with `"WhatsApp is the odd one out because it is a messaging app, while Twitter and Instagram are primarily used for sharing photos and videos."`, predicts `[Utility:5]`, and finishes with `</s>`.
- On `query_2` (`"Can you tell me the difference between llamas and alpacas?"`), which requires specific biological facts:
  1. The model writes `"Sure!"` and immediately emits `[Retrieval]<paragraph>`, signaling that it wants a retrieved Wikipedia passage inserted inside `<paragraph>...</paragraph>`.

Next, in the same Quick start section, the script calls `format_prompt` with the retrieved Wikipedia paragraph for `"alpaca"` inserted inside `[Retrieval]<paragraph>{0}</paragraph>`:

```python
prompt = format_prompt(
    "Can you tell me the difference between llamas and alpacas?",
    "The alpaca (Lama pacos) is a species of South American camelid mammal. It is similar to, and often confused with, the llama. Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber."
)
preds = model.generate([prompt], sampling_params)
print([pred.outputs[0].text for pred in preds])
```

And the model outputs:

```text
['[Relevant]Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber.[Fully supported][Utility:5]</s>']
```

Look at the exact sequence of tokens generated after `</paragraph>`:
1. `[Relevant]` (`IsRel`): confirms the inserted Wikipedia passage is relevant to the question.
2. `"Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber."` ($y_t$): generates the grounded response sentence.
3. `[Fully supported]` (`IsSup`): verifies that every claim in the generated sentence is entailed by the inserted passage.
4. `[Utility:5]` (`IsUse`) and `</s>`: rates the overall utility as $5$ out of $5$ and terminates generation.
