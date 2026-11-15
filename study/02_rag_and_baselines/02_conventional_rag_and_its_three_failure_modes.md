# Conventional RAG and Its Three Failure Modes

## 1. Plain-English Intuition: Turning a Closed-Book Exam into an Open-Book Exam

In Part 1, we saw why standalone language models hallucinate on rare facts: they are taking a closed-book exam, relying entirely on whatever facts were compressed into their fixed weight matrices $\theta$ during pre-training.

The most direct way to help a student on a closed-book exam is to make it an open-book exam. Instead of expecting the neural network weights $\theta$ to memorize every fact in Wikipedia, we keep an external searchable collection of millions of text passages on disk, denoted $\mathcal{C}_{\text{corpus}} = \{d_1, d_2, \dots, d_N\}$. Because this knowledge lives in explicit text documents outside the model's parameters $\theta$, it is called non-parametric memory.

In conventional Retrieval-Augmented Generation (RAG), answering a user prompt $x$ follows a fixed two-step recipe (shown on the left side of Figure 1 in the paper):
1. Step 1 (Retrieve $K$ documents): Before the language model generates anything, a search engine (the retriever $\mathcal{R}$) takes the user prompt $x$, searches the external corpus $\mathcal{C}_{\text{corpus}}$, and returns a fixed number $K$ of top-matching passages (for example, $K = 3$ or $K = 5$ passages).
2. Step 2 (Prompt LM with $K$ documents and generate): All $K$ retrieved passages are pasted into the input prompt right before the user's question $x$. The language model reads the combined prompt (`Passages 1..K + Question`) and generates the entire response $y$ in one go.

While pasting retrieved passages into the prompt helps on simple fact-lookup questions, this rigid two-step pipeline introduces three severe problems of its own. Below, we build up the math of conventional RAG, see how concatenating $K$ passages affects self-attention, and walk through the three failure modes illustrated in Figure 1 (left).

---

## 2. Step-by-Step Mathematical Build-Up: Conventional RAG and Attention Dilution

### 2.1 Formal Equations of Conventional RAG

Let $\mathcal{C}_{\text{corpus}} = \{d_1, d_2, \dots, d_N\}$ denote the external collection of $N$ text passages (for example, $N \approx 21\text{ million}$ Wikipedia paragraphs).

Given an input prompt $x$ and a fixed integer $K \in \mathbb{Z}^+$, Step 1 runs the retriever $\mathcal{R}$ to fetch the $K$ passages with the highest similarity scores to $x$:

$$D_K = \mathcal{R}(x; \mathcal{C}_{\text{corpus}}, K) = \left(d_{(1)}, d_{(2)}, \dots, d_{(K)}\right)$$

In Step 2, all $K$ passages in $D_K$ are concatenated with the prompt $x$ into a single long input sequence. The language model $\theta$ then generates the response $y = (y_1, y_2, \dots, y_n)$ token by token using the standard chain rule from Module 01:

$$p_\theta(y \mid x, D_K) = \prod_{j=1}^{n} p_\theta\left(y_j \;\middle|\; d_{(1)}, d_{(2)}, \dots, d_{(K)}, x, y_{<j}\right)$$

where:
- $K \in \mathbb{Z}^+$ is the fixed number of passages retrieved for every query.
- $d_{(k)}$ is the token sequence of the $k$-th retrieved passage, for $k \in \{1, \dots, K\}$.
- $x$ is the token sequence of the user prompt.
- $y_{<j} = (y_1, \dots, y_{j-1})$ is the prefix of response tokens generated before step $j$.
- $p_\theta(y_j \mid d_{(1)}, \dots, d_{(K)}, x, y_{<j})$ is the next-token probability distribution when the transformer attends across all $K$ passages, the prompt $x$, and $y_{<j}$ simultaneously.

### 2.2 How Concatenating $K$ Passages Dilutes Self-Attention

Why does pasting $K$ passages together into one long sequence cause trouble inside the transformer?

Recall from Module 01 Part 2 how causal self-attention works at generation step $j$: the current position computes a positive exponentiated similarity score $\exp(\langle q_t, k_i \rangle / \sqrt{d_h})$ for every earlier token position $i$, and divides by the sum over all earlier positions so the attention weights add up to $1.0$.

When $d_{(1)}, d_{(2)}, \dots, d_{(K)}$ and $x$ are concatenated into a single sequence, let $S_k > 0$ denote the sum of exponentiated attention scores across all tokens inside passage $d_{(k)}$, and let $S_x > 0$ denote the sum of exponentiated attention scores across the prompt $x$ and preceding response $y_{<j}$:

$$S_k = \sum_{i \in \text{tokens}(d_{(k)})} \exp\left(\frac{\langle q_t, k_i \rangle}{\sqrt{d_h}}\right), \quad S_x = \sum_{i \in \text{tokens}(x, y_{<j})} \exp\left(\frac{\langle q_t, k_i \rangle}{\sqrt{d_h}}\right)$$

Because the softmax denominator sums over the entire concatenated prompt, the fraction of total attention $A_k \in (0, 1)$ paid to passage $d_{(k)}$ is:

$$A_k = \frac{S_k}{S_1 + S_2 + \dots + S_K + S_x}$$

And the blended attention output vector $o_t \in \mathbb{R}^{d_h}$ becomes a weighted average of the value vectors $\bar{v}_{(k)}$ from all $K$ passages plus the prompt $\bar{v}_x$:

$$o_t = A_1 \bar{v}_{(1)} + A_2 \bar{v}_{(2)} + \dots + A_K \bar{v}_{(K)} + A_x \bar{v}_x$$

where:
- $S_k \in (0, \infty)$ is the unnormalized attention mass attracted by passage $d_{(k)}$.
- $A_k \in (0, 1)$ is the normalized share of attention paid to passage $d_{(k)}$, with $\sum_{k=1}^K A_k + A_x = 1.0$.
- $\bar{v}_{(k)} \in \mathbb{R}^{d_h}$ is the weighted average of value vectors inside passage $d_{(k)}$.

### 2.3 Toy Numerical Example of Attention Dilution

Suppose Passage 1 ($d_{(1)}$) is genuinely relevant to the user's question and attracts an unnormalized score of $S_1 = 8.0$, while the prompt $x$ attracts $S_x = 2.0$.

Compare what happens when the model reads only Passage 1 ($K = 1$) versus when we concatenate two extra retrieved passages $d_{(2)}$ and $d_{(3)}$ ($K = 3$) that are off-topic but contain overlapping keywords (so they attract $S_2 = 6.0$ and $S_3 = 4.0$):

| Setup | $S_1$ (Relevant) | $S_2$ (Distractor) | $S_3$ (Distractor) | $S_x$ (Prompt) | Total Denominator | Attention on Relevant Passage $A_1$ | Attention on Distractors $A_2 + A_3$ |
|---|---|---|---|---|---|---|---|
| Clean ($K = 1$) | $8.0$ | $0.0$ | $0.0$ | $2.0$ | $8.0 + 2.0 = 10.0$ | $8.0 / 10.0 = 0.80$ ($80\%$) | $0.00$ ($0\%$) |
| Concatenated ($K = 3$) | $8.0$ | $6.0$ | $4.0$ | $2.0$ | $8.0 + 6.0 + 4.0 + 2.0 = 20.0$ | $8.0 / 20.0 = 0.40$ ($40\%$) | $(6.0 + 4.0) / 20.0 = 0.50$ ($50\%$) |

Look at the two consequences in the $K = 3$ row:
1. The attention share on the truly relevant passage $A_1$ is cut in half (dropping from $80\%$ down to $40\%$) because $S_2$ and $S_3$ inflate the denominator from $10.0$ to $20.0$.
2. Half of the resulting hidden state vector $o_t$ ($A_2 + A_3 = 50\%$) is now made of value vectors $\bar{v}_{(2)}$ and $\bar{v}_{(3)}$ pulled from the distracting passages!

---

## 3. Walkthrough of the Three Failure Modes of Conventional RAG (Figure 1 Left)

Figure 1 (left) in the Self-RAG paper illustrates three concrete failure modes that result directly from this static, concatenated RAG design.

### 3.1 Failure Mode 1: Indiscriminate Retrieval Hurts Versatility

In conventional RAG, retrieval is hard-coded to run on every single input prompt $x$ with a fixed $K > 0$. There is no mechanism to ask whether retrieval is actually needed.

Look at the bottom-left example in Figure 1:
- Prompt $x$: `"Write an essay of your best summer vacation"`

This prompt asks for a creative, personal narrative. It does not ask for any encyclopedic fact. Yet conventional RAG still queries the retriever $\mathcal{R}$ with `"Write an essay of your best summer vacation"`, retrieves $K = 3$ Wikipedia passages matching words like `"summer"` and `"vacation"`, and forces the language model to condition on them:

$$p_\theta\left(y \;\middle|\; d_{(1)}, d_{(2)}, d_{(3)}, \text{``Write an essay of your best summer vacation''}\right)$$

Retrieving indiscriminately when factual grounding is unnecessary causes two problems:
1. Loss of versatility: Injecting encyclopedic passages into the attention window biases the model away from creative, open-ended instruction following.
2. Wasted computation: Running dense search over millions of passages and running transformer self-attention over hundreds of extra passage tokens slows down inference for zero benefit.

### 3.2 Failure Mode 2: Distraction and Contradiction from Irrelevant Context

Even when a prompt is a factual question that genuinely benefits from retrieval, search engines are imperfect: among the top $K$ retrieved passages, some passages will be relevant while others will be irrelevant keyword matches. Because conventional RAG concatenates all $K$ passages together without filtering them, the model gets distracted and mixes them up.

Let us trace the top-left example in Figure 1 step by step:
- Prompt $x$: `"How did US states get their names?"`
- Step 1 (Retrieve $K = 3$ documents):
  - Passage 1 ($d_{(1)}$): `"Of the fifty states, eleven are named after an individual person."` (Relevant fact about state name origins)
  - Passage 2 ($d_{(2)}$): `"Popular names by states. In Texas, Emma is a popular baby name."` (Irrelevant! The retriever fetched this passage only because it contains the words `"names"` and `"states"`, even though it is about baby names in Texas.)
  - Passage 3 ($d_{(3)}$): `"California was named after a fictional island in a Spanish book."` (Relevant fact about California's name origin)

- Step 2 (Prompt LM with `Prompt + [1] + [2] + [3]` and generate):
  Conditioned jointly on all three passages at once, the language model generates:

  `"US states got their names from a variety of sources. Eleven states are named after an individual person (e.g., California was named after Christopher Columbus). Some states including Texas and Utah, are named after American tribe."`

Look at the two specific factual errors highlighted in Figure 1 (left):

| Generated Span in Conventional RAG | Error Type in Figure 1 | Likely Cause (Interpretation; Figure 1 Only Labels the Error) |
|---|---|---|
| `"(e.g., California was named after Christopher Columbus)"` | `Contradictory` | Passage 1 talks about states named after an individual person, and Passage 3 mentions `"California"` and `"Spanish"`. Attending to both passages simultaneously in one context causes the model to blend them together and invent `"California was named after Christopher Columbus"`, directly contradicting Passage 3 (`"named after a fictional island in a Spanish book"`). |
| `"Some states including Texas and Utah, are named after American tribe."` | `No information in passages` | The irrelevant Passage 2 contains the word `"Texas"` (in `"In Texas, Emma is a popular baby name"`). Because `"Texas"` is sitting in the attention window, the model's attention pulls `"Texas"` into the generation and fabricates an unsupported claim about Texas and Utah that does not appear in any of the three passages. |

### 3.3 Failure Mode 3: Single-Pass Static Retrieval and No Verification

Finally, notice what happens during and after generation in conventional RAG:
1. Retrieval happens only once at the start: In `"How did US states get their names?"`, the retriever is called only once before the first word is generated. If the model wants to write a third sentence about states named after Native American tribes (such as Utah), it cannot pause mid-response to retrieve a new passage specifically about Native American state names.
2. No self-evaluation or verification: Conventional RAG generates the response in a single forward pass and never checks whether its output sentences are actually entailed by the passages. Because base and instruction-tuned LMs are not explicitly trained to verify support from provided passages, they frequently ignore the passages or attach wrong citations. In Table 2 of the paper, conventional RAG with Llama-2 7B/13B and Alpaca 7B/13B achieves citation precision between $2.0\%$ and $5.5\%$ on ASQA, meaning over $94\%$ of their citations fail to properly support the claimed sentence.
