# Distilling GPT-4 Supervision and Prompts

## 1. Plain-English Intuition: How Knowledge Distillation Creates the Critic Model ($\mathcal{C}$)

In Module 04, we saw that Self-RAG adds $15$ special tokens (such as `[Retrieval]`, `[Relevant]`, `[Fully supported]`, and `[Utility:5]`) to the Llama-2 vocabulary.

However, a pre-trained Llama-2 7B model has no idea what those $15$ new tokens mean because their embedding rows are freshly initialized. To annotate our $145,619$-example Generator training corpus $\mathcal{D}_{\text{gen}}$, we first need a Critic model $\mathcal{C}$ that can read an instruction $x$, a retrieved passage $d$, and an output sentence $y_t$, and accurately predict the right reflection token $r$.

Where do we get the training labels to teach the Critic $\mathcal{C}$?
While a base Llama-2 7B model does not know how to grade passages or sentences out of the box, a frontier proprietary model like GPT-4 can follow detailed natural-language grading rubrics and few-shot demonstrations; Appendix A.1 reports $80\%$ to $95\%$ agreement between GPT-4's labels and the authors' own judgments on $20$ sampled instances per aspect.

Self-RAG transfers GPT-4's grading ability into the local Llama-2 7B Critic $\mathcal{C}$ via knowledge distillation:
1. Prompt GPT-4 (the teacher): We sample a subset of examples from our seed datasets and prompt GPT-4 with a natural-language rubric $I_{\text{type}}$ and few-shot examples, asking GPT-4 to output both a reflection label $r$ and a natural-language explanation (`Explanation: ...`).
2. Strip the explanation and keep the atomic token $r$: Having GPT-4 write an explanation improves its grading accuracy and allows humans to audit its reasoning. However, when we build the Critic training set $\mathcal{D}_{\text{critic}}$, we discard the explanation and train the Critic $\mathcal{C}$ (the student) to output only the single reflection token $r$ (for example, `[Fully supported]`).
3. Fast, free offline annotation: Because the trained Critic $\mathcal{C}$ outputs a single token $r$ in $1$ forward pass on local GPUs, we can run $\mathcal{C}$ millions of times offline in Stage 3 at zero API cost.

---

## 2. Formal Distillation Setup, Dataset Counts, and Human Agreement (Section 3.2.1 & Appendix A.1)

### 2.1 Formalizing GPT-4 Supervision Generation

For each reflection token group $\text{type} \in \{\text{Retrieve}_{\text{init}}, \text{Retrieve}_{\text{multi}}, \text{IsRel}, \text{IsSup}, \text{IsUse}\}$, we construct a prompt consisting of:
- A task-specific grading instruction $I_{\text{type}}$.
- A sequence of $K_{\text{shot}}$ few-shot demonstrations $\mathcal{E}_{\text{demos}} = \{(x_{\text{demo}}^{(k)}, r_{\text{demo}}^{(k)}, e_{\text{demo}}^{(k)})\}_{k=1}^{K_{\text{shot}}}$, separated by `"##"` or `"###"`.
- The sampled test instance $x_{\text{inst}}$ (containing the relevant combination of instruction $x$, preceding sentences $y_{<t}$, retrieved evidence $d$, and target output $y_t$ or $y$).

GPT-4 samples a text completion containing the predicted reflection token $r$ followed by a natural-language explanation $e_{\text{expl}}$:

$$(r, e_{\text{expl}}) \sim p_{\text{GPT-4}}\left(r, e_{\text{expl}} \;\middle|\; I_{\text{type}}, \mathcal{E}_{\text{demos}}, x_{\text{inst}}\right)$$

As detailed in Appendix A.1, GPT-4 is queried with `temperature = 1` and `max_tokens = 200`. Any completion where GPT-4 violates the output format or produces a label string that does not match one of the valid reflection categories for that group is discarded.

### 2.2 Collected Supervision Counts and Human Validation Agreement

To verify that GPT-4's reflection token labels are trustworthy before distilling them into $\mathcal{C}$, the authors manually evaluated $20$ randomly sampled instances per reflection aspect against the same grading rubrics. The table below summarizes the collected dataset sizes and human agreement rates reported in Appendix A.1:

| Reflection Aspect | Inputs Provided to GPT-4 | Valid Output Categories | Collected Instances | Human Agreement (%) | Disagreement Noted in Appendix A.1 |
|---|---|---|---|---|---|
| `Retrieve` | $(x)$ for initial; $(x, y_{<t}, d, y_t)$ for $t > 1$ | `[Yes]`/`[No]` (initial); `[Continue to Use Evidence]`/`[No Retrieval]`/`[Retrieval]` ($t > 1$) | $12,594$ (printed as "1,2594" in the paper) | $95\%$ | Not reported |
| `IsRel` | $(x, y_{<t}, d)$ | `[Relevant]`, `[Irrelevant]` | $19,317$ | $95\%$ | Not reported |
| `IsSup` | $(x, y_{<t}, y_t, d)$ | `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]` | $11,181$ | $90\%$ | Not reported |
| `IsUse` | $(x, y)$ | `5`, `4`, `3`, `2`, `1` | $3,831$ | $80\%$ | Mostly adjacent ratings (`1` vs `2`, or `4` vs `5`) |
| Total | | | $46,923$ | | |

With only $20$ instances per aspect, each instance moves the percentage by $5$ points ($95\% = 19/20$, $80\% = 16/20$), so these agreement figures are rough estimates.

---

## 3. Walkthrough of the Five GPT-4 Instruction Rubrics and Demonstrations (Appendix D, Tables 8-12)

Appendix Section D (Tables 8 through 12) provides the exact instructions and demonstrations used to prompt GPT-4. Examining the demonstrations in each table reveals the exact decision boundaries that Self-RAG learns.

### 3.1 Rubric 1: Initial Retrieval Necessity (`Retrieve` Given Input $x$ Only, Table 8)

When evaluating whether an instruction $x$ requires retrieval at the very start of generation ($t = 1$), GPT-4 is given only the instruction $x$ and asked:
*"Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer [Yes] or [No] and write an explanation."*

Table 8 provides $7$ few-shot demonstrations teaching GPT-4 which task types benefit from Wikipedia retrieval:

| Demo # | Instruction $x$ | GPT-4 Label | Why This Decision Boundary Matters |
|---|---|---|---|
| $1$ | `"Give three tips for staying healthy."` | `[Yes]` | Even for broad advice queries, retrieving reliable health sources improves factual grounding. |
| $2$ | `"Describe a time when you had to make a difficult decision."` | `[No]` | Asks for personal experience; external web documents are irrelevant. |
| $3$ | `"Write a short story in third person narration about a protagonist who has to make an important career decision."` | `[No]` | Creative fiction writing does not require external factual verification. |
| $4$ | `"What is the capital of France?"` | `[Yes]` | Crucial rule: Even though `"Paris"` is a famous head fact that the LM already knows in its parametric memory, it is still a factual question verifiable via web documents, so it is labeled `[Yes]`. |
| $5$ | `"Find the area of a circle given its radius. Radius = 4"` | `[No]` | Pure mathematical calculation ($\pi r^2$); while Wikipedia has the formula, no document will contain the specific pre-computed answer for `Radius = 4`. |
| $6$ | `"Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped"` | `[No]` | Self-contained grammatical rearrangement task; external evidence is unhelpful. |
| $7$ | `"Explain the process of cellular respiration in plants."` | `[Yes]` | Scientific explanation where encyclopedia articles provide accurate technical details. |

### 3.2 Rubric 2: Sentence-Level Three-Way Retrieval Decision (`Retrieve` for $t > 1$, Table 9)

Once initial retrieval has occurred and the response has multiple sentences, deciding whether to retrieve at sentence $t > 1$ requires checking whether the previously retrieved passage $d$ already covers sentence $y_t$.

In Table 9, GPT-4 receives four inputs: `Instruction` ($x$), `Preceding sentences` ($y_{<t}$), `Evidence` ($d$), and `Output` ($y_t$), and chooses among three tokens (Appendix D notes that only a single demonstration is used for this prompt because the demonstration and test input are long):
1. `[Continue to Use Evidence]`: if the information in $y_t$ can be verified solely using the current `Evidence` $d$.
2. `[No Retrieval]`: if $y_t$ does not require any factual verification (for example, a subjective or common-sense transitional sentence).
3. `[Retrieval]`: if additional external information is needed to verify $y_t$ because $d$ does not cover the claims in $y_t$.

In the Table 9 demonstration:
- `Instruction`: `"Explain the use of word embeddings in Natural Language Processing."`
- `Evidence`: Defines what word embeddings are (`"...where words or phrases from the vocabulary are mapped to vectors of real numbers..."`), but says nothing about their downstream applications.
- `Output`: `"Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies."`
- `Rating`: `[Retrieval]`, because verifying this sentence about applications requires retrieving a new passage that actually discusses applications of word embeddings.

### 3.3 Rubric 3: Passage Relevance (`IsRel`, Table 10)

In Table 10, GPT-4 receives `Instruction` ($x$), optional `Preceding sentences` ($y_{<t}$), and `Evidence` ($d$), and judges whether $d$ provides useful information for completing the task (`[Relevant]` vs `[Irrelevant]`). (The opening paragraph of Appendix D mislabels this table's tokens as "three-way" and also describes Table 11 as an `IsRel` table; the table captions themselves are correct.)

Look at the contrast between the `[Relevant]` and `[Irrelevant]` demonstrations in Table 10 (and `chatgpt_relevance.py` lines 48-52):
- Demonstration 1 (`[Relevant]`):
  - `Instruction`: `"Given four answer options, A, B, C, and D, choose the best answer. Input: Earth's rotating causes A: the cycling of AM and PM ..."`
  - `Evidence`: `"Rotation causes the day-night cycle..."`
  - `Rating`: `[Relevant]`, because `"day-night cycle"` directly answers option `A` (`"the cycling of AM and PM"`).
- Demonstration 2 (`[Irrelevant]`, near-miss entity trap):
  - `Instruction`: `"age to run for US House of Representatives"`
  - `Evidence`: `"The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age)..."`
  - `Rating`: `[Irrelevant]`, because the passage gives the minimum age for the U.S. Senate ($30$), not for the U.S. House of Representatives. Teaching the Critic to reject near-miss entity passages prevents the Generator from confusing similar entities.
- Demonstration 3 (in `chatgpt_relevance.py` lines 48-52, `[Irrelevant]` due to preceding context mismatch):
  - `Instruction`: `"Describe a leader or a politician whom you admire."`
  - `Preceding sentences`: Introduces Nelson Mandela as the chosen leader.
  - `Evidence`: A biography paragraph about Barack Obama.
  - `Rating`: `[Irrelevant]`, because although Barack Obama is a political leader (matching the initial instruction), the preceding sentences already committed the essay to Nelson Mandela!

### 3.4 Rubric 4: Three-Way Factual Entailment (`IsSup`, Table 11)

In Table 11, GPT-4 receives `Instruction` ($x$), optional `Preceding sentences` ($y_{<t}$), `Output` ($y_t$), and `Evidence` ($d$), and grades factual support on a three-way entailment scale:
- `[Fully supported]`: Every claim in `Output` is supported by or extracted from `Evidence`.
- `[Partially supported]`: `Output` is supported to some extent, but contains major information not discussed in `Evidence` (for example, if `Output` makes two claims and `Evidence` only verifies one of them).
- `[No support / Contradictory]`: `Output` completely ignores `Evidence`, is unrelated to `Evidence`, or contradicts `Evidence`.

Notice the critical instruction at the bottom of the Table 11 prompt:

> *"Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not."*

Why is this rule essential? Suppose `Output` states a true world fact (such as `"Paris is in France"`), but `Evidence` is an unrelated article about giraffes. Without this strict instruction, GPT-4 might use its own parametric knowledge to say `"Paris is indeed in France, so [Fully supported]!"`. Forcing GPT-4 to ignore external knowledge ensures `IsSup` strictly measures whether `Evidence` entails `Output`.

### 3.5 Rubric 5: Five-Point Perceived Utility (`IsUse`, Table 12)

Finally, in Table 12, GPT-4 receives only `Instruction` ($x$) and the full `Output` ($y$) (without any retrieved `Evidence` $d$), and rates perceived utility from `1` (lowest) to `5` (highest).

Why is `Evidence` $d$ omitted when judging `IsUse`? Because `IsUse` evaluates whether the response addresses the user's request completely, coherently, and helpfully, whereas factual grounding is handled separately by `IsRel` and `IsSup`.

Look at the two demonstrations in Table 12:
1. Temporal mismatch (`Perceived utility: 2`):
   - `Instruction`: `"Who is the current prime minister of the UK as of 2023?"`
   - `Output`: `"Boris Johnson was the prime minister of the UK from 2019 - 2022."`
   - Even though the statement about 2019-2022 is factually true, it does not answer who the prime minister is in 2023, so it receives a low utility score of `2`.
2. Incomplete constraint satisfaction (`Perceived utility: 3`):
   - `Instruction`: `"Given a description of a travel destination, recommend 10 tourist attractions with detailed descriptions of each. The travel destination is Tokyo, Japan."`
   - `Output`: Lists only $7$ attractions by name (`"Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen"`) with zero descriptions.
   - Because it partially answers the topic (Tokyo attractions) but violates both the count constraint ($7 < 10$) and the detail constraint (no descriptions), it receives a middle score of `3`.
