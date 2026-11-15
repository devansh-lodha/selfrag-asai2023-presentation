# The Self-RAG Two-Model Training Overview

## 1. Plain-English Intuition: How Do We Create Training Data with Reflection Tokens?

In Part 1, we saw how a trained Self-RAG Generator model $\mathcal{M}$ works at test time: it writes a sentence, outputs `[Retrieval]`, reads an inserted `<paragraph>...</paragraph>` block, outputs `[Relevant]`, writes the next sentence, outputs `[Fully supported]`, and finishes with `[Utility:5]`.

Now consider a fundamental question: how do we train a base Llama-2 model to output those reflection tokens in the right places?

Recall from Module 01 Part 3 that supervised fine-tuning requires training examples where the target output string already contains the exact sequence of tokens we want the model to predict. However, existing instruction-tuning and question-answering datasets (such as Stanford Alpaca or Natural Questions) only contain plain text inputs $x$ and plain text outputs $y$:
- Plain Input $x$: `"How did US states get their names?"`
- Plain Output $y$: `"11 of 50 states' names come from persons. For instance, Louisiana was named in honor of King Louis XIV of France and Georgia was named after King George II."`

There are no `[Retrieval]`, `[Relevant]`, `[Fully supported]`, or `[Partially supported]` tokens anywhere in existing datasets. To train Self-RAG on $145,619$ instruction-output pairs, we need a way to automatically evaluate every sentence of every training example, retrieve candidate Wikipedia passages, and insert the appropriate reflection tokens into the text.

Why not ask human annotators or GPT-4 to label all $145,619$ examples directly?
1. Human annotation is prohibitively expensive: Splitting $145,619$ responses into sentences and having humans read $5$ to $10$ retrieved Wikipedia passages per sentence would require millions of manual judgments.
2. Calling GPT-4 on the entire training corpus is too costly and non-reproducible: Evaluating relevance and support across $10$ retrieved passages for every sentence of $145,619$ examples would require several million GPT-4 API calls, and Section 3.2.1 notes that depending on proprietary LMs raises API costs and diminishes reproducibility (their behavior can change over time).

Self-RAG solves this with an offline two-model distillation pipeline (Algorithm 2 in Appendix A.2):
1. First, we prompt GPT-4 on a small sampled subset of examples ($\sim 4\text{k}$ to $20\text{k}$ examples per token type) to collect reflection token labels $\mathcal{D}_{\text{critic}}$.
2. Second, we train an in-house $7\text{B}$ language model, called the Critic ($\mathcal{C}$), on $\mathcal{D}_{\text{critic}}$ so that $\mathcal{C}$ learns to predict reflection tokens just like GPT-4.
3. Third, we run our trained local Critic $\mathcal{C}$ and Retriever $\mathcal{R}$ offline across all $145,619$ plain training examples to insert retrieved passages and reflection tokens, creating the Generator training corpus $\mathcal{D}_{\text{gen}}$.
4. Fourth, we train the final Generator language model ($\mathcal{M}$, $7\text{B}$ or $13\text{B}$) on $\mathcal{D}_{\text{gen}}$ using standard next-token prediction loss. Once $\mathcal{M}$ is trained, the Critic $\mathcal{C}$ is no longer needed at inference time, because $\mathcal{M}$ has learned to generate both task responses and reflection tokens by itself.

---

## 2. Step-by-Step Walkthrough of the Four Training Stages (Algorithm 2 & Figure 2)

Let us trace each of the four offline training stages in Algorithm 2 and Figure 2 of the paper.

### 2.1 Stage 1: Collecting Critic Supervision from GPT-4 ($\mathcal{D}_{\text{critic}}$)

Starting from the raw seed corpus $\mathcal{D} = \{X, Y\}$ of plain input-output pairs $(x, y)$, we randomly sample a small subset $\{X_{\text{sample}}, Y_{\text{sample}}\} \sim \{X, Y\}$.

Because each of the four reflection token groups (`Retrieve`, `IsRel`, `IsSup`, `IsUse`) evaluates a different property and takes different inputs (Table 1), we prompt GPT-4 with a type-specific instruction $I_{\text{type}}$ and a few-shot list of demonstrations to predict the appropriate reflection token $r$:

$$r \sim p_{\text{GPT-4}}\left(r \;\middle|\; I_{\text{type}}, x, y, d\right)$$

where:
- $I_{\text{type}}$ is the natural-language rubric and few-shot demonstrations for that specific reflection token group (for example, `"Given an instruction, make a judgment on whether finding some external documents from the web helps to generate a better response."`).
- $x$ is the task instruction or question.
- $y$ is the target output (or current sentence segment $y_t$ and preceding sentences $y_{<t}$).
- $d$ is the retrieved Wikipedia passage (included when collecting `IsRel`, `IsSup`, and multi-sentence `Retrieve`, and omitted when collecting initial `Retrieve` or `IsUse`).
- $r \in \mathcal{V}_{\text{ref}}$ is the discrete reflection token predicted by GPT-4.

Collecting $3,831$ to $19,317$ GPT-4 annotations per group yields roughly $46,923$ total supervised examples, which are combined to form the Critic training dataset $\mathcal{D}_{\text{critic}}$.

### 2.2 Stage 2: Training the Critic Model $\mathcal{C}$ (Equation 1)

Next, we initialize the Critic $\mathcal{C}$ with a pre-trained Llama-2 7B model, expand its vocabulary with the special reflection tokens, and train it on $\mathcal{D}_{\text{critic}}$ using standard conditional language modeling (Equation 1 in Section 3.2.1):

$$\max_{\mathcal{C}} \mathbb{E}_{((x, y), r) \sim \mathcal{D}_{\text{critic}}} \log p_{\mathcal{C}}(r \mid x, y)$$

where:
- $\mathcal{C}$ denotes the trainable parameters of the Critic language model (Llama-2 7B).
- $\mathcal{D}_{\text{critic}}$ is the dataset of GPT-4-labeled examples collected in Stage 1.
- $(x, y)$ represents the complete conditioning input fed to the Critic (which includes the rubric instruction, task input $x$, task output $y$, and retrieved passage $d$ when applicable).
- $r \in \mathcal{V}_{\text{ref}}$ is the target reflection token to be predicted.
- $\log p_{\mathcal{C}}(r \mid x, y)$ is the conditional log-probability assigned by $\mathcal{C}$ to the target reflection token $r$ given the input $(x, y)$.

Using the prompt loss masking mechanism from Module 01 Part 3 (`label[:source_len] = -100`), all tokens in the conditioning input $(x, y)$ are masked to `-100`, so the Critic $\mathcal{C}$ only computes cross-entropy loss on predicting the reflection token $r$ (and `</s>`). After 3 epochs of training, the Llama-2 7B Critic achieves $93.8\%$ agreement with GPT-4 on `Retrieve`, $93.5\%$ on `IsSup`, $80.2\%$ on `IsRel`, and $73.5\%$ on `IsUse` (Appendix Figure 5).

### 2.3 Stage 3: Constructing the Generator Training Corpus $\mathcal{D}_{\text{gen}}$ Offline (Figure 2)

Once the Critic $\mathcal{C}$ is trained, we use $\mathcal{C}$ together with the Contriever retriever $\mathcal{R}$ to annotate the full $145,619$-example seed corpus $\mathcal{D} = \{X, Y\}$ offline, producing the augmented Generator dataset $\mathcal{D}_{\text{gen}}$.

Figure 2 in Section 3.2 of the paper shows two concrete examples of how a plain $(x, y)$ pair is transformed into an augmented target sequence in $\mathcal{D}_{\text{gen}}$:

#### Example A: A Prompt That Does Not Require Retrieval (Figure 2 Left)
- Original Input $x$:
  `"Write an essay of your best summer vacation"`
- Original Output $y$ ($2$ sentences):
  - Sentence 1 ($y_1$): `"My best summer vacation was a magical escape to the coastal town of Santorini."`
  - Sentence 2 ($y_2$): `"The azure waters, charming white-washed building are unforgettable."`

Here is how the Critic $\mathcal{C}$ processes $(x, y)$ offline:
1. Given $x$, Critic $\mathcal{C}$ predicts that external retrieval is not helpful: `[No Retrieval]`.
2. For the second sentence $y_2$, no retrieval is needed either: `[No Retrieval]`.
3. Given $(x, y)$, Critic $\mathcal{C}$ evaluates the overall perceived utility of the complete response and predicts: `[Utility:5]` (written as `Util: 5` in Figure 2).

Inserting these reflection tokens into $y$ produces the final augmented output stored in $\mathcal{D}_{\text{gen}}$:

`"[No Retrieval] My best summer vacation was a magical escape to the coastal town of Santorini. [No Retrieval] The azure waters, charming white-washed building are unforgettable experience. [Utility:5]"`

#### Example B: A Prompt That Requires Retrieval (Figure 2 Right)
- Original Input $x$:
  `"How did US states get their names?"`
- Original Output $y$ ($2$ sentences; the figure's `Output:` line prints `"1 of 50 states names come from persons."`, a typo for the `"11 of 50"` that appears in its augmented output):
  - Sentence 1 ($y_1$): `"11 of 50 states' names come from person."`
  - Sentence 2 ($y_2$): `"For instance, Louisiana was named in honor of King Louis XIV of France and Georgia was named after King George II."`

Here is how the Critic $\mathcal{C}$ and Retriever $\mathcal{R}$ process $(x, y)$ sentence by sentence:
1. Sentence 1 ($y_1$):
   - Critic $\mathcal{C}$ evaluates whether retrieval helps for $x$: predicts `[Retrieval]` (`Retrieve`).
   - Retriever $\mathcal{R}$ searches Wikipedia using $(x, y_1)$ and retrieves candidate passages, including Passage 1: `<p>Of the fifty states, eleven are named after an individual person</p>.`
   - Critic $\mathcal{C}$ evaluates whether Passage 1 is relevant to $x$: predicts `[Relevant]`.
   - Critic $\mathcal{C}$ evaluates whether Sentence 1 (`"11 of 50 states' names come from person."`) is entailed by Passage 1: predicts `[Fully supported]` (`Supported`).
2. Sentence 2 ($y_2$):
   - Critic $\mathcal{C}$ evaluates whether additional retrieval is needed to verify Sentence 2 (since Passage 1 does not mention Louisiana or Georgia): predicts `[Retrieval]` (`Retrieve`).
   - Retriever $\mathcal{R}$ searches Wikipedia using $(x, y_2)$ and retrieves Passage 2: `<p>LOUISIANA: Named in honor of Louis XIV of France.</p>.`
   - Critic $\mathcal{C}$ evaluates whether Passage 2 is relevant: predicts `[Relevant]`.
   - Critic $\mathcal{C}$ evaluates whether Sentence 2 (`"For instance, Louisiana was named after King Louis XIV, and Georgia was named after King George II."`) is entailed by Passage 2: because Passage 2 verifies the first half of the sentence (Louisiana / Louis XIV) but does not mention the second half (Georgia / George II), Critic $\mathcal{C}$ predicts `[Partially supported]` (`Partially`)!
3. End of response:
   - Critic $\mathcal{C}$ evaluates the overall utility of the full response given $x$: predicts `[Utility:5]` (`Util: 5`).

Putting these pieces together produces the augmented output stored in $\mathcal{D}_{\text{gen}}$:

`"[Retrieval] <p>Of the fifty states, eleven are named after an individual person</p>. [Relevant] 11 of 50 states' names come from person. [Fully supported] [Retrieval] <p>LOUISIANA: Named in honor of Louis XIV of France.</p>. [Relevant] For instance, Louisiana was named after King Louis XIV, and Georgia was named after King George II. [Partially supported] [Utility:5]"`

### 2.4 Stage 4: Training the Generator Model $\mathcal{M}$ (Equation 2)

Finally, we initialize the Generator $\mathcal{M}$ from a pre-trained Llama-2 7B or 13B model, expand its vocabulary $\mathcal{V}$ with the special tokens, and train $\mathcal{M}$ on $\mathcal{D}_{\text{gen}}$ using the standard next-token prediction objective (Equation 2 in Section 3.2.2):

$$\max_{\mathcal{M}} \mathbb{E}_{(x, y, r) \sim \mathcal{D}_{\text{gen}}} \log p_{\mathcal{M}}(y, r \mid x)$$

where:
- $\mathcal{M}$ denotes the trainable parameters of the Generator language model (Llama-2 7B or 13B).
- $\mathcal{D}_{\text{gen}}$ is the augmented training corpus created in Stage 3.
- $x$ is the task instruction or prompt.
- $(y, r)$ denotes the interleaved sequence of output text segments $y = (y_1, \dots, y_T)$ and reflection tokens $r$ in the augmented target string.
- $\log p_{\mathcal{M}}(y, r \mid x)$ is the sum of next-token log-probabilities across all output text tokens in $y$ and all reflection tokens in $r$.

### 2.5 Comparing the Critic Objective (Equation 1) vs the Generator Objective (Equation 2)

Comparing Equation 1 and Equation 2 side by side clarifies the exact role of both models and why loss masking is applied differently:

| Feature | Critic Model $\mathcal{C}$ (Equation 1) | Generator Model $\mathcal{M}$ (Equation 2) |
|---|---|---|
| Mathematical Objective | $\max_{\mathcal{C}} \mathbb{E} \log p_{\mathcal{C}}(r \mid x, y)$ | $\max_{\mathcal{M}} \mathbb{E} \log p_{\mathcal{M}}(y, r \mid x)$ |
| What Is Fed as Input (Masked with `-100` in Loss) | Instruction $x$, passage $d$, and target sentence $y$ | Instruction $x$ AND any retrieved passage span `<p>d</p>` (`<paragraph>d</paragraph>`) |
| What Is Predicted (Active in Cross-Entropy Loss) | Only the single reflection token $r \in \mathcal{V}_{\text{ref}}$ | Both the response sentences $y_t$ AND all reflection tokens $r$ (plus the `<paragraph>` and `</paragraph>` boundary tags) |
| Used at Test Time? | No (used only offline during Stage 3 to build $\mathcal{D}_{\text{gen}}$) | Yes ($\mathcal{M}$ generates both answers and critiques at test time) |

Why do we mask out the retrieved passage tokens inside `<p>...</p>` (`<paragraph>...</paragraph>`) with `-100` when training the Generator $\mathcal{M}$?
- At test time, $\mathcal{M}$ does not generate the Wikipedia passage $d$ from memory; when $\mathcal{M}$ outputs `[Retrieval]`, the external retriever $\mathcal{R}$ fetches $d$ from Wikipedia and pastes `<paragraph>d</paragraph>` into the context.
- By setting the labels of all tokens inside $d$ to `-100` during training (as we saw in `preprocess` in Module 01 Part 3), the Generator $\mathcal{M}$ is never asked to predict the words of the Wikipedia passage $d$. Instead, $\mathcal{M}$ reads $d$ through causal self-attention and is trained strictly to predict `[Relevant]` (or `[Irrelevant]`), the grounded answer sentence $y_t$, and `[Fully supported]` (or `[Partially supported]` / `[No support / Contradictory]`) after $d$.
