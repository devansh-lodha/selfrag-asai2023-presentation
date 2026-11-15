# Parametric Memory and Hallucination

## 1. Plain-English Intuition: What Is Parametric Memory and Why Do Models Hallucinate?

In Module 01, we traced how a standalone language model generates text: it converts input tokens into vectors via $E_{\text{in}}$, blends context across $L_{\text{layers}}$ causal transformer layers, projects the final hidden vector via $W_{\text{out}}$ into vocabulary logits, and picks the next token.

Now ask a simple question: when a user prompts a standalone language model with `"When did Season 7 of The Walking Dead premiere?"`, where does the model look to find the date `"October 23, 2016"`?

A standalone language model has no search engine, no database, and no copy of Wikipedia stored on disk. The only place any knowledge about the world can exist is inside the learned floating-point numbers of its weight matrices (the embedding table $E_{\text{in}}$, the attention and feed-forward weights across all layers, and the output head $W_{\text{out}}$), which we collectively denote as $\theta$.

Storing knowledge implicitly inside fixed neural network weights $\theta$ is called parametric memory.

Think of parametric memory like a student taking a closed-book exam after reading millions of web pages once:
1. For very famous facts that appeared tens of thousands of times in the reading material (such as `"What is the capital of France?"`), the student's memory is crystal clear and they immediately write `"Paris"`.
2. For rare or obscure facts that appeared only once or twice in the reading material (such as the birth date of a minor historical figure, or the genre of an obscure book), the student's memory is blurry.
3. However, because a language model is trained to always output a high-probability, grammatically fluent next token $y_j$, it does not naturally stop and say `"I do not remember"` when its factual weights are blurry. Instead, it knows the grammatical pattern of the answer (for example, that `"was born on "` must be followed by a month, day, and four-digit year) and fills that pattern with whichever date or name happens to have a slightly higher logit in $\theta$.

This phenomenon, where a language model generates fluent, confident-sounding statements that are factually false, is called hallucination.

---

## 2. Step-by-Step Build-Up: Why Pre-Training Fails on Long-Tail Facts

### 2.1 Pre-Training as Lossy Compression into Fixed Weights $\theta$

During pre-training, a language model reads a massive text corpus $\mathcal{D}_{\text{pretrain}}$ containing trillions of tokens (for example, $2 \times 10^{12}$ tokens for Llama-2) and adjusts a fixed number of parameters $\theta \in \mathbb{R}^P$ (where $P = 7 \times 10^9$ parameters in Llama-2 7B, or $P = 13 \times 10^9$ parameters in Llama-2 13B) to minimize the cross-entropy loss from Module 01 Part 3.

Because trillions of tokens of world knowledge are being squeezed into a fixed set of $7\text{B}$ or $13\text{B}$ floating-point numbers, pre-training is a form of lossy compression.

How does gradient descent decide which facts get preserved in $\theta$ and which facts get blurred or overwritten? Through repetition (corpus frequency).

Let $e$ denote a specific entity (such as a person, place, or book), and let $C(e) \in \mathbb{Z}_{\ge 0}$ denote the number of times a factual statement $(x_e, y_{\text{fact}})$ about entity $e$ appears in the pre-training corpus $\mathcal{D}_{\text{pretrain}}$. Across one pass over the pre-training corpus, the total gradient update $\Delta \theta(e)$ pushing the model's weights to memorize $y_{\text{fact}}$ given $x_e$ is the sum of the gradients over all $C(e)$ occurrences:

$$\Delta \theta(e) = \eta \sum_{i=1}^{C(e)} \nabla_\theta \log p_\theta\left(y_{\text{fact}} \;\middle|\; x_e^{(i)}\right)$$

where:
- $\eta > 0$ is the learning rate.
- $C(e) \in \mathbb{Z}_{\ge 0}$ is the occurrence count of entity $e$'s fact in the pre-training corpus.
- $x_e^{(i)}$ is the surrounding context in the $i$-th document mentioning entity $e$.
- $\nabla_\theta \log p_\theta(y_{\text{fact}} \mid x_e^{(i)}) \in \mathbb{R}^P$ is the gradient vector nudging the weights $\theta$ to increase the probability of the true fact $y_{\text{fact}}$.

Compare what this sum looks like for two different types of entities:
- High-frequency ("head") entities: An entity like `"France"` or `"Albert Einstein"` has a massive count $C(e_{\text{head}}) \gg 100000$. The weights $\theta$ receive hundreds of thousands of consistent gradient pushes in the exact same direction, carving a deep, reliable association in $\theta$.
- Low-frequency ("long-tail") entities: Most real-world entities (specific scientific claims, niche books, local public figures, or rare medical conditions) appear in the long tail of the frequency distribution, where $C(e_{\text{tail}})$ may be $1$, $5$, or $0$. Those few tiny gradient steps are easily washed out by billions of competing gradient updates from other documents sharing the same weights $\theta$.

### 2.2 Toy Numerical Comparison: Syntax Confidence vs Factual Uncertainty

To see how a language model hallucinates at the token level, let us compare the logits $z_j$ and softmax probabilities $p_\theta(y_j \mid x)$ when the model is asked for the birth year of a famous head entity versus a rare long-tail entity.

Suppose the vocabulary contains four candidate year tokens (`"1879"`, `"1955"`, `"1958"`, `"1959"`) and one non-year token (`"apple"`).

Case A (Famous head entity): Prompt $x_{\text{head}} =$ `"Albert Einstein was born in "`
Because `"Albert Einstein"` and `"1879"` co-occurred thousands of times during pre-training, the hidden state $h_t$ aligns strongly with `"1879"` in $W_{\text{out}}$.

Case B (Rare long-tail entity): Prompt $x_{\text{tail}} =$ `"Rory Tapner was born in "` (where the true birth year is $1959$, as in Appendix Table 7 of the paper).
Because `"was born in "` is a familiar grammatical phrase, the model strongly suppresses non-year words like `"apple"` ($z = -5.0$). It knows a four-digit year token must come next, so it gives all plausible 20th-century years positive logits around $2.0$. However, because `"Rory Tapner"` appeared rarely during pre-training, the model has no strong signal separating `"1959"` from `"1955"` or `"1958"`:

| Candidate Token $v$ | Case A Logit $z_{j,v}$ (Head Entity: Einstein) | Case A Prob $p_\theta(v \mid x_{\text{head}})$ | Case B Logit $z_{j,v}$ (Tail Entity: Tapner) | Case B Prob $p_\theta(v \mid x_{\text{tail}})$ |
|---|---|---|---|---|
| `"1879"` (True for Einstein) | $6.0$ | $0.9746$ ($97.46\%$) | $-1.0$ | $0.0156$ ($1.56\%$) |
| `"1955"` | $1.0$ | $0.0066$ ($0.66\%$) | $2.1$ | $0.3458$ ($34.58\%$) |
| `"1958"` | $1.5$ | $0.0108$ ($1.08\%$) | $2.2$ | $0.3822$ ($38.22\%$) |
| `"1959"` (True for Tapner) | $1.2$ | $0.0080$ ($0.80\%$) | $1.8$ | $0.2562$ ($25.62\%$) |
| `"apple"` (Ungrammatical) | $-5.0$ | $0.00002$ ($0.002\%$) | $-5.0$ | $0.0003$ ($0.03\%$) |

Look at Case B in the table above:
1. Grammatical fluency is $99.97\%$ intact: the model assigns $0.03\%$ probability to ungrammatical words like `"apple"` and $99.97\%$ probability to valid four-digit years.
2. Factual accuracy fails: because `"1958"` happens to have a slightly higher logit ($2.2$) than the true year `"1959"` ($1.8$), greedy decoding ($\arg\max$) deterministically picks `"1958"`, and plain sampling at $T = 1$ picks a wrong year about $74\%$ of the time ($1 - 0.2562 - 0.0003$)!
3. Once the model outputs the wrong year `"1958"`, that wrong token is appended to the prefix for step $j + 1$, and the model continues generating a smooth, confident-sounding sentence built around a false date.

---

## 3. Four Empirical Failure Modes of Pure Parametric Models in Self-RAG

Section 1, Section 4.1, and Table 2 of the Self-RAG paper measure how pure parametric models (models without retrieval) perform across diverse tasks, demonstrating four concrete failure modes.

### 3.1 Failure on Long-Tail Open-Domain QA (PopQA)

In Section 4.1, the authors evaluate models on the long-tail subset of PopQA, which consists of $1,399$ factual questions about rare entities whose Wikipedia pages receive fewer than $100$ monthly page views (for example, `"What genre is A Question and Answer Guide to Astronomy?"`).

Look at the PopQA accuracy of standalone parametric models (without retrieval) from Table 2 of the paper:

| Model Family | Model Size | Training Type | PopQA Long-Tail Accuracy (%) |
|---|---|---|---|
| Llama-2 | $7\text{B}$ | Pre-trained base LM | $14.7\%$ |
| Llama-2 | $13\text{B}$ | Pre-trained base LM | $14.7\%$ |
| Llama-2-chat | $13\text{B}$ | Instruction-tuned + RLHF (proprietary data) | $20.0\%$ |
| Alpaca | $7\text{B}$ | Instruction-tuned on Llama-2 7B | $23.6\%$ |
| Alpaca | $13\text{B}$ | Instruction-tuned on Llama-2 13B | $24.4\%$ |
| ChatGPT | Proprietary | Instruction-tuned + RLHF (`gpt-3.5-turbo`) | $29.3\%$ |

Notice two striking facts in these numbers:
1. Nearly doubling the parameter count of the base model from `Llama-2 7B` to `Llama-2 13B` yields $14.7\%$ in both cases ($0.0\%$ gain). Simply having more weights does not recover long-tail facts that were rarely seen during pre-training.
2. Even ChatGPT answers more than $70\%$ of these rare-entity questions incorrectly when relying solely on its parametric memory.

### 3.2 Confident Fabrication in Long-Form Generation (Biography FactScore)

In short-form QA, a model only has to output one entity or date. In long-form generation, such as writing a multi-sentence biography (`Tell me a bio about [Person Name]`), the model must generate dozens of specific factual claims (birth date, birthplace, education, career roles, awards) in a single paragraph.

The paper evaluates biographies using FactScore (`Bio (FS)` in Table 2), which breaks each generated biography into individual atomic facts and checks what percentage of those facts are supported by the person's Wikipedia article:
- `Llama-2 7B` (no retrieval): $44.5\%$ FactScore
- `Alpaca 7B` (no retrieval): $45.8\%$ FactScore
- `Alpaca 13B` (no retrieval): $50.2\%$ FactScore
- `Llama-2 13B` (no retrieval): $53.4\%$ FactScore
- `Llama-2-chat 13B` (no retrieval): $55.9\%$ FactScore
- `ChatGPT` (no retrieval): $71.8\%$ FactScore

For open-weights $7\text{B}$ and $13\text{B}$ models without retrieval, roughly half of every biography they write is fabricated.

### 3.3 Instruction Tuning Teaches Style, Not Missing Facts

Compare `Llama-2 13B` (the raw pre-trained base model) against `Alpaca 13B` (the instruction-tuned version of `Llama-2 13B`) in Table 2:
- On fluency (ASQA `MAUVE`), instruction tuning causes a massive jump from $16.0$ (`Llama-2 13B`) to $70.6$ (`Alpaca 13B`), because instruction tuning teaches the model to write polished, well-structured paragraphs that follow user instructions.
- However, on biography factuality (`Bio (FS)`), `Alpaca 13B` ($50.2\%$) actually scores *lower* than the raw `Llama-2 13B` base model ($53.4\%$).

Why does instruction tuning slightly hurt factual precision on biographies? The paper does not analyze this. A plausible explanation is that supervised instruction tuning rewards complete, detailed, helpful-sounding responses (`Perceived Utility`), which encourages longer biographical paragraphs even when the parametric weights $\theta$ do not contain the facts for that person.

### 3.4 Complete Absence of Source Attribution and Verifiability

Finally, look at the last two columns of Table 2 for ASQA long-form question answering: citation precision (`pre`) and citation recall (`rec`).

For every single baseline without retrieval (`Llama-2 7B/13B`, `Alpaca 7B/13B`, `Llama-2-chat 13B`, `ChatGPT`), citation precision and recall are listed as `–` (not applicable). Because a standalone parametric model synthesizes its output tokens from distributed matrix multiplications across billions of weights, it cannot attach verifiable document citations (`[1]`, `[2]`) to its sentences. A user reading a 5-sentence answer from a parametric model has no way to tell whether sentence 3 is a true fact memorized during pre-training or a plausible-sounding hallucination.
