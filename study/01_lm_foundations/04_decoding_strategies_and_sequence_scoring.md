# Decoding Strategies and Sequence Scoring

## 1. Plain-English Intuition: How a Model Generates Text at Test Time

In Part 3, we saw how training works: because the training dataset already contains the complete ground-truth answer $y = (y_1, \dots, y_n)$, we can feed the prompt $x$ and answer $y$ into the transformer all at once in a single parallel pass.

At test time (inference), however, we only have the user's prompt $x$ (such as `["Capital", "of", "France?"]`). The answer $y$ does not exist yet.

Therefore, at test time, the model must generate the response one token at a time in a sequential loop:
1. Step $j = 1$: Pass the prompt `["Capital", "of", "France?"]` into the model. The model outputs logits $z_1 \in \mathbb{R}^V$ and softmax probabilities $p_\theta(y_1 \mid x)$ over all $V$ tokens in the vocabulary. We use a selection rule (called a decoding strategy) to pick one token $\hat{y}_1$ (say `"Paris"`), and append `"Paris"` to the end of our input sequence.
2. Step $j = 2$: Pass the updated sequence `["Capital", "of", "France?", "Paris"]` into the model. The model outputs a new probability distribution $p_\theta(y_2 \mid x, \hat{y}_1)$. We pick the second token $\hat{y}_2$ (say `"is"`), and append `"is"` to the sequence.
3. Repeat step by step until the model picks the end-of-sequence token `</s>` or the number of generated tokens reaches the maximum budget `max_new_tokens`.

This raises two practical questions that appear throughout the Self-RAG codebase:
1. At each step $j$, given the probability distribution over all $V$ tokens, how should we choose the next token $\hat{y}_j$?
2. Once the model generates several candidate sentences of different lengths (for example, one candidate sentence from each retrieved Wikipedia passage), how do we fairly compare the probability of a $4$-token sentence against a $10$-token sentence?

---

## 2. Step-by-Step Build-Up: Token Selection Strategies

### 2.1 Toy Numerical Scenario ($V = 5$ Tokens)

To see how different decoding strategies behave on the exact same model output, suppose our model has just read the prompt `"The weather today is"` and outputs the following raw logits $z_j \in \mathbb{R}^5$ over a $5$-token toy vocabulary $\mathcal{V}_{\text{toy}}$:
- Token $0$ (`"sunny"`): $z_{j,0} = 3.0$
- Token $1$ (`"warm"`): $z_{j,1} = 2.0$
- Token $2$ (`"cloudy"`): $z_{j,2} = 1.0$
- Token $3$ (`"freezing"`): $z_{j,3} = 0.0$
- Token $4$ (`"purple"`): $z_{j,4} = -3.0$ (a nonsensical word in this context)

Applying the standard softmax formula from Part 2 ($\exp(z_{j,v}) / \sum_{v'} \exp(z_{j,v'})$) gives the baseline probabilities:
- $\exp(3.0) \approx 20.0855$
- $\exp(2.0) \approx 7.3891$
- $\exp(1.0) \approx 2.7183$
- $\exp(0.0) = 1.0000$
- $\exp(-3.0) \approx 0.0498$
- Total sum $= 20.0855 + 7.3891 + 2.7183 + 1.0000 + 0.0498 = 31.2427$

Dividing each $\exp(z_{j,v})$ by $31.2427$ gives:
- `"sunny"`: $20.0855 / 31.2427 \approx 0.6429$ ($64.29\%$)
- `"warm"`: $7.3891 / 31.2427 \approx 0.2365$ ($23.65\%$)
- `"cloudy"`: $2.7183 / 31.2427 \approx 0.0870$ ($8.70\%$)
- `"freezing"`: $1.0000 / 31.2427 \approx 0.0320$ ($3.20\%$)
- `"purple"`: $0.0498 / 31.2427 \approx 0.0016$ ($0.16\%$)

### 2.2 Strategy 1: Greedy Decoding ($\arg\max$)

The simplest decoding rule is greedy decoding: at every step $j$, deterministically pick the single token $v$ that has the highest probability (which is also the token with the highest raw logit $z_{j,v}$):

$$\hat{y}_j = \arg\max_{v \in \mathcal{V}} z_{j,v} = \arg\max_{v \in \mathcal{V}} p_\theta(y_j = v \mid x, y_{<j})$$

where:
- $\arg\max_{v \in \mathcal{V}}$ means "return the vocabulary token $v$ that maximizes the expression".
- $\hat{y}_j \in \mathcal{V}$ is the chosen token at step $j$.

In our $5$-token example, `"sunny"` has the maximum logit ($3.0$, probability $64.29\%$), so greedy decoding always selects $\hat{y}_j = \text{``sunny''}$. If you run greedy decoding ten times on the same prompt, it will produce the exact same output all ten times.

### 2.3 Strategy 2: Temperature Scaling ($z_{j,v} / T$)

Instead of always picking the top token deterministically, we can sample a token randomly according to the probabilities $p_\theta(y_j = v \mid x, y_{<j})$. However, we often want a dial to control how concentrated or spread out those probabilities are before we sample.

Temperature scaling divides every raw logit $z_{j,v}$ by a positive number $T > 0$ (called the temperature) before applying the softmax function:

$$p_\theta(y_j = v \mid x, y_{<j}; T) = \frac{\exp\left(\frac{z_{j,v}}{T}\right)}{\sum_{v'=0}^{V-1} \exp\left(\frac{z_{j,v'}}{T}\right)}$$

where:
- $T > 0$ is the scalar temperature parameter.
- Dividing $z_{j,v}$ by $T$ scales the gap between any two logits by $1 / T$: $\frac{z_{j,a}}{T} - \frac{z_{j,b}}{T} = \frac{1}{T}(z_{j,a} - z_{j,b})$.

Let us plug $T = 0.5$ (low temperature), $T = 1.0$ (standard temperature), and $T = 2.0$ (high temperature) into our $5$-token example to see the exact numerical effect:
- When $T = 0.5$, dividing by $0.5$ doubles every logit: $z / 0.5 = [6.0,\; 4.0,\; 2.0,\; 0.0,\; -6.0]$.
- When $T = 2.0$, dividing by $2.0$ halves every logit: $z / 2.0 = [1.5,\; 1.0,\; 0.5,\; 0.0,\; -1.5]$.

| Token $v$ | Raw Logit $z_{j,v}$ | Prob at $T = 0.5$ (Cooler / Sharper) | Prob at $T = 1.0$ (Unmodified) | Prob at $T = 2.0$ (Hotter / Flatter) | Limit as $T \to 0^+$ (Greedy) |
|---|---|---|---|---|---|
| `"sunny"` | $3.0$ | $0.8650$ ($86.50\%$) | $0.6429$ ($64.29\%$) | $0.4450$ ($44.50\%$) | $1.0000$ ($100\%$) |
| `"warm"` | $2.0$ | $0.1171$ ($11.71\%$) | $0.2365$ ($23.65\%$) | $0.2699$ ($26.99\%$) | $0.0000$ ($0\%$) |
| `"cloudy"` | $1.0$ | $0.0158$ ($1.58\%$) | $0.0870$ ($8.70\%$) | $0.1637$ ($16.37\%$) | $0.0000$ ($0\%$) |
| `"freezing"` | $0.0$ | $0.0021$ ($0.21\%$) | $0.0320$ ($3.20\%$) | $0.0993$ ($9.93\%$) | $0.0000$ ($0\%$) |
| `"purple"` | $-3.0$ | $0.000005$ ($0.0005\%$) | $0.0016$ ($0.16\%$) | $0.0222$ ($2.22\%$) | $0.0000$ ($0\%$) |

Look at how changing $T$ reshapes the table:
- Lowering $T$ below $1.0$ ($T = 0.5$) exaggerates the lead of the top token (`"sunny"` jumps from $64.29\%$ to $86.50\%$) and crushes low-scoring tokens.
- As $T$ approaches $0$ ($T \to 0^+$), $100\%$ of the probability concentrates on `"sunny"`, which recovers greedy decoding! Because you cannot divide by $0.0$ directly in floating-point arithmetic, inference libraries like vLLM treat `temperature=0.0` as a flag to execute deterministic greedy decoding ($\arg\max$).
- Raising $T$ above $1.0$ ($T = 2.0$) flattens the probabilities toward a uniform distribution, increasing randomness (and dangerously raising the chance of picking `"purple"` to $2.22\%$).

### 2.4 Strategy 3: Nucleus (`top_p`) Sampling

When we sample with a non-zero temperature (such as `temperature=0.8` in `run_baseline_lm.py`), there is a problem with the long tail of the vocabulary. In a vocabulary of $V = 32000$ tokens, at least $30000$ tokens are completely nonsensical in the current context (like `"purple"` above). Even if each nonsensical token has a tiny probability of $0.0005\%$, all $30000$ nonsensical tokens together add up to a $15\%$ chance of picking a garbage word at every single step!

Nucleus (`top_p`) sampling solves this by cutting off the low-probability tail before sampling, keeping only the "nucleus" of top tokens that account for a chosen fraction $p_{\text{top}} \in (0, 1]$ (such as $p_{\text{top}} = 0.90$ or $0.95$) of the total probability mass.

Here is the step-by-step procedure for nucleus sampling with threshold $p_{\text{top}} = 0.90$ on our $T = 1.0$ probabilities:

1. Step 1 (Sort the vocabulary from highest probability to lowest probability):
   In our toy example, the tokens are already sorted: `"sunny"` ($0.6429$), `"warm"` ($0.2365$), `"cloudy"` ($0.0870$), `"freezing"` ($0.0320$), `"purple"` ($0.0016$).

2. Step 2 (Accumulate probabilities from the top until the running sum reaches or exceeds $p_{\text{top}} = 0.90$):
   - Add Token $0$ (`"sunny"`): running sum $= 0.6429$ (still $< 0.90$, so keep going).
   - Add Token $1$ (`"warm"`): running sum $= 0.6429 + 0.2365 = 0.8794$ (still $< 0.90$, so keep going).
   - Add Token $2$ (`"cloudy"`): running sum $= 0.8794 + 0.0870 = 0.9664$ (now $\ge 0.90$, so include `"cloudy"` and STOP!).

   Formally, we find the smallest cutoff rank $k^*$ whose cumulative probability is at least $p_{\text{top}}$:

   $$k^* = \min \left\{ k \in \{1, \dots, V\} \;\middle|\; \sum_{r=1}^{k} p_\theta\left(v_{(r)} \mid x, y_{<j}; T\right) \ge p_{\text{top}} \right\}$$

   where $v_{(1)}, v_{(2)}, \dots, v_{(V)}$ denotes the vocabulary tokens sorted in descending order of probability. Here $k^* = 3$, so the nucleus set is $\mathcal{V}^{(0.90)} = \{\text{``sunny''}, \text{``warm''}, \text{``cloudy''}\}$.

3. Step 3 (Zero out all tokens outside the nucleus and renormalize the kept tokens to sum to $1.0$):
   We set the probabilities of `"freezing"` and `"purple"` to $0$, and divide the three kept probabilities by their sum ($0.9664$):

   $$\tilde{p}_\theta(y_j = v \mid x, y_{<j}) = \begin{cases} \dfrac{p_\theta(y_j = v \mid x, y_{<j}; T)}{\sum_{v' \in \mathcal{V}^{(p_{\text{top}})}} p_\theta(y_j = v' \mid x, y_{<j}; T)} & \text{if } v \in \mathcal{V}^{(p_{\text{top}})} \\ 0 & \text{otherwise} \end{cases}$$

| Token $v$ | Original Prob ($T=1.0$) | Running Cumulative Sum | Inside Nucleus ($p_{\text{top}} = 0.90$)? | Final Renormalized Prob $\tilde{p}_\theta$ |
|---|---|---|---|---|
| `"sunny"` | $0.6429$ | $0.6429$ | Yes ($1$st) | $0.6429 / 0.9664 \approx 0.6653$ ($66.53\%$) |
| `"warm"` | $0.2365$ | $0.8794$ | Yes ($2$nd) | $0.2365 / 0.9664 \approx 0.2447$ ($24.47\%$) |
| `"cloudy"` | $0.0870$ | $0.9664 \ge 0.90$ (Stop) | Yes ($3$rd) | $0.0870 / 0.9664 \approx 0.0900$ ($9.00\%$) |
| `"freezing"` | $0.0320$ | $0.9984$ | No (Dropped) | $0.0000$ ($0\%$) |
| `"purple"` | $0.0016$ | $1.0000$ | No (Dropped) | $0.0000$ ($0\%$) |

By zeroing out the tail, the model can still choose among reasonable weather words (`"sunny"`, `"warm"`, `"cloudy"`), while having zero chance of picking `"purple"`. Notice that if `top_p=1.0` is passed, $100\%$ of the vocabulary is kept and no truncation occurs.

---

## 3. Step-by-Step Build-Up: Comparing Sequences of Different Lengths (`seq_score`)

### 3.1 Why Raw Cumulative Log-Probability Penalizes Longer Sentences

In Self-RAG, when the model retrieves $K$ passages, it generates one candidate sentence $y^{(k)}$ from each passage $d_k$, and then scores the candidates to pick the best one.

Suppose Passage 1 yields a short candidate sentence $y^{(A)}$ of $n_A = 4$ tokens, and Passage 2 yields a more detailed candidate sentence $y^{(B)}$ of $n_B = 10$ tokens. Suppose the language model is equally confident in every word of both sentences, assigning probability $p = 0.80$ ($\log(0.80) = -0.2231$) to each token in $y^{(A)}$ and $p = 0.80$ ($\log(0.80) = -0.2231$) to each token in $y^{(B)}$.

Look at what happens if we compare their raw sequence log-probabilities (called `cumulative_logprob` in vLLM) from Part 2:
- For the $4$-token sentence $y^{(A)}$:
  $$\log p_\theta\left(y^{(A)} \mid x, d_1\right) = \sum_{j=1}^{4} \log(0.80) = 4 \times (-0.2231) = -0.8924 \quad \left(\text{raw probability } 0.80^4 = 0.4096\right)$$
- For the $10$-token sentence $y^{(B)}$:
  $$\log p_\theta\left(y^{(B)} \mid x, d_2\right) = \sum_{j=1}^{10} \log(0.80) = 10 \times (-0.2231) = -2.2310 \quad \left(\text{raw probability } 0.80^{10} = 0.1074\right)$$

Even though every single token in both sentences had the exact same $80\%$ probability, the $10$-token sentence has a much worse (more negative) cumulative log-probability ($-2.2310 < -0.8924$) simply because adding more negative numbers always produces a more negative sum! If we ranked candidates by raw cumulative log-probability, the decoder would always favor artificially short, clipped answers.

### 3.2 Length Normalization and the Geometric Mean Token Probability

To compare sentences of different lengths fairly, we divide the cumulative log-probability by the number of generated tokens $n = \vert y \vert$ (using $\max(n, 1)$ in code so an empty $0$-token string never causes division by zero). This average log-probability per token is called `seq_score` in the codebase:

$$\text{seq\_score}(y) = \frac{1}{\max(n, 1)} \sum_{j=1}^{n} \log p_\theta(y_j \mid x, d, y_{<j}) \in (-\infty, 0]$$

where:
- $n \in \mathbb{Z}_{\ge 0}$ is the number of tokens in candidate sequence $y$ (`len(pred.outputs[0].token_ids)`).
- $\sum_{j=1}^n \log p_\theta(y_j \mid x, d, y_{<j})$ is the cumulative log-probability (`pred.outputs[0].cumulative_logprob`).
- $\text{seq\_score}(y) \in (-\infty, 0]$ is the mean log-probability per token.

In our example, dividing by length gives:
- Sentence $y^{(A)}$ ($n_A = 4$): $\text{seq\_score}(y^{(A)}) = \frac{-0.8924}{4} = -0.2231$.
- Sentence $y^{(B)}$ ($n_B = 10$): $\text{seq\_score}(y^{(B)}) = \frac{-2.2310}{10} = -0.2231$.

Finally, because $\text{seq\_score}(y)$ is still a negative log-probability in $(-\infty, 0]$, the codebase applies the exponential function $\exp(\text{seq\_score}(y))$ (`np.exp(seq_score)`) to convert it back into a positive probability score in $(0, 1]$:

$$p_{\text{norm}}(y \mid x, d) = \exp\left(\text{seq\_score}(y)\right) = \exp\left(\frac{1}{n} \sum_{j=1}^{n} \log p_\theta(y_j \mid x, d, y_{<j})\right)$$

Using the logarithm rules $\frac{1}{n} \sum_{j=1}^n \log a_j = \log\left(\prod_{j=1}^n a_j\right)^{1/n}$ and $\exp(\log z) = z$, we can see what $\exp(\text{seq\_score}(y))$ represents algebraically:

$$p_{\text{norm}}(y \mid x, d) = \exp\left(\log \left[ \left( \prod_{j=1}^{n} p_\theta(y_j \mid x, d, y_{<j}) \right)^{\frac{1}{n}} \right] \right) = \left( \prod_{j=1}^{n} p_\theta(y_j \mid x, d, y_{<j}) \right)^{\frac{1}{n}} \in (0, 1]$$

Taking the $n$-th root of the product of $n$ numbers is the geometric mean of the per-token probabilities! For both $y^{(A)}$ and $y^{(B)}$, $\exp(-0.2231) = (0.80^4)^{1/4} = (0.80^{10})^{1/10} = 0.80$.

| Candidate | Token Count $n$ | Per-Token Prob $p_j$ | Raw `cumulative_logprob` $\sum \log p_j$ | Raw Sequence Prob $\prod p_j$ | Length-Normalized `seq_score` | Final `np.exp(seq_score)` |
|---|---|---|---|---|---|---|
| Sentence $y^{(A)}$ | $4$ | $0.80$ | $-0.8924$ | $0.4096$ | $-0.8924 / 4 = -0.2231$ | $\exp(-0.2231) = 0.80$ |
| Sentence $y^{(B)}$ | $10$ | $0.80$ | $-2.2310$ | $0.1074$ | $-2.2310 / 10 = -0.2231$ | $\exp(-0.2231) = 0.80$ |

---

## 4. Paper and Codebase Grounding

### 4.1 Nucleus Sampling in Baseline Evaluation (`run_baseline_lm.py`)

In `reference/code/retrieval_lm/run_baseline_lm.py` (lines 47-53), standard open-weights baselines (like vanilla Llama-2 and Alpaca) are evaluated using temperature scaling combined with nucleus sampling:

```python
def call_model(prompts, model, max_new_tokens=50):
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    preds = model.generate(prompts, sampling_params)
    preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds
```

Here, `temperature=0.8` slightly sharpens the logit distribution ($z_{j,v} / 0.8$), and `top_p=0.95` truncates the bottom $5\%$ probability tail before sampling each token up to `max_new_tokens`.

### 4.2 Greedy Decoding and Length-Normalized `seq_score` in Self-RAG (`run_short_form.py` and `run_long_form_static.py`)

In contrast, Self-RAG's own inference scripts (`reference/code/retrieval_lm/run_short_form.py` lines 86-152 and `reference/code/retrieval_lm/run_long_form_static.py` lines 20-35, 86-91) use deterministic greedy decoding (`temperature=0.0, top_p=1.0`) for token generation within each passage branch, and compute the exact length-normalized sequence score `np.exp(seq_score)` derived in Section 3.2. The excerpt below is from `run_short_form.py`:

```python
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params)
...
    for p_idx, pred in enumerate(preds):
        pred_token_ids = pred.outputs[0].token_ids
        pred_text = pred.outputs[0].text
        pred_log_probs = pred.outputs[0].logprobs
        seq_score = pred.outputs[0].cumulative_logprob / \
            max(len(pred.outputs[0].token_ids), 1)
...
        if use_seqscore is True:
            final_score = np.exp(seq_score) + w_rel * relevance_score + \
                w_sup * ground_score + w_use * utility_score
        else:
            final_score = w_rel * relevance_score + \
                w_sup * ground_score + w_use * utility_score
```

Tracing these lines:
1. `SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)` tells vLLM to select the $\arg\max$ token at every step (`temperature=0.0`) without nucleus truncation (`top_p=1.0`), and to return the log-probabilities of the $5000$ most probable tokens at each generated step (`run_long_form_static.py` requests `logprobs=32000` for the same purpose). A reflection token that falls outside the returned set is later treated as having log-probability $-100$.
2. `seq_score = pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1)` divides the sum of token log-probabilities by the number of generated tokens $n$.
3. When `--use_seqscore` is passed on the command line, `np.exp(seq_score)` converts that average log-probability into the geometric mean token probability in $(0, 1]$ and adds it to the weighted critique scores (`w_rel * relevance_score + w_sup * ground_score + w_use * utility_score`) to rank the candidate passages.
