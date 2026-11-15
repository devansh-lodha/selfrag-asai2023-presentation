# Causal Attention, Logits, and Softmax

## 1. Plain-English Intuition: How Token Vectors Share Context to Predict the Next Word

In Part 1, we saw how an input sequence of $t$ tokens $u = (u_1, u_2, \dots, u_t)$ is converted by the embedding lookup table $E_{\text{in}} \in \mathbb{R}^{V \times d}$ into a sequence of $t$ vectors:

$$H^{(0)} = \begin{bmatrix} e_1 \\ e_2 \\ \vdots \\ e_t \end{bmatrix} \in \mathbb{R}^{t \times d}$$

However, each row vector $e_k \in \mathbb{R}^d$ in $H^{(0)}$ is still completely isolated from its neighbors. Suppose our input prompt is:

`"France's capital is"` $\quad \longrightarrow \quad (u_1 = \text{``France's''},\; u_2 = \text{``capital''},\; u_3 = \text{``is''})$

Before passing through the transformer layers, the vector $e_3$ at the third position (`"is"`) only contains the static lookup row for the word `"is"`. It has no information that the first word was `"France's"` or that the second word was `"capital"`. If we tried to predict the fourth word (`"Paris"`) using only the isolated vector $e_3$ for `"is"`, the model could only guess words that commonly follow `"is"` in general (like `"a"` or `"the"`).

To predict `"Paris"`, the vector at the final position $t = 3$ (`"is"`) must look back at the earlier positions $1$ (`"France's"`) and $2$ (`"capital"`), decide how relevant each earlier word is, and pull their information into position $3$.

The mechanism that lets each token position look back at earlier token positions and blend their information together is called causal self-attention. Once the transformer layers finish blending context into the final position's vector $h_t \in \mathbb{R}^d$, the model projects $h_t$ into a score (called a logit) for every token in the vocabulary $\mathcal{V}$, and turns those scores into probabilities using the softmax function.

---

## 2. Step-by-Step Build-Up: Causal Self-Attention

### 2.1 The Intuition of Query ($q$), Key ($k$), and Value ($v$)

How does token position $t$ decide how much information to pull from each earlier position $i \in \{1, 2, \dots, t\}$?

Self-attention models this like searching a filing cabinet where each position $i$ has a folder:
- Query vector $q_t \in \mathbb{R}^{d_h}$: Produced at the current position $t$, representing what information position $t$ is looking for.
- Key vector $k_i \in \mathbb{R}^{d_h}$: Written on the label of folder $i$, representing what kind of information position $i$ contains.
- Value vector $v_i \in \mathbb{R}^{d_h}$: Stored inside folder $i$, representing the actual semantic content that position $i$ will pass along if selected.

All three vectors ($q_i, k_i, v_i$) at every position $i$ are computed by multiplying the token's current hidden vector $h_i \in \mathbb{R}^d$ by three learned weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{d_h \times d}$:

$$q_i = W_Q h_i \in \mathbb{R}^{d_h}, \quad k_i = W_K h_i \in \mathbb{R}^{d_h}, \quad v_i = W_V h_i \in \mathbb{R}^{d_h}$$

where:
- $h_i \in \mathbb{R}^d$ is the input hidden vector at token position $i$.
- $d \in \mathbb{Z}^+$ is the full model hidden dimension ($d = 4096$ in Llama-2 7B).
- $d_h \in \mathbb{Z}^+$ is the dimension of a single attention head (in Llama-2 7B, the $4096$ dimensions are split across $H = 32$ parallel attention heads, so each head has $d_h = 4096 / 32 = 128$).

To measure how well the Query $q_t$ at position $t$ matches the Key $k_i$ at an earlier position $i$, we compute their dot product (inner product), which multiplies matching coordinates and adds them up:

$$\langle q_t, k_i \rangle = \sum_{c=1}^{d_h} q_{t,c} \, k_{i,c} \in \mathbb{R}$$

If $q_t$ and $k_i$ point in similar directions, $\langle q_t, k_i \rangle$ is a large positive number (strong match). If they point in opposite directions, $\langle q_t, k_i \rangle$ is negative (weak match).

### 2.2 Why Attention Must Be Causal (Left-to-Right Only)

When generating text one token at a time, position $t$ only has access to tokens at positions $i \le t$ (itself and the tokens to its left). Position $t$ must never be allowed to look at future positions $i > t$ to its right, because at test time those future tokens do not exist yet, and during training peeking at position $t + 1$ would let the model cheat by reading the exact next token it is supposed to predict.

Therefore, self-attention in autoregressive language models is causal: position $t$ only computes attention weights over positions $i \in \{1, 2, \dots, t\}$. Any future position $i > t$ is blocked (masked out by setting its similarity score to $-\infty$ so that its weight becomes $0$).

### 2.3 Worked 3-Token Toy Numerical Example of Causal Self-Attention

Let us trace a concrete 3-token sequence $(u_1 = \text{``France's''},\; u_2 = \text{``capital''},\; u_3 = \text{``is''})$ with head dimension $d_h = 2$. We are at position $t = 3$ (`"is"`) and want to compute its context-blended output vector $o_3 \in \mathbb{R}^2$.

Suppose that after multiplying by $W_Q, W_K, W_V$, our 2-dimensional vectors are:
- Query at position $t = 3$ (`"is"`):
  $$q_3 = [2.0,\; 1.0]$$
- Keys and Values at all three positions $i \in \{1, 2, 3\}$:

| Position $i$ | Token | Key Vector $k_i \in \mathbb{R}^2$ | Value Vector $v_i \in \mathbb{R}^2$ |
|---|---|---|---|
| $i = 1$ | `"France's"` | $[1.0,\; 1.0]$ | $[10.0,\; 0.0]$ |
| $i = 2$ | `"capital"` | $[1.0,\; 0.0]$ | $[0.0,\; 10.0]$ |
| $i = 3$ | `"is"` | $[0.0,\; -1.0]$ | $[1.0,\; 1.0]$ |

Causal self-attention computes the output vector $o_3 \in \mathbb{R}^2$ in four arithmetic steps:

1. Step 1 (Compute dot-product similarity between $q_3$ and each key $k_i$ for $i \le 3$):
   - Match with $i = 1$ (`"France's"`):
     $$\langle q_3, k_1 \rangle = (2.0)(1.0) + (1.0)(1.0) = 3.0$$
   - Match with $i = 2$ (`"capital"`):
     $$\langle q_3, k_2 \rangle = (2.0)(1.0) + (1.0)(0.0) = 2.0$$
   - Match with $i = 3$ (`"is"`):
     $$\langle q_3, k_3 \rangle = (2.0)(0.0) + (1.0)(-1.0) = -1.0$$

2. Step 2 (Scale by $\sqrt{d_h}$):
   When $d_h$ is large (such as $d_h = 128$), summing $128$ products can make the dot products very large in magnitude, pushing the exponential function in Step 3 into regions where gradients vanish. Dividing each dot product by $\sqrt{d_h}$ keeps the variance of the scores near $1$. Here $d_h = 2$, so $\sqrt{d_h} = \sqrt{2} \approx 1.414$:
   - $s_{3,1} = \frac{3.0}{\sqrt{2}} \approx 2.121$
   - $s_{3,2} = \frac{2.0}{\sqrt{2}} \approx 1.414$
   - $s_{3,3} = \frac{-1.0}{\sqrt{2}} \approx -0.707$

3. Step 3 (Convert scores into positive weights $\alpha_{3,i}$ that sum to $1$):
   We exponentiate each scaled score ($\exp(s_{3,i}) > 0$) and divide by their sum:
   - $\exp(s_{3,1}) = \exp(2.121) \approx 8.339$
   - $\exp(s_{3,2}) = \exp(1.414) \approx 4.112$
   - $\exp(s_{3,3}) = \exp(-0.707) \approx 0.493$
   - Total denominator sum $= 8.339 + 4.112 + 0.493 = 12.944$
   - Attention weights $\alpha_{3,i}$:
     $$\alpha_{3,1} = \frac{8.339}{12.944} \approx 0.644, \quad \alpha_{3,2} = \frac{4.112}{12.944} \approx 0.318, \quad \alpha_{3,3} = \frac{0.493}{12.944} \approx 0.038$$
   Notice that $\alpha_{3,1} > 0, \alpha_{3,2} > 0, \alpha_{3,3} > 0$ and $0.644 + 0.318 + 0.038 = 1.000$. Position $3$ places $64.4\%$ of its attention on `"France's"`, $31.8\%$ on `"capital"`, and only $3.8\%$ on `"is"`.

4. Step 4 (Take the weighted average of the Value vectors $v_i$):
   $$o_3 = \alpha_{3,1} v_1 + \alpha_{3,2} v_2 + \alpha_{3,3} v_3$$
   $$o_3 = 0.644 \begin{bmatrix} 10.0 \\ 0.0 \end{bmatrix} + 0.318 \begin{bmatrix} 0.0 \\ 10.0 \end{bmatrix} + 0.038 \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 6.44 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 3.18 \end{bmatrix} + \begin{bmatrix} 0.038 \\ 0.038 \end{bmatrix} = \begin{bmatrix} 6.478 \\ 3.218 \end{bmatrix}$$

Look at the resulting vector $o_3 = [6.478,\; 3.218]$: its first coordinate ($6.478$) carries strong signal from $v_1$ (`"France's"`) and its second coordinate ($3.218$) carries strong signal from $v_2$ (`"capital"`). By attending to the past tokens, position $3$ now holds a combined representation of `"France's capital is"`.

### 2.4 General Equations for Causal Self-Attention Across Layers

For any token position $t \in \{1, \dots, L\}$, causal self-attention computes the attention weight $\alpha_{t,i} \in (0, 1)$ assigned to earlier position $i \in \{1, \dots, t\}$ as:

$$\alpha_{t,i} = \frac{\exp\left(\frac{\langle q_t, k_i \rangle}{\sqrt{d_h}}\right)}{\sum_{r=1}^{t} \exp\left(\frac{\langle q_t, k_r \rangle}{\sqrt{d_h}}\right)}, \quad \text{for } i \in \{1, \dots, t\}$$

and blends the value vectors $v_1, \dots, v_t$ into the head output $o_t \in \mathbb{R}^{d_h}$:

$$o_t = \sum_{i=1}^{t} \alpha_{t,i} \, v_i \in \mathbb{R}^{d_h}$$

where:
- $t \in \{1, \dots, L\}$ is the current query position index.
- $i \in \{1, \dots, t\}$ indexes all valid non-future positions up to $t$.
- $\alpha_{t,i} \in (0, 1)$ is the fraction of attention that position $t$ pays to position $i$, satisfying $\sum_{i=1}^t \alpha_{t,i} = 1$.

In a full transformer block $\ell \in \{1, \dots, L_{\text{layers}}\}$ ($L_{\text{layers}} = 32$ in Llama-2 7B, $L_{\text{layers}} = 40$ in Llama-2 13B), the outputs of all $H$ attention heads are concatenated, projected back to dimension $d$, added to the input via a residual connection, and passed through a feed-forward network (MLP):

$$H^{(\ell)} = \text{TransformerBlock}_\ell\left(H^{(\ell-1)}\right) \in \mathbb{R}^{t \times d}, \quad \text{for } \ell = 1, 2, \dots, L_{\text{layers}}$$

After the final layer $L_{\text{layers}}$, we extract the last row of $H^{(L_{\text{layers}})}$, which is the final hidden vector at position $t$:

$$h_t = H^{(L_{\text{layers}})}[t, :] \in \mathbb{R}^d$$

This single vector $h_t \in \mathbb{R}^d$ summarizes the entire sequence $(u_1, u_2, \dots, u_t)$ up to position $t$.

---

## 3. From the Final Hidden State $h_t$ to Vocabulary Logits $z_j$

### 3.1 Intuition: Scoring Every Token in the Vocabulary

Suppose our sequence up to position $t$ consists of an $m$-token prompt $x = (x_1, \dots, x_m)$ followed by $j - 1$ already-generated response tokens $y_{<j} = (y_1, \dots, y_{j-1})$, so $t = m + j - 1$. We now want to predict the $j$-th response token $y_j \in \mathcal{V}$.

We have the context summary vector $h_t \in \mathbb{R}^d$ (for example, $4096$ numbers), and we need to produce a score for every one of the $V = 32000$ tokens in our vocabulary $\mathcal{V}$.

To do this, the model uses a second lookup table of size $V \times d$ at the very top of the network, called the output projection matrix or language modeling head (LM head), denoted $W_{\text{out}} \in \mathbb{R}^{V \times d}$.

Each row $W_{\text{out}}[v, :] \in \mathbb{R}^d$ corresponds to one vocabulary token $v \in \{0, 1, \dots, V - 1\}$. To score how well candidate token $v$ fits after the context summarized by $h_t$, the model computes the dot product between row $W_{\text{out}}[v, :]$ and $h_t$:

$$z_{j,v} = \langle W_{\text{out}}[v, :], h_t \rangle = \sum_{c=1}^{d} W_{\text{out}}[v, c] \, h_{t,c} \in (-\infty, \infty)$$

Computing this dot product for all $V$ rows simultaneously is the matrix-vector multiplication:

$$z_j = W_{\text{out}} h_t \in \mathbb{R}^V$$

where:
- $W_{\text{out}} \in \mathbb{R}^{V \times d}$ is the output projection weight matrix (`model.get_output_embeddings().weight` or `model.lm_head.weight` in PyTorch).
- $h_t \in \mathbb{R}^d$ is the final hidden vector at position $t = m + j - 1$.
- $z_j = [z_{j,0}, z_{j,1}, \dots, z_{j,V-1}]^\top \in \mathbb{R}^V$ is the vector of raw, unnormalized scores over the vocabulary at response step $j$, called the logits.
- $z_{j,v} \in \mathbb{R}$ is the raw logit score for vocabulary token $v$: a large positive $z_{j,v}$ means $h_t$ aligns strongly with row $W_{\text{out}}[v, :]$, whereas a negative $z_{j,v}$ means they point in opposite directions.

### 3.2 Toy Numerical Example of Logit Projection ($V = 4, d = 2$)

Suppose we have a tiny 4-token vocabulary $\mathcal{V}_{\text{toy}} = \{0: \text{``London''},\; 1: \text{``Paris''},\; 2: \text{``Rome''},\; 3: \text{``</s>''}\}$ and hidden dimension $d = 2$.

After reading `"France's capital is"`, suppose the final hidden vector is:

$$h_t = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} \in \mathbb{R}^2$$

and the output matrix $W_{\text{out}} \in \mathbb{R}^{4 \times 2}$ has rows:
- Row $0$ (`"London"`): $[-1.0,\; 0.0]$
- Row $1$ (`"Paris"`): $[1.0,\; 1.0]$
- Row $2$ (`"Rome"`): $[-1.0,\; 1.0]$
- Row $3$ (`"</s>"`): $[0.0,\; 0.0]$

Multiplying $z_j = W_{\text{out}} h_t$ computes one dot product per token:
- $z_{j,0}$ (`"London"`): $(-1.0)(1.0) + (0.0)(2.0) = -1.0$
- $z_{j,1}$ (`"Paris"`): $(1.0)(1.0) + (1.0)(2.0) = 3.0$
- $z_{j,2}$ (`"Rome"`): $(-1.0)(1.0) + (1.0)(2.0) = 1.0$
- $z_{j,3}$ (`"</s>"`): $(0.0)(1.0) + (0.0)(2.0) = 0.0$

Thus, the raw logit vector is:

$$z_j = \begin{bmatrix} -1.0 \\ 3.0 \\ 1.0 \\ 0.0 \end{bmatrix} \in \mathbb{R}^4$$

---

## 4. Turning Logits into Probabilities and Log-Probabilities (Softmax)

### 4.1 Why Raw Logits Are Not Probabilities

Look at our logit vector $z_j = [-1.0,\; 3.0,\; 1.0,\; 0.0]^\top$. We can see that `"Paris"` has the highest score ($3.0$), `"Rome"` is second ($1.0$), `"</s>"` is third ($0.0$), and `"London"` is lowest ($-1.0$).

However, raw logits cannot be used directly as probabilities for two reasons:
1. A logit can be any negative or positive real number (such as $-1.0$ or $3.0$), whereas a valid probability must lie between $0$ and $1$.
2. The sum of probabilities over all mutually exclusive outcomes must equal $1.0$, whereas $-1.0 + 3.0 + 1.0 + 0.0 = 3.0 \neq 1.0$.

### 4.2 The Softmax Transformation

The softmax function converts any vector of raw real numbers $z_j \in \mathbb{R}^V$ into a valid probability distribution $p_\theta(y_j = v \mid x, y_{<j}) \in (0, 1)$ in two steps:
1. Apply the exponential function $\exp(x) = e^x$ to each logit $z_{j,v}$. Because $e^x > 0$ for every real number $x \in (-\infty, \infty)$, all numbers become strictly positive while preserving their exact ranking ($z_{j,a} > z_{j,b} \implies \exp(z_{j,a}) > \exp(z_{j,b})$).
2. Divide each positive number $\exp(z_{j,v})$ by the sum of all $V$ positive numbers $\sum_{v'=0}^{V-1} \exp(z_{j,v'})$ so the fractions add up to $1.0$:

$$p_\theta(y_j = v \mid x, y_{<j}) = \frac{\exp(z_{j,v})}{\sum_{v'=0}^{V-1} \exp(z_{j,v'})}$$

where:
- $z_{j,v} \in \mathbb{R}$ is the raw logit for token $v$ at step $j$.
- $\exp(z_{j,v}) \in (0, \infty)$ is the unnormalized positive weight for token $v$.
- $\sum_{v'=0}^{V-1} \exp(z_{j,v'}) \in (0, \infty)$ is the normalization denominator summing across all $V$ tokens in $\mathcal{V}$.
- $p_\theta(y_j = v \mid x, y_{<j}) \in (0, 1)$ is the resulting conditional probability of generating token $v$ at step $j$.

### 4.3 Worked Numerical Example: Softmax, Max-Shift Stability, and Log-Probabilities

Let us plug our 4-token logit vector $z_j = [-1.0,\; 3.0,\; 1.0,\; 0.0]^\top$ into the softmax formula:
- $\exp(-1.0) \approx 0.3679$
- $\exp(3.0) \approx 20.0855$
- $\exp(1.0) \approx 2.7183$
- $\exp(0.0) = 1.0000$
- Denominator sum $= 0.3679 + 20.0855 + 2.7183 + 1.0000 = 24.1717$

Dividing each $\exp(z_{j,v})$ by $24.1717$ gives the probabilities $p_\theta(y_j = v \mid x, y_{<j})$, and taking the natural logarithm $\log(\cdot)$ of each probability gives the log-probability $\log p_\theta(y_j = v \mid x, y_{<j})$:

| Token ID $v$ | Token String | Raw Logit $z_{j,v}$ | $\exp(z_{j,v})$ | Probability $p_\theta(y_j = v \mid x, y_{<j})$ | Log-Probability $\log p_\theta(y_j = v \mid x, y_{<j})$ |
|---|---|---|---|---|---|
| $0$ | `"London"` | $-1.0$ | $0.3679$ | $0.3679 / 24.1717 \approx 0.0152$ ($1.52\%$) | $\log(0.0152) \approx -4.1852$ |
| $1$ | `"Paris"` | $3.0$ | $20.0855$ | $20.0855 / 24.1717 \approx 0.8310$ ($83.10\%$) | $\log(0.8310) \approx -0.1852$ |
| $2$ | `"Rome"` | $1.0$ | $2.7183$ | $2.7183 / 24.1717 \approx 0.1125$ ($11.25\%$) | $\log(0.1125) \approx -2.1852$ |
| $3$ | `"</s>"` | $0.0$ | $1.0000$ | $1.0000 / 24.1717 \approx 0.0414$ ($4.14\%$) | $\log(0.0414) \approx -3.1852$ |
| Total | | | $24.1717$ | $1.0000$ ($100\%$) | |

Look closely at the rightmost column ($\log p_\theta$) in the table above, because vLLM and the Self-RAG inference scripts (`run_short_form.py` and `run_long_form_static.py`) store and return log-probabilities (`pred_log_probs`) rather than raw probabilities:
- Because every probability $p$ is between $0$ and $1$, its logarithm $\log(p)$ is always negative (or $0$ if $p = 1.0$).
- A high probability like $0.8310$ (`"Paris"`) corresponds to a log-probability close to zero ($-0.1852$).
- A tiny probability like $0.0152$ (`"London"`) corresponds to a more negative log-probability ($-4.1852$). If a token has a minuscule probability like $e^{-100} \approx 3.7 \times 10^{-44}$, its log-probability is $-100.0$.
- To convert any log-probability back into a normal probability in $[0, 1]$, we simply apply $\exp(\cdot)$:

$$\exp\left(\log p_\theta(y_j = v \mid x, y_{<j})\right) = p_\theta(y_j = v \mid x, y_{<j})$$

For example, $\exp(-0.1852) = 0.8310$.

### 4.4 The Max-Subtraction Trick for Numerical Stability

In 16-bit floating-point arithmetic (`float16`), the largest representable number is $65504$. Since $\exp(12) \approx 162755 > 65504$, if a logit $z_{j,v}$ ever exceeds $11.1$, computing $\exp(z_{j,v})$ directly overflows to `inf`, and dividing `inf / inf` produces `NaN`.

To prevent overflow, we subtract the maximum logit $z_j^{\max} = \max_{v'} z_{j,v'}$ from every logit before exponentiating. By the exponent rule $\exp(a - b) = \exp(a)\exp(-b)$, multiplying both numerator and denominator by $\exp(-z_j^{\max})$ cancels out and leaves the result mathematically identical:

$$p_\theta(y_j = v \mid x, y_{<j}) = \frac{\exp(z_{j,v})\exp(-z_j^{\max})}{\left(\sum_{v'=0}^{V-1} \exp(z_{j,v'})\right)\exp(-z_j^{\max})} = \frac{\exp(z_{j,v} - z_j^{\max})}{\sum_{v'=0}^{V-1} \exp(z_{j,v'} - z_j^{\max})}$$

In our 4-token example, $z_j^{\max} = 3.0$. Subtracting $3.0$ from $z_j = [-1.0,\; 3.0,\; 1.0,\; 0.0]^\top$ gives shifted logits $[-4.0,\; 0.0,\; -2.0,\; -3.0]^\top$. Now the largest shifted logit is $0.0$ ($\exp(0.0) = 1.0$), all other shifted logits are negative ($\exp(\text{negative}) < 1.0$), overflow is impossible, and $\frac{\exp(0.0)}{\exp(-4.0) + \exp(0.0) + \exp(-2.0) + \exp(-3.0)} = \frac{1.0}{0.0183 + 1.0 + 0.1353 + 0.0498} = \frac{1.0}{1.2034} \approx 0.8310$, the exact same probability for `"Paris"`.

---

## 5. The Chain Rule: Probability of a Complete Multi-Token Response

### 5.1 Step-by-Step Multiplication Across Token Positions

So far, we have computed the probability of generating a single token $y_j$ at step $j$. How do we compute the probability $p_\theta(y \mid x)$ that the model generates an entire multi-token response $y = (y_1, y_2, \dots, y_n)$ given prompt $x$?

By the chain rule of conditional probability, the joint probability of a sequence of events is the product of the probability of each event conditioned on all preceding events:

$$p_\theta(y \mid x) = p_\theta(y_1, y_2, \dots, y_n \mid x) = p_\theta(y_1 \mid x) \times p_\theta(y_2 \mid x, y_1) \times p_\theta(y_3 \mid x, y_1, y_2) \times \dots \times p_\theta(y_n \mid x, y_{<n})$$

Writing this compactly using product notation ($\prod$):

$$p_\theta(y \mid x) = \prod_{j=1}^{n} p_\theta(y_j \mid x, y_{<j})$$

where:
- $n = \vert y \vert \in \mathbb{Z}^+$ is the number of tokens in the response $y$.
- $y_{<j} = (y_1, \dots, y_{j-1})$ is the prefix of response tokens before step $j$, with $y_{<1} = \emptyset$.
- $p_\theta(y_j \mid x, y_{<j}) \in (0, 1)$ is the softmax probability of token $y_j$ at step $j$.

### 5.2 Why We Add Log-Probabilities Instead of Multiplying Probabilities

Suppose a generated response has $n = 3$ tokens: $y = (y_1 = \text{``Paris''},\; y_2 = \text{``is''},\; y_3 = \text{``</s>''})$, with step probabilities:
- Step $j = 1$: $p_\theta(\text{``Paris''} \mid x) = 0.80$ $\quad (\log p = -0.2231)$
- Step $j = 2$: $p_\theta(\text{``is''} \mid x, \text{``Paris''}) = 0.50$ $\quad (\log p = -0.6931)$
- Step $j = 3$: $p_\theta(\text{``</s>''} \mid x, \text{``Paris''}, \text{``is''}) = 0.90$ $\quad (\log p = -0.1054)$

For $n = 3$ tokens, multiplying $0.80 \times 0.50 \times 0.90 = 0.36$ is easy. However, for a long-form response of $n = 200$ tokens where the average token probability is $0.50$, multiplying $200$ numbers equal to $0.50$ yields $0.50^{200} \approx 6.22 \times 10^{-61}$, which rounds down (underflows) to `0.0` in 16-bit and 32-bit floating-point arithmetic (64-bit floats underflow a little later, after roughly $1075$ such factors).

Taking the natural logarithm of both sides of the chain rule converts the product of probabilities into a sum of log-probabilities (using $\log(a \cdot b) = \log a + \log b$):

$$\log p_\theta(y \mid x) = \log \left( \prod_{j=1}^{n} p_\theta(y_j \mid x, y_{<j}) \right) = \sum_{j=1}^{n} \log p_\theta(y_j \mid x, y_{<j})$$

In our 3-token example, summing the log-probabilities gives:

$$\log p_\theta(y \mid x) = -0.2231 + (-0.6931) + (-0.1054) = -1.0216$$

Adding 200 numbers around $-0.6931$ simply equals $-138.62$, which never underflows! And if we ever need the probability back, exponentiating the sum gives $\exp(-1.0216) = 0.36$.

---

## 6. Paper and Codebase Grounding

### 6.1 Query, Key, and Value Projections in `llama_flash_attn_monkey_patch.py`

In `reference/code/retrieval_lm/llama_flash_attn_monkey_patch.py` (lines 28-44), we can see the exact PyTorch code that computes the Query, Key, and Value vectors $q_i, k_i, v_i$ from `hidden_states` ($H^{(\ell-1)}$) inside each Llama-2 attention layer:

```python
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
```

Tracing the tensor dimensions:
- `hidden_states` enters with shape `[bsz, q_len, hidden_size]` ($[B, L, d]$, where $d = 4096$ in Llama-2 7B).
- `self.q_proj`, `self.k_proj`, and `self.v_proj` are linear layers ($W_Q, W_K, W_V$) that project $[B, L, 4096]$ to $[B, L, 4096]$.
- `.view(bsz, q_len, self.num_heads, self.head_dim)` reshapes the $4096$ channels into `num_heads = 32` heads of size `head_dim = 128` ($32 \times 128 = 4096$), producing shape $[B, L, 32, 128]$.
- `.transpose(1, 2)` swaps the sequence length and head dimensions to `[bsz, num_heads, q_len, head_dim]` ($[B, 32, L, 128]$) so that all 32 attention heads compute causal self-attention in parallel.

### 6.2 Extracting Log-Probabilities and Converting to Probabilities in `run_short_form.py`

During inference in `reference/code/retrieval_lm/run_short_form.py` (lines 89-104), vLLM generates a response segment and returns:
- `pred.outputs[0].token_ids`: the list of generated token IDs $(y_1, y_2, \dots, y_n)$.
- `pred.outputs[0].logprobs`: a list of length $n$, where `pred_log_probs[j]` is a dictionary mapping token IDs $v$ to their log-probabilities $\log p_\theta(y_{j+1} = v \mid x, y_{\le j}) \in (-\infty, 0]$ at generation step $j$. Only the most probable token IDs are included: as many as the `logprobs` argument of `SamplingParams` requests (`logprobs=5000` for the passage-conditioned call in `run_short_form.py`), plus the token actually generated.
- `pred.outputs[0].cumulative_logprob`: the sum of log-probabilities $\sum_{j=1}^n \log p_\theta(y_j \mid x, y_{<j})$ derived in Section 5.2.

Look at how lines 90-104 of `run_short_form.py` read the log-probabilities of the `[Relevant]` and `[Irrelevant]` reflection tokens at the very first generated position (`pred_log_probs[0]`) and convert them into probabilities:

```python
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))
```

Every detail of this snippet now connects directly to the math in Section 4.3:
- `pred_log_probs[0]` contains the log-probability distribution over the vocabulary at the first generated token position ($j = 1$).
- If token `id` is present in `pred_log_probs[0]`, `prob` is its log-probability $\log p_\theta(y_1 = \text{id} \mid \cdot) \le 0$ (a number that is zero or negative). If `id` was not among the top returned `logprobs`, it defaults to `-100` (a log-probability of $-100$, corresponding to $\exp(-100) \approx 3.7 \times 10^{-44} \approx 0$).
- `np.exp(float(prob))` applies the exponential function $\exp(\log p) = p$, converting the negative log-probability back into a standard probability in $(0, 1]$ stored in `relevance_score_dict[p_idx][tok]`.
