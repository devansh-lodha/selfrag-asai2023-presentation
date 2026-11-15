# Presentation Plan

Complete plan for the Self-RAG deck, with the full content of every slide and the speaker notes. The talk has 44 slides, and a short appendix of 14 slides follows it. The Typst deck follows this plan.

## How this plan was decided

- Keep an example only when it explains better than the abstract version. A real before and after opens the method section, and the critic's full rule is one table.
- Drop a walkthrough when a formula or diagram already explains the idea. The score arithmetic and the three-sentence labeling example are in the appendix, and the GPT-4 prompt examples are left out.
- Put each important number next to the claim it supports. The authors' check of GPT-4's labels sits on the GPT-4 slide, the critic's accuracy on the critic slide, each ASQA metric's meaning on the ASQA chart, and the training-size finding on the Llama2-FT slide. This is how the deck keeps the detail with fewer slides.
- Each slide still makes one point. It can hold several facts only when they serve that point.
- Inference is explained before it is compared with training, and its settings are explained where the threshold and the weights first appear.
- Sections 5 and 6 share one divider, with three analysis slides and one impact slide. They keep only points that trace to the paper or to our reading of the code.
- The appendix holds only walkthroughs and the full charts behind numbers the talk already states.

## Left out of the deck

- The abstract table of token forms, since the real examples show the same thing.
- The GPT-4 prompt examples.
- The illustrative table for choosing a passage, since slide 18 states the rule.
- A table of the critic's answers on real sentences, since slide 14 shows that example with its tokens and slide 18 states the full rule.
- The summary of results, since every result slide states its own finding.
- A separate slide on why the judgments are tokens, since slide 11 states it.
- A separate slide on test-time control, since slides 21 and 24 give those uses where the threshold and the weights appear.

## Design rules

The deck keeps its current style, with grayscale, Lato for text, and Lete Sans Math for math. Titles are plain. A result slide states its finding in a full-width box right under the exhibit. Step numbers sit above boxes, and text inside a box stays clear of its top edge. Every slide keeps at least about 1.2 cm of clear space under the title and has no stranded words.

## Time

The script is 2361 words, about 18:10 at 130 words per minute. Slide changes and seven handovers add about 50 seconds, for about 19:00 in total.

| Speaker | Part | Slides | Words | Time |
| --- | --- | --- | --- | --- |
| 1 | Problem | 1 to 7 | 300 | 2:18 |
| 2 | Motivation and key contribution | 8 to 12 | 269 | 2:04 |
| 3 | Training overview, GPT-4, and the critic | 13 to 17 | 261 | 2:00 |
| 4 | Labeling, the generator, and the inference overview | 18 to 20 | 272 | 2:06 |
| 5 | Inference steps | 21 to 25 | 300 | 2:18 |
| 6 | Training vs inference, setup, and first results | 26 to 31 | 329 | 2:32 |
| 7 | ChatGPT, Llama2-FT, and ablations | 32 to 36 | 316 | 2:26 |
| 8 | Grounding, limitations, impact, and close | 37 to 44 | 314 | 2:25 |

Speakers are numbered for now, and names still need to be assigned.

## Rubric coverage

| Rubric item | Talk slides |
| --- | --- |
| Problem and motivation | 3 to 8 |
| Core contribution | 10 to 12 |
| Architecture and algorithms | 10, 15, 20 to 25 |
| Training procedure | 14 to 19 |
| Objectives and equations | 17, 19, 21, 23, 24 |
| Design decisions | 12, 15, 19, 21, 24 |
| Setup, datasets, metrics, baselines | 28, 29, 31 |
| Main results | 30 to 33 |
| Ablations | 34, 35 |
| Further analysis | 36, 37 |
| Strengths and assumptions | 39 |
| Limitations and improvements | 40 |
| Are the experiments sufficient? | 41 |
| Impact | 42 |
| Team Contribution | 43 |

## Talk slides and speaker notes

Brackets such as [Retrieve = Yes] mark a token chip, and italics mark a shaded retrieved passage. "Second line" means smaller gray text under a name. The time in each heading is for the speaker note, which is quoted below the slide content.


### Speaker 1: Problem (slides 1 to 7, 2:18)

#### 1. Title slide (0:09)

Corners: "CS 613: Natural Language Processing" at top left, "Indian Institute of Technology Gandhinagar" at top right, "Fall 2026" at bottom left, and the page number at bottom right.

Center, stacked: "Self-RAG", then "Learning to Retrieve, Generate, and Critique through Self-Reflection", then "Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi", then "ICLR 2024", then "Group: Mixture of Experts".

> Hi everyone. We are Mixture of Experts, and today we will walk you through Self-RAG, a paper from ICLR 2024.

#### 2. Divider: What problem does the paper solve? (0:02)

Section slide 1 of 5.

> Let's start with the problem.

#### 3. Language models get rare facts wrong (0:25)

Horizontal bars.

| Model | Accuracy (%) |
| --- | --- |
| Llama2 7B | 14.7 |
| Llama2 13B | 14.7 |
| Alpaca 13B | 24.4 |
| ChatGPT | 29.3 |

Caption: Accuracy (%) on PopQA, 1,399 questions about rare entities with under 100 Wikipedia views a month. 7B and 13B are sizes in billions of parameters, and Alpaca is Llama2 tuned on instructions.

Takeaway box: Going from 7B to 13B changes nothing, and even ChatGPT is mostly wrong.

> Language models get facts wrong, and rare topics are the worst case. PopQA asks about entities with under a hundred Wikipedia views a month. Llama2 gets about fifteen percent right, going from seven to thirteen billion parameters changes nothing, and even ChatGPT is under a third. A bigger model alone does not fix this.

#### 4. Retrieval-Augmented Generation (RAG) (0:21)

Diagram, left to right: Input $x$, then Retriever $\mathcal{R}$ (searches a large collection), then $d_1, \dots, d_K$ (the top $K$ passages), then Language model (reads $x$ and all $K$ passages), then Output $y$.

Text: RAG searches for passages related to the input $x$ and adds the top $K$ to the prompt. The language model then writes the output $y$ from them.

Takeaway box: On PopQA, RAG lifts Alpaca 13B from 24.4% to 46.1% and Llama2-chat from 20.0% to 51.8%.

> The standard fix is retrieval-augmented generation, or RAG. A retriever searches a large collection for the input, the top K passages go into the prompt, and the language model writes from them. It clearly helps. On PopQA, Alpaca 13B goes from twenty-four to forty-six percent.

#### 5. Three problems with standard RAG (0:29)

| Problem | Example |
| --- | --- |
| It retrieves even when no facts are needed | An essay about a summer vacation still fetches passages |
| It can retrieve off-topic passages | For "How did US states get their names?" one passage is "In Texas, Emma is a popular baby name." |
| Nothing makes the output agree with the passages | It says California was named after Christopher Columbus, while its passage says a fictional island in a Spanish book |

> But standard RAG has three problems. It retrieves even when no facts are needed, so a vacation essay still fetches passages. It can retrieve off-topic passages, like a line about baby names for a question about state names. And nothing makes the output agree with the passages. Here the model credits Christopher Columbus for California, while its own passage says a fictional island.

#### 6. Retrieval alone is not enough (0:23)

Two charts side by side.

Left chart, "PubHealth accuracy (%)", paired bars:

| Model | No retrieval | With retrieval |
| --- | --- | --- |
| ChatGPT | 70.1 | 54.7 |
| Alpaca 13B | 55.5 | 51.1 |
| Alpaca 7B | 49.8 | 40.2 |
| Llama2 7B | 34.2 | 30.0 |

Right chart, "ASQA citation precision (%)", bars:

| Model | Citation precision |
| --- | --- |
| Alpaca 13B | 2.0 |
| Llama2 13B | 2.3 |
| Llama2 7B | 2.9 |
| Alpaca 7B | 5.5 |
| Llama2-chat 13B | 19.8 |
| ChatGPT | 65.1 |

Caption: PubHealth checks health claims, and these four of the six baselines lose accuracy with retrieval. ASQA needs long answers with citations. Citation precision is the share of citations that support their statement.

Takeaway box: Retrieval can lower accuracy, and open models rarely cite supporting evidence.

> The numbers show these problems are real. On PubHealth, which checks health claims, adding retrieval lowers accuracy for four of the six baselines. And on ASQA, where answers must cite passages, open models rarely cite anything that supports their claim. The best reaches about twenty percent, while ChatGPT reaches sixty-five.

#### 7. Existing work fixes only one piece (0:29)

One table. The Area cell spans its three rows, and each approach has a smaller gray second line.

| Area | Approach | What it leaves open |
| --- | --- | --- |
| Retrieval | Always-retrieve training (second line: SAIL) | It retrieves a fixed number of passages for every input |
| Retrieval | Adaptive retrieval (second line: Toolformer and others) | It often loses speed, robustness to bad passages, and attribution |
| Retrieval | Passage filters (second line: using a separate model) | An extra model runs at test time, and the output goes unchecked |
| Critique | Reinforcement learning (second line: from human feedback, or RLHF) | Reward models run during training, and changing behavior often means retraining |
| Critique | Control tokens (second line: added at the start of generation) | The tokens never assess the output |
| Critique | Repeated prompting (second line: CoVE and others) | Generating and revising several times makes answering slow |

> People have tried to fix parts of this. On the retrieval side, SAIL always retrieves a fixed number of passages, adaptive methods often lose speed or attribution, and passage filters add another model at test time. On the critique side, RLHF needs reward models during training, control tokens never assess the output, and repeated prompting makes answering slow. Each fixes only one piece.


### Speaker 2: Motivation and key contribution (slides 8 to 12, 2:04)

#### 8. Motivation (0:20)

Three rows with no heading row.

| | |
| --- | --- |
| Why it matters | Factual errors can spread misinformation and give incorrect, dangerous advice |
| The goal | One language model that decides when to retrieve, judges what it finds, and checks what it writes |
| Also needed | Low training overhead and adjustable at test time |

> This matters because factual errors can spread misinformation and give dangerous advice. So the goal is one language model that decides when to retrieve, judges what it finds, and checks what it writes, while adding little training overhead and staying adjustable at test time.

#### 9. Divider: What is the key contribution? (0:03)

Section slide 2 of 5.

> So what is the key idea?

#### 10. How Self-RAG works (0:34)

Diagram: $x$ enters a dashed box. Inside the box, the shaded box "Need evidence for the next sentence?" has a "yes" arrow to "Retrieve $K$ passages", then to "Write one candidate sentence per passage", then to the shaded box "Critique candidates and keep the best". A "no" arrow goes down to "Write the sentence without retrieval". The dashed box carries the label "Repeated until the answer ends". An arrow leaves the dashed box for the shaded box "Rate the answer", then $y$.

Text: Self-Reflective RAG, or Self-RAG, writes one sentence at a time. In each shaded box it writes a reflection token that records its decision.

> Self-RAG writes one sentence at a time. Before each sentence, it asks whether it needs evidence. If not, it just writes the sentence. If yes, it retrieves K passages, writes one candidate from each, critiques them, and keeps the best. This repeats until the answer is finished, and then the model rates the whole answer. Each shaded box is a decision, which the model records by writing a special token, called a reflection token.

#### 11. Four types of reflection tokens (0:36)

The left column groups the tokens, with "Critique" spanning three rows.

| | Token | The question it answers | Values |
| --- | --- | --- | --- |
| Retrieval | Retrieve | Does the next sentence need retrieval? | Yes, No, Continue |
| Critique | IsRel | Is this passage useful for the input? | Relevant, Irrelevant |
| Critique | IsSup | Does this passage support the sentence? | Fully, Partially, No support |
| Critique | IsUse | Is the answer useful, whether or not it is true? | 5 (highest) to 1 |

Caption: Continue means keep using the passage already retrieved.

Takeaway box: A token is predicted like a word, so ordinary training teaches it, and its probability can serve as a score.

> There are four types of these tokens. Retrieve controls retrieval, and Continue means keep using the passage you already have. The other three are critiques. IsRel asks if a passage is useful, IsSup asks if it supports the sentence, and IsUse rates the whole answer from one to five. The key insight is that a token is predicted just like a word, so ordinary training can teach it, and its probability can later serve as a score.

#### 12. Standard RAG vs Self-RAG (0:31)

| | Standard RAG | Self-RAG | Why it helps |
| --- | --- | --- | --- |
| Retrieval | Always, once, before writing | On demand, for each sentence | Tasks that need no facts keep the model's creativity |
| Passages | All $K$ together in one prompt | One candidate per passage | A bad passage affects only its own candidate |
| Quality check | None | Relevance, support, usefulness | The best candidate is chosen with no extra model |

Takeaway box: Each design choice answers one of the three problems with standard RAG.

> Each row here answers one of the three problems. Standard RAG always retrieves once before writing, while Self-RAG retrieves on demand, so tasks that need no facts keep their creativity. Standard RAG puts all passages in one prompt, while Self-RAG writes one candidate per passage, so a bad passage spoils only its own candidate. And Self-RAG checks relevance, support, and usefulness, with no extra model at test time.


### Speaker 3: Training overview, GPT-4, and the critic (slides 13 to 17, 2:00)

#### 13. Divider: How does the method work? (0:03)

Section slide 3 of 5.

> Now let's see how this is built.

#### 14. The training data we need (0:31)

Three rows with no heading row. Tokens appear as chips, written here in brackets. Retrieved passages are shaded, written here in italics.

- Input $x$: How did US states get their names?
- Before: 11 of 50 states' names come from persons. For instance, Louisiana was named after King Louis XIV, and Georgia was named after King George II.
- After, first sentence: [Retrieve = Yes] *Of the fifty states, eleven are named after an individual person.* [IsRel = Relevant] 11 of 50 states' names come from persons. [IsSup = Fully]
- After, second sentence: [Retrieve = Yes] *Louisiana: named in honor of Louis XIV of France.* [IsRel = Relevant] For instance, Louisiana was named after King Louis XIV, and Georgia was named after King George II. [IsSup = Partially] [IsUse = 5]

Caption: Shaded text is a retrieved passage. The second one says nothing about Georgia, so support is only partial.

Takeaway box: Training has to turn plain pairs into text like this, then teach a model to write it.

> Let's start from where training has to end up. At the top is a plain training pair. Below is the same output after labeling. Each sentence now has a Retrieve token, a retrieved passage, a relevance token, and a support token, and one usefulness token closes the answer. If a model reads enough text like this, it learns to write it. So how do we produce this text?

#### 15. Training phase (0:29)

Four boxes, left to right, with step numbers above them. Each box has a bold name and one line of detail.

| Step | Name | Detail |
| --- | --- | --- |
| 1 | GPT-4 | Labels a sample of the data |
| 2 | Critic $\mathcal{C}$ | A Llama2 7B model that learns GPT-4's labels |
| 3 | Labeling | The critic and retriever label all 145,619 pairs |
| 4 | Generator $\mathcal{M}$ | A Llama2 7B or 13B model that learns from the result |

Caption: Labeling every sentence by hand is too expensive, and using GPT-4 on everything raises API costs and hurts reproducibility. The 145,619 input-output pairs start as plain text.

Takeaway box: No existing data contains these tokens, so GPT-4 and the critic create them.

> Training has four steps. No dataset contains these tokens, hand labeling is too expensive, and using GPT-4 on everything raises costs and hurts reproducibility. So GPT-4 labels only a sample, and a Llama2 model called the critic learns to copy it. The critic and the retriever then label all 145,619 pairs, and the generator, the model we actually use, learns from the result.

#### 16. GPT-4 labels a sample of the data (0:26)

Footer: Training step 1 of 4.

| Token | GPT-4 reads | Labels collected | Authors agreed |
| --- | --- | --- | --- |
| Retrieve | Input alone, or input, earlier text, passage, and one sentence | 12,594 | 95% |
| IsRel | Input and passage | 19,317 | 95% |
| IsSup | Input, passage, and output | 11,181 | 90% |
| IsUse | Input and output | 3,831 | 80% |

Caption: Each token type has its own prompt, with an instruction, a few demonstrations, and one sampled example.

Takeaway box: The authors checked 20 labels per type and agreed with at least 90%, except 80% on IsUse.

> Step one. Each token type has its own prompt, with an instruction, a few demonstrations, and one sampled example. The table shows what GPT-4 reads for each type and how many labels it produced. The authors then read twenty labels per type by hand and agreed with at least ninety percent, except eighty percent on usefulness.

#### 17. The critic learns from GPT-4's labels (0:31)

Footer: Training step 2 of 4.

Diagram, left to right: "Input $x$, output $y$, and passage $d$ when the token needs one", then "Critic $\mathcal{C}$" with the gray line "starts as Llama2 7B", then "A predicted token, trained to match GPT-4's token $r$".

Equation:

$$
\max_{\mathcal{C}} \quad \mathbb{E}_{((x, y), r) \sim \mathcal{D}_{\text{critic}}} \log p_{\mathcal{C}}(r \mid x, y)
$$

Text: Adjust the critic so that GPT-4's token gets the highest probability, averaged over the labeled examples $\mathcal{D}_{\text{critic}}$. This is ordinary training with one token as the target.

Takeaway box: The critic matches GPT-4 on 93.8% of Retrieve and 93.5% of IsSup labels, but only 80.2% of IsRel and 73.5% of IsUse.

> Step two. The critic starts as Llama2 7B and learns to predict the token GPT-4 chose. The equation says exactly that. Make GPT-4's token as probable as possible, averaged over the labeled examples. So it is ordinary training with one token as the target. The critic matches GPT-4 over ninety-three percent of the time on Retrieve and support, but only eighty on relevance and seventy-three on usefulness.


### Speaker 4: Labeling, the generator, and the inference overview (slides 18 to 20, 2:06)

#### 18. The critic labels every sentence (1:08)

Footer: Training step 3 of 4.

Line above the table: For each sentence, $\mathcal{C}$ first writes Retrieve. After a Yes, $\mathcal{R}$ searches with $x$ and $y_t$, and $\mathcal{C}$ writes IsRel for each of the top $K$ passages and IsSup for each relevant one.

| Case | Decide | Evidence | Judge evidence | Write | Verify |
| --- | --- | --- | --- | --- | --- |
| Good passage | [Retrieve = Yes] | [&lt;p&gt;] $d$ [&lt;/p&gt;] | [IsRel = Relevant] | $y_t$ | [IsSup] |
| Off-topic passage | [Retrieve = Yes] | [&lt;p&gt;] $d$ [&lt;/p&gt;] | [IsRel = Irrelevant] | $y_t$ | – |
| Passage reused | [Retrieve = Continue] | – | – | $y_t$ | – |
| No facts needed | [Retrieve = No] | – | – | $y_t$ | – |

Caption: Only one of the $K$ passages goes into the text. It is relevant, at least partially supported, and has the top retrieval score. If none qualifies, one is picked at random. One IsUse token closes the whole output.

> Step three is the critic's real job, turning every plain pair into the labeled text from the start of this section. This table is the full rule for one sentence. The critic first writes Retrieve. No means the sentence needs no facts, like a line in an essay about a summer vacation, so it goes in as it is. Continue means a passage retrieved earlier still covers it, so nothing new is searched. Yes means the retriever searches with the input and this sentence and returns the top K passages. The critic judges every one of them, writing IsRel, and IsSup if the passage is relevant. Only one goes into the text, the relevant, supported passage with the best retrieval score. If none qualifies, one is picked at random, so an off-topic passage can appear, marked Irrelevant. At the very end, the critic rates the whole answer once.

#### 19. The generator learns from the labeled data (0:30)

Footer: Training step 4 of 4.

A strip of seven cells, each labeled underneath as "read only" or "predicted".

| Cell | Label |
| --- | --- |
| input $x$ | read only |
| Retrieve = Yes | predicted |
| &lt;p&gt; passage &lt;/p&gt; | read only |
| IsRel = Relevant | predicted |
| sentence $y_t$ | predicted |
| IsSup = Fully | predicted |
| IsUse = 5 | predicted |

Equation:

$$
\max_{\mathcal{M}} \quad \mathbb{E}_{(x, y, r) \sim \mathcal{D}_{\text{gen}}} \log p_{\mathcal{M}}(y, r \mid x)
$$

Text: Make the sentences and tokens $r$ in the labeled data $\mathcal{D}_{\text{gen}}$ as likely as possible. Passages stay in the input but are masked from the loss.

Caption: Critique was added to the data offline, so no critic runs during this training. Llama2 7B or 13B, 3 epochs, batch size 128.

> Step four. The generator trains on this labeled text with the standard next-token objective. The equation looks like the critic's, except the model now predicts the sentences as well as the tokens. Passages stay in the input but are masked from the loss, which makes sense because the retriever supplies them at answer time. And since critique was added offline, no critic runs during this training.

#### 20. Inference phase (0:27)

Five boxes, left to right, with step numbers above them. Each box has a bold name and one line of detail, with clear space above the name.

| Step | Name | Detail |
| --- | --- | --- |
| 1 | Generator $\mathcal{M}$ | Decides if the next sentence needs evidence |
| 2 | Retriever $\mathcal{R}$ | Searches with $x$ and the previous sentence |
| 3 | Generator $\mathcal{M}$ | Writes one candidate per passage |
| 4 | Scoring | Turns the tokens into scores |
| 5 | Ranking | Orders the candidates |

Caption: These steps repeat for every sentence. If step 1 says no, the model writes the sentence directly. If the model writes Continue, it keeps the earlier passage.

Takeaway box: Inference means answering a new question, using only $\mathcal{M}$ and $\mathcal{R}$.

> Once training is done, we can answer new questions. This is inference, and it uses only the generator and the retriever. For every sentence there are five steps. The generator decides if it needs evidence, the retriever searches, the generator writes one candidate per passage, the tokens become scores, and the candidates are ranked. Let's go through them.


### Speaker 5: Inference steps (slides 21 to 25, 2:18)

#### 21. The model decides whether it needs evidence (0:33)

Footer: Inference step 1 of 5.

Equation:

$$
\text{retrieve if} \quad \frac{p(\text{Yes})}{p(\text{Yes}) + p(\text{No})} > \delta
$$

Text: $\mathcal{M}$ retrieves when the share of probability $p$ on Yes is above the threshold $\delta$.

A horizontal scale with three ticks: $\delta = 0$ always retrieves, $\delta = 0.2$ is used on most tasks, and $\delta = 1$ never retrieves.

Takeaway box: Changing $\delta$ needs no retraining. ASQA uses 0 because it needs citations. Lower it for factual tasks and raise it for open-ended ones.

> Step one. The generator compares the probability of Retrieve equals Yes against No, and retrieves when Yes has a large enough share. The threshold delta is a dial you set at answer time, with no retraining. Zero means always retrieve, which the paper uses for ASQA because it needs citations, and most tasks use point two. You would lower it for factual tasks and raise it for open-ended ones like essays.

#### 22. The model retrieves passages and writes candidates (0:30)

Footer: Inference steps 2 and 3 of 5.

| | Passage | Judge evidence | Candidate sentence | Verify |
| --- | --- | --- | --- | --- |
| $d_1$ | *Of the fifty states, eleven are named after an individual person.* | [IsRel = Relevant] | 11 of 50 states' names come from persons. | [IsSup = Fully] |
| $d_2$ | *Popular names by states. In Texas, Emma is a popular baby name.* | [IsRel = Irrelevant] | Texas is named after a Native American tribe. | no token |
| $d_3$ | *California was named after a fictional island in a Spanish book.* | [IsRel = Relevant] | California's name has its origins in a 16th-century novel. | [IsSup = Partially] |

Takeaway box: Each candidate uses one passage, so a bad passage spoils only its own candidate.

> Steps two and three. The retriever searches with the input and the previous sentence and returns five passages by default. The generator handles each passage separately. It judges the passage, writes a sentence from it, and checks whether the passage supports that sentence. In the middle row, the model itself marks the baby-names passage irrelevant, and that passage can only spoil its own candidate.

#### 23. Each critique token becomes a score (0:30)

Footer: Inference step 4 of 5.

Text: Each value $v$ has a probability $p(v)$, and those give a score $s_t$.

| Token | Score $s_t$ | What it means |
| --- | --- | --- |
| IsRel | $\dfrac{p(\text{Relevant})}{p(\text{Relevant}) + p(\text{Irrelevant})}$ | The share of probability on Relevant, from 0 to 1 |
| IsSup | $\dfrac{p(\text{Fully}) + 0.5 \cdot p(\text{Partially})}{p(\text{Fully}) + p(\text{Partially}) + p(\text{No support})}$ | Partial support counts half, and the score runs from 0 to 1 |
| IsUse | $\dfrac{\sum_{v=1}^{5} u_v\, p(v)}{\sum_{v=1}^{5} p(v)}$ | The average rating, with $u$ counting ratings 1 to 5 as −1, −0.5, 0, 0.5, and 1 |

> Step four. To compare candidates we need numbers, so each critique token becomes a score from the probabilities the model gave its values. IsRel is the share of probability on Relevant. IsSup is similar, but partial support counts half. IsUse is an average rating from minus one to one. So every score comes straight from probabilities the model already produced, with no extra model.

#### 24. Candidates are ranked by score (0:27)

Footer: Inference step 5 of 5.

Equation:

$$
f(y_t, d) = p(y_t \mid x, d, y_{<t}) + \sum_{G} w^{G} s_t^{G}
$$

Text: A good candidate is a likely sentence with high critique scores.

Caption: $y_{<t}$ means the sentences before $y_t$, and $s_t^G$ is the score of critique $G$. $G$ runs over IsRel, IsSup, and IsUse, with default weights $w^G$ of 1, 1, and 0.5.

Takeaway box: A higher weight on IsSup favors well-supported output, and a hard filter can drop any sentence marked [IsSup = No support]. Neither needs retraining.

> Step five. A candidate's final score adds how likely the sentence is to the weighted critique scores. By default relevance and support get weight one and usefulness gets a half. You can change these at answer time. Raising the weight on IsSup favors well-supported output, and a hard rule can drop any sentence marked no support. Neither needs retraining.

#### 25. How the final answer is chosen (0:19)

Left, a small tree. From $y_{<t}$ three branches lead to candidates for sentence $t$ with scores 3.05, 2.50, and 0.55. The two best stay solid and 0.55 is dashed. Each kept candidate branches to three circles for sentence $t+1$, and two of those six stay solid. Note under the tree: Numbers are candidate scores $f$. Solid paths are kept and dashed ones dropped.

Right, three short points:

- The best sentence now can lead to a worse answer later.
- The $B = 2$ best partial answers survive each sentence. This is segment-level beam search.
- Each one grows by one candidate per passage, and the best complete answer wins.

Caption: Words within a sentence are chosen greedily.

> But the best sentence now can lead to a worse answer later. So the model keeps the two best partial answers after each sentence and returns the best complete one. This is segment-level beam search. Inside a sentence, words are chosen greedily.


### Speaker 6: Training vs inference, setup, and first results (slides 26 to 31, 2:32)

#### 26. Training vs inference (0:56)

| | Labeling training data | Answering a new question |
| --- | --- | --- |
| The answer | Already known | Written one sentence at a time |
| Tokens written by | Critic $\mathcal{C}$ | Generator $\mathcal{M}$ |
| Retrieve decided | After reading the sentence | Before the sentence exists |
| Search query | $x$ and the sentence itself | $x$ and the previous sentence |
| Passages per sentence | $K$ judged, one inserted | $K$, each with its own candidate |
| Result | Text to train on | The final answer |

Takeaway box: The generator learns to predict the critic's decision before the sentence exists.

> Now let's put the two phases side by side, because they are easy to mix up. When labeling, the answer already exists, so the critic can read a sentence first and then decide if it needed evidence. When answering, the sentence does not exist yet, so the generator has to decide before writing it, and that is exactly what its training teaches. The search differs for the same reason. Labeling can search with the sentence itself, while answering only has the previous sentence. A training text holds one passage per sentence, so K are judged and one is inserted, while answering tries all K and keeps the best. And labeling produces text to train on, while answering produces the final answer.

#### 27. Divider: What are the experimental findings? (0:03)

Section slide 4 of 5.

> So how well does it work?

#### 28. Six tasks and how each is scored (0:26)

Each task has a smaller gray second line.

| Task | Metric | What it measures |
| --- | --- | --- |
| PubHealth, ARC-Challenge (second line: pick from fixed options) | Accuracy | Share of test items answered correctly |
| PopQA, TriviaQA (second line: answer a factual question) | Accuracy | Correct if the output contains a gold answer |
| Biographies (second line: write a bio of a named person) | FactScore | Share of facts backed by a knowledge source |
| ASQA (second line: long answer with citations) | Five metrics | Correctness, fluency, and citations |

Caption: Every test is zero-shot, with no demonstrations. A gold answer is a correct reference answer. Higher is better on every metric.

> The paper tests six tasks, all zero-shot. PubHealth and ARC-Challenge pick from fixed options and use accuracy. PopQA and TriviaQA answer factual questions, and an answer counts if it contains a gold answer. Biographies use FactScore, the share of facts backed by a knowledge source. And ASQA needs long answers with citations, so it has five metrics.

#### 29. Baselines (0:27)

| Group | Models | What it tests |
| --- | --- | --- |
| No retrieval | Llama2, Alpaca, Llama2-chat, ChatGPT, CoVE | Do evidence and critique beat larger or better-tuned models? |
| Standard RAG | Llama2, Alpaca, Llama2-chat, and ChatGPT with the top passages | Does critique add anything to retrieval? |
| Same pairs before labeling | Llama2-FT 7B, which retrieves only at test time | Is the gain only from the training pairs? |
| Other systems | Perplexity.ai, plus published results for SAIL 7B and Toolformer 6B | How do other retrieval methods compare? |

Caption: Retrieval models get the top 5 passages from Wikipedia. Biographies and open QA add 5 from web search, and ASQA uses a provided set for every model.

> The baselines isolate where any gain comes from. No-retrieval models test whether evidence and critique beat larger or better-tuned models. Standard RAG tests whether critique adds anything to retrieval. The key control is Llama2-FT, trained on the same pairs before labeling and retrieving only at test time. If it matched Self-RAG, the gain would come from the data alone.

#### 30. Self-RAG vs open models (0:18)

Bars show the lead of Self-RAG 13B over the best open baseline, under the heading "Self-RAG ahead", with the three values on the right.

| Task and metric | Lead | Self-RAG 13B | 7B | Best open |
| --- | --- | --- | --- | --- |
| ARC-Challenge accuracy | +7.3 | 73.1 | 67.3 | 65.8 |
| PubHealth accuracy | +5.3 | 74.5 | 72.4 | 69.2 |
| PopQA accuracy | +4.0 | 55.8 | 54.9 | 51.8 |
| TriviaQA accuracy | +2.4 | 69.3 | 66.4 | 66.9 |
| Biography FactScore | +0.3 | 80.2 | 81.2 | 79.9 |

Caption: Bars are Self-RAG 13B minus the best of Llama2, Llama2-chat, Llama2-FT, Alpaca, and SAIL. The paper attributes 7B's lead on biographies to shorter, precisely grounded outputs.

Takeaway box: Self-RAG 13B beats the best open baseline on all five tasks.

> Against open models, Self-RAG 13B beats the best baseline on all five measures, most clearly on ARC-Challenge and PubHealth. On biographies the smaller 7B model is even ahead, which the paper attributes to its shorter, more precisely grounded outputs.

#### 31. Self-RAG vs open models on ASQA (0:22)

Bars as on the previous slide. Each row has the metric name with its meaning on a smaller gray second line.

| Metric | Meaning | Lead | Self-RAG 13B | 7B | Best open |
| --- | --- | --- | --- | --- | --- |
| Citation precision | citations that support their statement | +50.5 | 70.3 | 66.9 | 19.8 |
| Citation recall | statements fully supported by their citations | +35.2 | 71.3 | 67.8 | 36.1 |
| MAUVE | fluency, closeness to human-written text | +1.0 | 71.6 | 74.3 | 70.6 |
| ROUGE | correctness, word overlap with a reference answer | +0.3 | 37.0 | 35.7 | 36.7 |
| str-em | correctness, gold answers found word for word | −3.1 | 31.7 | 30.0 | 34.8 |

Caption: Bars are Self-RAG 13B minus the best of Llama2, Llama2-chat, Llama2-FT, and Alpaca.

Takeaway box: Citation precision jumps from 19.8 to 70.3, and correctness barely moves.

> ASQA has five metrics, and the gray text says what each one means. Citations are the big win. Citation precision jumps from about twenty to seventy, and recall nearly doubles. Correctness barely moves, so the gain here is in checkable output rather than in more correct answers.


### Speaker 7: ChatGPT, Llama2-FT, and ablations (slides 32 to 36, 2:26)

#### 32. Self-RAG vs ChatGPT (0:23)

Bars show Self-RAG 13B minus ChatGPT, to the right when Self-RAG leads and to the left when ChatGPT leads.

| Dataset and metric | Difference | Self-RAG 13B | ChatGPT |
| --- | --- | --- | --- |
| Biography FactScore | +8.4 | 80.2 | 71.8 |
| ASQA citation precision | +5.2 | 70.3 | 65.1 |
| PopQA accuracy | +5.0 | 55.8 | 50.8 |
| PubHealth accuracy | +4.4 | 74.5 | 70.1 |
| ARC-Challenge accuracy | −2.2 | 73.1 | 75.3 |
| ASQA ROUGE | −2.9 | 37.0 | 39.9 |
| TriviaQA accuracy | −5.0 | 69.3 | 74.3 |
| ASQA citation recall | −5.3 | 71.3 | 76.6 |
| ASQA MAUVE | −8.1 | 71.6 | 79.7 |
| ASQA str-em | −9.0 | 31.7 | 40.7 |

Caption: Self-RAG 13B minus the better of ChatGPT with and without retrieval. Five of the ten rows are ASQA.

Takeaway box: Self-RAG 13B wins 4 of 10 measures, and ChatGPT wins the other 6.

> Against ChatGPT the picture is mixed. Taking the better of ChatGPT with and without retrieval, Self-RAG wins four of ten measures, namely biographies, citation precision, PopQA, and PubHealth. ChatGPT wins the other six. Five of the ten rows come from ASQA, so that one task weighs heavily in this count.

#### 33. Self-RAG vs Llama2-FT (0:29)

Bars show Self-RAG 7B minus Llama2-FT 7B.

| Dataset and metric | Difference | Self-RAG 7B | Llama2-FT 7B |
| --- | --- | --- | --- |
| ASQA citation precision | +61.9 | 66.9 | 5.0 |
| ASQA citation recall | +60.3 | 67.8 | 7.5 |
| ASQA MAUVE | +23.1 | 74.3 | 51.2 |
| TriviaQA accuracy | +9.1 | 66.4 | 57.3 |
| PubHealth accuracy | +8.1 | 72.4 | 64.3 |
| PopQA accuracy | +6.2 | 54.9 | 48.7 |
| Biography FactScore | +3.0 | 81.2 | 78.2 |
| ARC-Challenge accuracy | +1.5 | 67.3 | 65.8 |
| ASQA ROUGE | −0.1 | 35.7 | 35.8 |
| ASQA str-em | −1.0 | 30.0 | 31.0 |

Caption: Llama2-FT trains on the plain pairs and retrieves only at test time. The paper reports no significant gain for it from 50k to 150k pairs, unlike Self-RAG.

Takeaway box: Trained on the same pairs before labeling, Llama2-FT trails on 8 of 10 measures.

> So is the gain just the data? Llama2-FT trains on the same pairs before labeling and retrieves only at test time. It falls behind on eight of ten measures, most of all on citations, and the paper reports no significant gain for it when data grows from fifty to a hundred and fifty thousand pairs. So the gain is not just the data.

#### 34. Removing parts during training (0:37)

One row per model. Each row shows the model's name with a smaller gray second line, a strip of six cells for its training text, and three scores. The cells are Retrieve, passage, IsRel, $y_t$, IsSup, and IsUse. A cell missing from that model's training text is empty and dashed.

| Model | Second line | Cells present | PopQA | PubHealth | ASQA str-em |
| --- | --- | --- | --- | --- | --- |
| Full model | passages and tokens | all six | 45.5 | 73.5 | 32.1 |
| No retriever | plain pairs, like Llama2-FT | $y_t$ | 43.6 | 67.8 | 31.0 |
| No critic | top passage, no tokens | passage, $y_t$ | 42.6 | 72.0 | 18.1 |

Caption: Three Llama2 7B models on the same 50k pairs. A dashed cell is missing from that model's training text. PopQA and PubHealth report accuracy, and ASQA uses 150 sampled questions.

Takeaway box: Passages without tokens hurt ASQA most, and plain pairs hurt PubHealth most.

> An ablation removes one part to see how much it matters. These three 7B models train on the same fifty thousand pairs, and only their training text differs. The full model sees passages and tokens. No retriever sees plain pairs, which is how Llama2-FT is trained, just on less data. No critic always gets the top passage but no tokens. With passages but no tokens, ASQA falls from thirty-two to eighteen. With neither, PubHealth falls from about seventy-four to sixty-eight.

#### 35. Removing parts while answering (0:29)

One row per variant, with its description on a smaller gray second line, and aligned bars for three tasks. "Not run" appears where the paper has no result.

| Variant | What changes | PopQA | PubHealth | ASQA str-em |
| --- | --- | --- | --- | --- |
| Full model | default settings | 45.5 | 73.5 | 32.1 |
| No retrieval | never retrieves | 24.7 | 73.0 | not run |
| Hard constraint | retrieves only if it writes Yes | 28.3 | 72.6 | not run |
| Retrieve top 1 | always uses only the top passage | 41.8 | 73.1 | 28.6 |
| Remove IsSup | ranks without the support score | 44.1 | 73.2 | 30.6 |

Caption: The same Llama2 7B model trained on 50k pairs each time. Only the answering step changes.

Takeaway box: Retrieving less hurts PopQA most, and PubHealth barely changes.

> These use the same trained model and change only how it answers. If it never retrieves, PopQA nearly halves while PubHealth barely moves. Retrieving only when the model actually writes Yes, instead of using the threshold, also cuts PopQA sharply. Using only the top passage, as standard RAG does, lowers PopQA and ASQA, and ranking without the support score also hurts ASQA.

#### 36. Effect of the test-time settings (0:28)

| Setting | Change | Effect |
| --- | --- | --- |
| IsSup weight, on ASQA | From about 0.1 to 2 | Citation precision rises about 1 point, and MAUVE falls about 10 |
| Threshold $\delta$, on PopQA | From 0 to 0.7 | Retrieval falls to about 20%, and normalized accuracy to about 0.45 |
| Threshold $\delta$, on PubHealth | From 0 to 0.7 | Retrieval falls to 0, and normalized accuracy stays about 0.99 |

Caption: The IsSup row uses Self-RAG 7B. Normalized accuracy is 1 at $\delta = 0$.

Takeaway box: A larger IsSup weight barely helps. Retrieving less hurts PopQA far more than PubHealth.

> These are the dials from the method section, tested directly. On ASQA, a larger IsSup weight raises citation precision by about one point but lowers fluency by about ten, because longer, more fluent text often has more claims the citations do not fully support. Raising the retrieval threshold costs PopQA more than half its normalized accuracy and leaves PubHealth almost unchanged.


### Speaker 8: Grounding, limitations, impact, and close (slides 37 to 44, 2:25)

#### 37. Are the answers grounded? (0:23)

Two charts side by side.

Left chart, "Human evaluation (%)", paired bars:

| Measure | PopQA | Biographies |
| --- | --- | --- |
| Plausible and supported | 92.5 | 70.0 |
| IsRel token correct | 95.0 | 90.0 |
| IsSup token correct | 90.0 | 85.0 |

Right chart, "Correct answers not in passages (%)", bars:

| Model | Share |
| --- | --- |
| Self-RAG 7B | 2 |
| Alpaca | 15 |
| Llama2-chat 13B | 18 |
| Alpaca 30B | 20 |

Caption: Annotators reviewed 50 samples and left out answers the model marked Irrelevant or No support. Plausible means a reasonable, on-topic response, and supported means the evidence can verify the answer. The right chart uses TriviaQA and PopQA.

Takeaway box: Annotators judged most answers plausible and supported, and only 2% of Self-RAG's correct answers are missing from its passages.

> Finally, are the answers grounded? In a small human study, annotators judged most answers plausible and supported and most tokens correct, with short answers doing better than biographies. And only two percent of Self-RAG's correct answers are missing from its passages, against fifteen to twenty percent for other models.

#### 38. Divider: What are its limitations and impact? (0:04)

Section slide 5 of 5.

> So how far can we trust all this?

#### 39. Strengths and assumptions (0:36)

| Strength | What it assumes |
| --- | --- |
| No critic runs during training, because critique is added offline | GPT-4's labels are good enough, though only 20 per type were checked by hand |
| One model retrieves, writes, and critiques | A 7B critic can teach it, though it matches GPT-4 on 80.2% of IsRel and 73.5% of IsUse |
| Bad passages can be flagged and dropped | The search returns useful passages, and the authors report weaker retrieval on open-ended instructions |
| Behavior can be tuned without retraining | Token probabilities work as scores, which the paper checks only through final metrics |

> The method has real strengths, and each rests on an assumption. No critic runs during training because critique is added offline, but only twenty GPT-4 labels per type were checked by hand. One model does everything, but the critic that teaches it matches GPT-4 on only eighty percent of relevance labels. Bad passages can be dropped only if the search finds good ones. And tuning assumes token probabilities work as scores, which the paper checks only through final metrics.

#### 40. Limitations and improvements (0:25)

| Limitation | What we would do |
| --- | --- |
| Answering cost is not reported | Measure it, and stop a branch once the model writes Irrelevant |
| Retrieval is weaker on open-ended instructions | Train the retriever on those instructions, as the authors also suggest |
| A sentence can mix supported and unsupported facts | Judge units smaller than a sentence, which the framework allows |

> The main limitations are practical. The paper never reports how long answering takes, retrieval is weaker on open-ended instructions, and one sentence can mix supported and unsupported facts. We would measure the cost and stop a branch once the model writes Irrelevant, train the retriever on those instructions, and judge units smaller than a sentence.

#### 41. Are the experiments sufficient? (0:30)

Two independent lists side by side.

| What is missing |
| --- |
| No variance or significance tests |
| A human study of only 50 samples |
| An output length limit stated only for the baselines on ASQA |

| What we found wrong |
| --- |
| The abstract says Self-RAG beats ChatGPT on QA and reasoning, but ChatGPT wins TriviaQA and ARC-Challenge |
| The paper says the critic tops 90% on most token types, but that holds for two of the four |
| Reading the released code, we found over 50 issues, including 13 crashes |

Takeaway box: The coverage is broad, and none of this undoes the core idea.

> The experiments are broad, but there are gaps. There are no significance tests, the human study has only fifty samples, and on ASQA an output length limit is stated only for the baselines. We also checked the claims. ChatGPT wins TriviaQA and ARC-Challenge despite the abstract, and reading the released code we found over fifty issues, including thirteen crashes. None of this undoes the core idea.

#### 42. Impact since publication (0:19)

Four rows with no heading row.

| | |
| --- | --- |
| Recognition | Oral at ICLR 2024, which the authors list as top 1% |
| Citations | Over 1,000 on OpenAlex as of November 2026 |
| Later research | Corrective RAG adds a retrieval evaluator and is built on top of standard RAG and Self-RAG |
| Practice | The code and 7B and 13B models are public, and IBM publishes a Self-RAG agent tutorial using LangGraph |

> The paper was an oral at ICLR 2024, which the authors list as top one percent, and OpenAlex shows over a thousand citations. Corrective RAG is built on top of standard RAG and Self-RAG, and the code and models are public.

#### 43. Team Contribution (0:03)

| Student | Roll number | Contribution | Score |
| --- | --- | --- | --- |
| Chavda Haarit Ravindrakumar | 23110077 | Equal | 1.0 |
| Darpana Dharmesh Desai | 23110085 | Equal | 1.0 |
| Devansh Lodha | 23110091 | Equal | 1.0 |
| Jagtap Arnav Shrikant | 23110143 | Equal | 1.0 |
| Keval Pintubhai Shah | 26120003 | Equal | 1.0 |
| Ruddhi Pintubhai Shah | 26250038 | Equal | 1.0 |
| Suhani | 24110358 | Equal | 1.0 |
| Vedant Acharya | 23110010 | Equal | 1.0 |

> All eight of us contributed equally.

#### 44. Thank you (0:05)

Same corners as the title slide.

Center, stacked: "Thank you", "Questions?", a short rule, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "Group: Mixture of Experts".

> Thank you. We would be happy to take your questions.


## Appendix

The appendix follows the thank-you slide and is not presented.

#### 45. Divider: Appendix

A title and four rows that list where each group starts.

| Group | Slides |
| --- | --- |
| Training details | 46 to 49 |
| Scoring and analysis | 50 to 54 |
| When tokens catch or miss mistakes | 55 and 56 |
| Mistakes in the paper and code | 57 and 58 |

Every appendix slide has the footer Appendix.

#### 46. The training data has 145,619 input-output pairs

Two bar charts side by side.

| Instruction following (92,008) | Pairs |
| --- | --- |
| GPT-4 Alpaca | 26,168 |
| Stanford Alpaca | 25,153 |
| FLAN-V2 | 17,817 |
| ShareGPT | 13,406 |
| Open Assistant 1 | 9,464 |

| Knowledge-intensive (53,611) | Pairs |
| --- | --- |
| Wizard of Wikipedia | 17,367 |
| Natural Questions | 15,535 |
| FEVER | 9,966 |
| OpenBookQA | 4,699 |
| ASQA | 3,897 |
| ARC-Easy | 2,147 |

Takeaway box: At this stage these pairs are plain text with no reflection tokens.

#### 47. How closely the critic matches GPT-4

Paired bars.

| Token | Llama2 7B (the critic used) | FLAN-3B (a smaller model) |
| --- | --- | --- |
| Retrieve | 93.8 | 85.6 |
| IsSup | 93.5 | 73.1 |
| IsRel | 80.2 | 82.0 |
| IsUse | 73.5 | 72.1 |

Caption: Share of GPT-4's labels each model reproduces on a held-out test set (%).

Takeaway box: Agreement is over 93% on Retrieve and IsSup, and lower on the other two.

#### 48. Training setup

| Item | Setting |
| --- | --- |
| Critic | Llama2 7B |
| Generator | Llama2 7B or 13B |
| Retriever | Contriever-MS MARCO, off the shelf |
| Training | 3 epochs, batch size 128 |
| Learning rate | Peak of 2e-5, 3% warmup, then linear decay |
| Maximum length | 2,048 tokens for 7B and 1,524 for 13B |
| Hardware | Four Nvidia A100 GPUs with 80GB of memory |

#### 49. Labeling a three-sentence output

| Part | $\mathcal{C}$ writes | Search with | Passage added to the text |
| --- | --- | --- | --- |
| Whole example | [Retrieve = Yes] | $x$ and all of $y$ | None. $\mathcal{C}$ reads this initial passage to write Retrieve for each sentence. |
| Sentence $y_1$ | [Retrieve = Yes] | $x$ and $y_1$ | The best result, just before $y_1$ |
| Sentence $y_2$ | [Retrieve = Continue] | No search | None |
| Sentence $y_3$ | [Retrieve = No] | No search | None |

Finished training text: [Retrieve = Yes] [&lt;p&gt;] $d$ [&lt;/p&gt;] [IsRel = Relevant] $y_1$ [IsSup = Fully] [Retrieve = Continue] $y_2$ [Retrieve = No] $y_3$ [IsUse = 5]

Takeaway box: Each Yes sentence gets its own passage, and the others get none.

#### 50. A worked score example

| | $p(y_t \mid \dots)$ | IsRel | IsSup | IsUse | $f$ |
| --- | --- | --- | --- | --- | --- |
| $d_1$ | 0.80 | 0.90 | 0.90 | 0.90 | 3.05 |
| $d_3$ | 0.80 | 0.80 | 0.55 | 0.70 | 2.50 |
| $d_2$ | 0.70 | 0.05 | 0.00 | −0.40 | 0.55 |

Caption: Illustrative numbers, where $f$ for $d_1$ is 0.80 + 0.90 + 0.90 + 0.5 × 0.90 = 3.05. $d_2$ has no IsSup token, so its support score is 0. IsUse runs from −1 to 1.

Takeaway box: The candidate built on $d_1$ has the highest score, so it is kept.

#### 51. Standard RAG vs no retrieval

One row per task with one mark per baseline. Filled means the score rose, hollow means it fell, and gray means no change.

| Task | Rose | No change | Fell | Largest change |
| --- | --- | --- | --- | --- |
| PopQA | 6 | 0 | 0 | Llama2-chat 13B, 20.0 → 51.8 |
| Biographies | 5 | 0 | 0 | Llama2 7B, 44.5 → 78.0 |
| TriviaQA | 5 | 0 | 1 | Llama2 7B, 30.5 → 42.5 |
| ARC-Challenge | 3 | 1 | 2 | Llama2 7B, 21.8 → 48.0 |
| PubHealth | 2 | 0 | 4 | ChatGPT, 70.1 → 54.7 |

Caption: Each mark is one baseline going from no retrieval to RAG. ChatGPT has no biography score with retrieval.

Takeaway box: Retrieval helps when passages state the answer, and PubHealth mostly falls.

#### 52. Effect of the support weight

Two line charts, with weight on IsSup on the x axis.

| Weight on IsSup | About 0.1 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- |
| Citation precision | 69.7 | 70.0 | 70.5 | 70.5 |
| MAUVE fluency | 96.4 | 87.8 | 86.5 | 86.1 |

Caption: Self-RAG 7B on ASQA. Longer, more fluent text often contains more claims the citations do not fully support.

Takeaway box: A larger IsSup weight barely raises precision and clearly lowers fluency.

#### 53. Effect of the retrieval threshold

Two line charts, one for each task, with threshold $\delta$ on the x axis.

| Threshold $\delta$ | 0 | 0.1 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| PopQA normalized accuracy | 1.00 | 0.89 | 0.75 | 0.45 |
| PopQA retrieval frequency | 1.00 | 1.00 | 0.60 | 0.20 |
| PubHealth normalized accuracy | 1.00 | 1.00 | 0.985 | 0.99 |
| PubHealth retrieval frequency | 1.00 | 1.00 | 0.16 | 0.00 |

Caption: Normalized accuracy starts at 1 for $\delta = 0$, and frequency is how often the model retrieves. A higher $\delta$ means less retrieval.

Takeaway box: Retrieving less costs little accuracy on PubHealth and a lot on PopQA.

#### 54. Effect of training data size

One line chart, with training pairs on a log-spaced x axis.

| Training pairs | 5k | 10k | 20k | 50k | 150k |
| --- | --- | --- | --- | --- | --- |
| PubHealth accuracy | 71.8 | 70.9 | 71.8 | 73.5 | 72.8 |
| ASQA citation precision | 41.2 | 51.6 | 57.4 | 58.9 | 70.1 |
| PopQA accuracy | 33.9 | 39.3 | 40.8 | 45.5 | 54.9 |

Caption: Self-RAG 7B, scores in %.

Takeaway box: More data helps PopQA and ASQA citations most, and PubHealth stays flat.

#### 55. Tokens can catch the model's mistakes

| | Asked for a bio of Rory Tapner | Asked for a bio of G. Venugopal |
| --- | --- | --- |
| Passage | *... in September 2010 was appointed CEO of Coutts ... In February 2015 he stepped down as CEO ...* | *... known for his work in Malayalam films ... he has sung in more than 300 films ...* |
| Model wrote | "... has served as the Chief Executive Officer of Coutts ... since 2010." | "He has sung over 300 songs in Malayalam, Tamil, and Hindi movies." |
| Its token | [IsSup = No support] | [IsSup = Partially] |
| Was the token right? | Yes. The role ended in 2015. | Yes. It says films, not songs, and omits Tamil and Hindi. |

Takeaway box: The model writes these tokens itself, so it can flag its own mistakes.

#### 56. Tokens can also miss mistakes

| | Asked for the author of The Lie | Same Tapner bio, next sentence |
| --- | --- | --- |
| Passage | *Lying is a 2011 long-form essay book by American author ... Sam Harris.* | *Rory Tapner (born 30 September 1959) is a UK businessman ...* |
| Model wrote | "The author of The Lie is Sam Harris." | "Rory Tapner was born on 30 September 1959 in London, England." |
| Its tokens | [IsRel = Relevant] [IsSup = Fully] | [IsSup = Fully] |
| Was the token right? | No. *Lying* is a different book. | No. The printed passage does not say London. |

Takeaway box: Tokens make answers easier to verify, but someone must still verify them.

#### 57. Mistakes we noticed in the paper

| The paper says | What we found |
| --- | --- |
| Self-RAG outperforms ChatGPT on open-domain QA and reasoning (abstract) | ChatGPT is higher on TriviaQA, 74.3 against 69.3, and on ARC-Challenge, 75.3 against 73.1 |
| Self-RAG is the best non-proprietary model on all tasks (Section 5.1) | On ASQA str-em, Alpaca 13B with retrieval scores 34.8 against 31.7 |
| The critic agrees with GPT-4 above 90% on most token types (Section 3.2.1) | The paper's appendix shows this for two of the four |
| Citation precision checks that a claim is fully supported by its citations (Section 5.1) | In the benchmark's definition, that is citation recall, and precision looks for irrelevant citations |

Takeaway box: None of these undo the core idea, and we also found smaller issues such as typos.

#### 58. Mistakes we noticed in the released code

| What the code does | Effect |
| --- | --- |
| Long-form usefulness reads the wrong token list | The score is always zero |
| The long-form script ignores weight arguments | Weights stay at 1, 1, and 0.5 |
| Short-form threshold uses raw log-probabilities | The retrieval decision is inverted |
| The loss mask runs from first passage to last | Text between passages is not trained on |

Caption: These describe the public code and do not show that the reported numbers are wrong.

Takeaway box: We found over 50 issues, including 13 crashes, and cannot tell which version produced the paper's numbers.


## If the first rehearsal runs long

- Move slide 36 to the appendix, which saves about 0:28.
- Move slide 35 to the appendix, which saves about 0:29.
