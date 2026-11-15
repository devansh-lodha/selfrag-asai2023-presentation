# Speaker Notes: Self-RAG

Spoken script for the talk, 2361 words. That is about 18:10 at 130 words per minute. Slide changes and seven handovers add about 50 seconds, for about 19:00 in total.

## Division

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

Speakers are numbered for now, and names still need to be assigned. The appendix, slides 45 to 58, is not presented.

## Speaker 1: Problem (slides 1 to 7, 2:18)

Slide 1, Title (0:09):

Hi everyone. We are Mixture of Experts, and today we will walk you through Self-RAG, a paper from ICLR 2024.

Slide 2, What problem does the paper solve? (0:02):

Let's start with the problem.

Slide 3, Language models get rare facts wrong (0:25):

Language models get facts wrong, and rare topics are the worst case. PopQA asks about entities with under a hundred Wikipedia views a month. Llama2 gets about fifteen percent right, going from seven to thirteen billion parameters changes nothing, and even ChatGPT is under a third. A bigger model alone does not fix this.

Slide 4, Retrieval-Augmented Generation (RAG) (0:21):

The standard fix is retrieval-augmented generation, or RAG. A retriever searches a large collection for the input, the top K passages go into the prompt, and the language model writes from them. It clearly helps. On PopQA, Alpaca 13B goes from twenty-four to forty-six percent.

Slide 5, Three problems with standard RAG (0:29):

But standard RAG has three problems. It retrieves even when no facts are needed, so a vacation essay still fetches passages. It can retrieve off-topic passages, like a line about baby names for a question about state names. And nothing makes the output agree with the passages. Here the model credits Christopher Columbus for California, while its own passage says a fictional island.

Slide 6, Retrieval alone is not enough (0:23):

The numbers show these problems are real. On PubHealth, which checks health claims, adding retrieval lowers accuracy for four of the six baselines. And on ASQA, where answers must cite passages, open models rarely cite anything that supports their claim. The best reaches about twenty percent, while ChatGPT reaches sixty-five.

Slide 7, Existing work fixes only one piece (0:29):

People have tried to fix parts of this. On the retrieval side, SAIL always retrieves a fixed number of passages, adaptive methods often lose speed or attribution, and passage filters add another model at test time. On the critique side, RLHF needs reward models during training, control tokens never assess the output, and repeated prompting makes answering slow. Each fixes only one piece.

## Speaker 2: Motivation and key contribution (slides 8 to 12, 2:04)

Slide 8, Motivation (0:20):

This matters because factual errors can spread misinformation and give dangerous advice. So the goal is one language model that decides when to retrieve, judges what it finds, and checks what it writes, while adding little training overhead and staying adjustable at test time.

Slide 9, What is the key contribution? (0:03):

So what is the key idea?

Slide 10, How Self-RAG works (0:34):

Self-RAG writes one sentence at a time. Before each sentence, it asks whether it needs evidence. If not, it just writes the sentence. If yes, it retrieves K passages, writes one candidate from each, critiques them, and keeps the best. This repeats until the answer is finished, and then the model rates the whole answer. Each shaded box is a decision, which the model records by writing a special token, called a reflection token.

Slide 11, Four types of reflection tokens (0:36):

There are four types of these tokens. Retrieve controls retrieval, and Continue means keep using the passage you already have. The other three are critiques. IsRel asks if a passage is useful, IsSup asks if it supports the sentence, and IsUse rates the whole answer from one to five. The key insight is that a token is predicted just like a word, so ordinary training can teach it, and its probability can later serve as a score.

Slide 12, Standard RAG vs Self-RAG (0:31):

Each row here answers one of the three problems. Standard RAG always retrieves once before writing, while Self-RAG retrieves on demand, so tasks that need no facts keep their creativity. Standard RAG puts all passages in one prompt, while Self-RAG writes one candidate per passage, so a bad passage spoils only its own candidate. And Self-RAG checks relevance, support, and usefulness, with no extra model at test time.

## Speaker 3: Training overview, GPT-4, and the critic (slides 13 to 17, 2:00)

Slide 13, How does the method work? (0:03):

Now let's see how this is built.

Slide 14, The training data we need (0:31):

Let's start from where training has to end up. At the top is a plain training pair. Below is the same output after labeling. Each sentence now has a Retrieve token, a retrieved passage, a relevance token, and a support token, and one usefulness token closes the answer. If a model reads enough text like this, it learns to write it. So how do we produce this text?

Slide 15, Training phase (0:29):

Training has four steps. No dataset contains these tokens, hand labeling is too expensive, and using GPT-4 on everything raises costs and hurts reproducibility. So GPT-4 labels only a sample, and a Llama2 model called the critic learns to copy it. The critic and the retriever then label all 145,619 pairs, and the generator, the model we actually use, learns from the result.

Slide 16, GPT-4 labels a sample of the data (0:26):

Step one. Each token type has its own prompt, with an instruction, a few demonstrations, and one sampled example. The table shows what GPT-4 reads for each type and how many labels it produced. The authors then read twenty labels per type by hand and agreed with at least ninety percent, except eighty percent on usefulness.

Slide 17, The critic learns from GPT-4's labels (0:31):

Step two. The critic starts as Llama2 7B and learns to predict the token GPT-4 chose. The equation says exactly that. Make GPT-4's token as probable as possible, averaged over the labeled examples. So it is ordinary training with one token as the target. The critic matches GPT-4 over ninety-three percent of the time on Retrieve and support, but only eighty on relevance and seventy-three on usefulness.

## Speaker 4: Labeling, the generator, and the inference overview (slides 18 to 20, 2:06)

Slide 18, The critic labels every sentence (1:08):

Step three is the critic's real job, turning every plain pair into the labeled text from the start of this section. This table is the full rule for one sentence. The critic first writes Retrieve. No means the sentence needs no facts, like a line in an essay about a summer vacation, so it goes in as it is. Continue means a passage retrieved earlier still covers it, so nothing new is searched. Yes means the retriever searches with the input and this sentence and returns the top K passages. The critic judges every one of them, writing IsRel, and IsSup if the passage is relevant. Only one goes into the text, the relevant, supported passage with the best retrieval score. If none qualifies, one is picked at random, so an off-topic passage can appear, marked Irrelevant. At the very end, the critic rates the whole answer once.

Slide 19, The generator learns from the labeled data (0:30):

Step four. The generator trains on this labeled text with the standard next-token objective. The equation looks like the critic's, except the model now predicts the sentences as well as the tokens. Passages stay in the input but are masked from the loss, which makes sense because the retriever supplies them at answer time. And since critique was added offline, no critic runs during this training.

Slide 20, Inference phase (0:27):

Once training is done, we can answer new questions. This is inference, and it uses only the generator and the retriever. For every sentence there are five steps. The generator decides if it needs evidence, the retriever searches, the generator writes one candidate per passage, the tokens become scores, and the candidates are ranked. Let's go through them.

## Speaker 5: Inference steps (slides 21 to 25, 2:18)

Slide 21, The model decides whether it needs evidence (0:33):

Step one. The generator compares the probability of Retrieve equals Yes against No, and retrieves when Yes has a large enough share. The threshold delta is a dial you set at answer time, with no retraining. Zero means always retrieve, which the paper uses for ASQA because it needs citations, and most tasks use point two. You would lower it for factual tasks and raise it for open-ended ones like essays.

Slide 22, The model retrieves passages and writes candidates (0:30):

Steps two and three. The retriever searches with the input and the previous sentence and returns five passages by default. The generator handles each passage separately. It judges the passage, writes a sentence from it, and checks whether the passage supports that sentence. In the middle row, the model itself marks the baby-names passage irrelevant, and that passage can only spoil its own candidate.

Slide 23, Each critique token becomes a score (0:30):

Step four. To compare candidates we need numbers, so each critique token becomes a score from the probabilities the model gave its values. IsRel is the share of probability on Relevant. IsSup is similar, but partial support counts half. IsUse is an average rating from minus one to one. So every score comes straight from probabilities the model already produced, with no extra model.

Slide 24, Candidates are ranked by score (0:27):

Step five. A candidate's final score adds how likely the sentence is to the weighted critique scores. By default relevance and support get weight one and usefulness gets a half. You can change these at answer time. Raising the weight on IsSup favors well-supported output, and a hard rule can drop any sentence marked no support. Neither needs retraining.

Slide 25, How the final answer is chosen (0:19):

But the best sentence now can lead to a worse answer later. So the model keeps the two best partial answers after each sentence and returns the best complete one. This is segment-level beam search. Inside a sentence, words are chosen greedily.

## Speaker 6: Training vs inference, setup, and first results (slides 26 to 31, 2:32)

Slide 26, Training vs inference (0:56):

Now let's put the two phases side by side, because they are easy to mix up. When labeling, the answer already exists, so the critic can read a sentence first and then decide if it needed evidence. When answering, the sentence does not exist yet, so the generator has to decide before writing it, and that is exactly what its training teaches. The search differs for the same reason. Labeling can search with the sentence itself, while answering only has the previous sentence. A training text holds one passage per sentence, so K are judged and one is inserted, while answering tries all K and keeps the best. And labeling produces text to train on, while answering produces the final answer.

Slide 27, What are the experimental findings? (0:03):

So how well does it work?

Slide 28, Six tasks and how each is scored (0:26):

The paper tests six tasks, all zero-shot. PubHealth and ARC-Challenge pick from fixed options and use accuracy. PopQA and TriviaQA answer factual questions, and an answer counts if it contains a gold answer. Biographies use FactScore, the share of facts backed by a knowledge source. And ASQA needs long answers with citations, so it has five metrics.

Slide 29, Baselines (0:27):

The baselines isolate where any gain comes from. No-retrieval models test whether evidence and critique beat larger or better-tuned models. Standard RAG tests whether critique adds anything to retrieval. The key control is Llama2-FT, trained on the same pairs before labeling and retrieving only at test time. If it matched Self-RAG, the gain would come from the data alone.

Slide 30, Self-RAG vs open models (0:18):

Against open models, Self-RAG 13B beats the best baseline on all five measures, most clearly on ARC-Challenge and PubHealth. On biographies the smaller 7B model is even ahead, which the paper attributes to its shorter, more precisely grounded outputs.

Slide 31, Self-RAG vs open models on ASQA (0:22):

ASQA has five metrics, and the gray text says what each one means. Citations are the big win. Citation precision jumps from about twenty to seventy, and recall nearly doubles. Correctness barely moves, so the gain here is in checkable output rather than in more correct answers.

## Speaker 7: ChatGPT, Llama2-FT, and ablations (slides 32 to 36, 2:26)

Slide 32, Self-RAG vs ChatGPT (0:23):

Against ChatGPT the picture is mixed. Taking the better of ChatGPT with and without retrieval, Self-RAG wins four of ten measures, namely biographies, citation precision, PopQA, and PubHealth. ChatGPT wins the other six. Five of the ten rows come from ASQA, so that one task weighs heavily in this count.

Slide 33, Self-RAG vs Llama2-FT (0:29):

So is the gain just the data? Llama2-FT trains on the same pairs before labeling and retrieves only at test time. It falls behind on eight of ten measures, most of all on citations, and the paper reports no significant gain for it when data grows from fifty to a hundred and fifty thousand pairs. So the gain is not just the data.

Slide 34, Removing parts during training (0:37):

An ablation removes one part to see how much it matters. These three 7B models train on the same fifty thousand pairs, and only their training text differs. The full model sees passages and tokens. No retriever sees plain pairs, which is how Llama2-FT is trained, just on less data. No critic always gets the top passage but no tokens. With passages but no tokens, ASQA falls from thirty-two to eighteen. With neither, PubHealth falls from about seventy-four to sixty-eight.

Slide 35, Removing parts while answering (0:29):

These use the same trained model and change only how it answers. If it never retrieves, PopQA nearly halves while PubHealth barely moves. Retrieving only when the model actually writes Yes, instead of using the threshold, also cuts PopQA sharply. Using only the top passage, as standard RAG does, lowers PopQA and ASQA, and ranking without the support score also hurts ASQA.

Slide 36, Effect of the test-time settings (0:28):

These are the dials from the method section, tested directly. On ASQA, a larger IsSup weight raises citation precision by about one point but lowers fluency by about ten, because longer, more fluent text often has more claims the citations do not fully support. Raising the retrieval threshold costs PopQA more than half its normalized accuracy and leaves PubHealth almost unchanged.

## Speaker 8: Grounding, limitations, impact, and close (slides 37 to 44, 2:25)

Slide 37, Are the answers grounded? (0:23):

Finally, are the answers grounded? In a small human study, annotators judged most answers plausible and supported and most tokens correct, with short answers doing better than biographies. And only two percent of Self-RAG's correct answers are missing from its passages, against fifteen to twenty percent for other models.

Slide 38, What are its limitations and impact? (0:04):

So how far can we trust all this?

Slide 39, Strengths and assumptions (0:36):

The method has real strengths, and each rests on an assumption. No critic runs during training because critique is added offline, but only twenty GPT-4 labels per type were checked by hand. One model does everything, but the critic that teaches it matches GPT-4 on only eighty percent of relevance labels. Bad passages can be dropped only if the search finds good ones. And tuning assumes token probabilities work as scores, which the paper checks only through final metrics.

Slide 40, Limitations and improvements (0:25):

The main limitations are practical. The paper never reports how long answering takes, retrieval is weaker on open-ended instructions, and one sentence can mix supported and unsupported facts. We would measure the cost and stop a branch once the model writes Irrelevant, train the retriever on those instructions, and judge units smaller than a sentence.

Slide 41, Are the experiments sufficient? (0:30):

The experiments are broad, but there are gaps. There are no significance tests, the human study has only fifty samples, and on ASQA an output length limit is stated only for the baselines. We also checked the claims. ChatGPT wins TriviaQA and ARC-Challenge despite the abstract, and reading the released code we found over fifty issues, including thirteen crashes. None of this undoes the core idea.

Slide 42, Impact since publication (0:19):

The paper was an oral at ICLR 2024, which the authors list as top one percent, and OpenAlex shows over a thousand citations. Corrective RAG is built on top of standard RAG and Self-RAG, and the code and models are public.

Slide 43, Team Contribution (0:03):

All eight of us contributed equally.

Slide 44, Thank you (0:05):

Thank you. We would be happy to take your questions.

## Likely questions

Why not label all the data with GPT-4 directly? (slides 15 and 16)

The paper says relying on a proprietary model raises API costs and hurts reproducibility. GPT-4 labels only a sample, between about four and twenty thousand examples per token type, and the critic labels the rest.

What exactly does GPT-4 read? (slide 16)

An instruction for that token type, a few demonstrations, and one example. The example holds whatever that token needs, as the table on slide 16 shows.

If the generator learns the tokens, why do we need the critic? (slides 15 and 19)

The critic only creates the training data. Once the generator has learned to write the tokens, no critic runs during its training or at test time.

How many passages does the critic judge for one sentence? (slide 18)

The top K. Section 3.2.2 says the critic predicts IsRel for each passage and IsSup for each relevant one. The paper does not give K for labeling, and the released README uses 10.

Is labeling more involved than slide 18 shows? (slides 18 and 49)

Slightly. Slide 18 follows Section 3.2.2, and the paper's appendix adds two details. The critic first reads the input alone, and if it says No, the text gets no passages and only IsUse is added. Otherwise the retriever searches once with the input and the entire output, and the critic reads that initial passage when it writes Retrieve for each sentence. The initial passage is not added to the text. In the released code it is the top result of that search.

Why not check Continue against the passages already in the text? (slide 18)

That would match what Continue means, and it is a cleaner design. The paper does not explain its choice. Our guess is that one fixed passage lets every Retrieve decision run in a single batch, while checking earlier passages would make each decision wait for the passage chosen for the previous sentence.

Can the first sentence get Continue? (slide 18)

The appendix has the critic read the initial passage for every sentence, so in principle yes. But no passage is in the text yet, so there is nothing to continue from. The released code avoids this case. The first sentence reuses the Yes from the check on the input alone, so it always gets its own search, and any text that starts with Continue is dropped.

Why are retrieved passages masked in the loss? (slide 19)

The paper states that it masks them but does not give a reason. Our reading is that the passage comes from the retriever at test time, so there is nothing useful in teaching the model to predict it.

Why this loss function? (slides 11, 17, and 19)

Reflection tokens are just extra entries in the vocabulary, so the standard next-token objective trains them along with the text. No separate critic or reward model has to run.

What does Retrieve = Continue do? (slides 11 and 18)

It keeps using the passage retrieved earlier, because one passage can support several sentences, so no new retrieval happens. In the appendix procedure the critic judges this against the initial passage, while in the finished text it refers to the passage already there. The paper does not discuss that gap.

What happens if every passage is irrelevant? (slides 18 and 22)

During data creation, one passage is picked at random. When answering, the model writes IsRel = Irrelevant, which the paper says signals that the answer may not be grounded.

What is the difference between the threshold and a hard constraint? (slides 21 and 35)

The threshold retrieves when Yes is likely enough compared with No. The hard constraint in the ablation retrieves only when the model actually writes Retrieve = Yes.

Which token probabilities does the threshold normalize over? (slide 21)

The paper is inconsistent. Section 3.3 normalizes over all Retrieve tokens, while Appendix A.3 uses only Yes and No. The slide uses the appendix version, so Continue is left out of the ratio.

What does a threshold of zero mean? (slide 21)

The ratio is always above zero, so the model effectively retrieves every time. The paper uses this for ASQA because it needs citations.

What settings did GPT-4 use? (slide 16)

Temperature 1 and at most 200 output tokens. Outputs that broke the format were discarded.

Is IsUse scored at every sentence or once per answer? (slides 23 and 50)

The paper is inconsistent. Algorithm 1 and Equation 4 score it for every candidate, while Appendix A.1 gives it once per output. The worked example on slide 50 follows Equation 4.

How do the fixed-option tasks pick an answer? (slides 25 and 28)

On PubHealth and ARC-Challenge the model does not return its single best output. The paper aggregates the scores for each option and selects the option with the highest score.

How can a sentence probability be 0.80? (slide 50)

Those numbers are made up to show the arithmetic. The paper writes a sentence probability in Equation 3 without saying how it is computed. In the released code it is the exponential of the mean token log-probability, and only when a flag is set, which the long-form script does not forward.

Why is there no math for the metrics? (slide 28)

Each metric is one idea for this talk. The paper only names most of them and relies on the benchmarks that define them, so the slides give the plain meaning of each.

Is the paper's definition of citation precision standard? (slides 6, 31, and 57)

No. The paper says citation precision checks whether a claim is fully supported by the evidence it cites. In the benchmark the paper uses, that is citation recall, and precision looks for irrelevant citations. Our slides use the benchmark's meaning.

Why do only the baselines have a token cap on ASQA? (slide 41)

The paper caps baselines at 100 tokens because long outputs inflated str-em. It gives no cap for Self-RAG there, and the README command allows 300 tokens per segment with a maximum depth of 7, so the comparison may not be equal. On the closed-set tasks the paper states 50 only for baselines, and the README also uses 50 for Self-RAG. On biographies, baselines get 500 tokens and Self-RAG gets 200 per segment.

Is No retriever the same as Llama2-FT? (slides 33 and 34)

The paper describes them alike but never equates them. Both train on plain input-output pairs. No retriever uses 50k pairs and Llama2-FT uses all the data. Llama2-FT retrieves at test time, and the paper does not say how No retriever answers.

How do No retriever and No critic get passages at test time? (slide 34)

The paper does not say, and these models cannot write Retrieve tokens. Their PopQA scores are far above the full model with retrieval disabled, which suggests they do receive passages.

What is Alpaca 30B? (slide 37)

That is the name the paper prints, but it lists only Alpaca 7B and 13B as evaluated, so it is probably a typo.

Is the ChatGPT comparison fair? (slides 32 and 41)

It is mixed, which is why we show the better of ChatGPT with and without retrieval. Against ChatGPT without retrieval, Self-RAG also wins on ROUGE and MAUVE. The paper also adjusts instructions per model, and on ASQA only the baselines are told to cite.

Why do the analysis figures disagree with the main table? (slides 36 and 52)

The paper does not say. ASQA citation precision is about 70 in Figure 3b and 66.9 in Table 2, and MAUVE is about 87 in Figure 3b at the default weight and 74.3 in Table 2. The paper does not say which evaluation set its analysis figures use.

Why is the 7B model sometimes more factual than 13B? (slide 30)

The paper says the smaller model tends to write precisely grounded but shorter outputs.

Why does a larger support weight lower fluency? (slides 36 and 52)

The paper says longer, more fluent text often contains more claims that the citations do not fully support.

How expensive is answering? (slides 22 and 40)

The paper does not report latency or a compute comparison. Each retrieved sentence means five candidates by default and a beam of two.

Does this need a special retriever? (slides 40 and 48)

No, it uses an off-the-shelf retriever, Contriever-MS MARCO. The authors note it has limited effectiveness for open-ended generation and leave adapting it for future work.

What is the biggest limitation? (slides 39 and 40)

Answering cost is not measured, and everything rests on GPT-4 labels that were spot-checked on only twenty examples per type.

What other training details does the paper give? (slide 48)

It uses DeepSpeed stage 3 for multi-GPU training, Bfloat16 precision, and FlashAttention. For inference it uses one or two Quadro RTX 6000 GPUs with 24GB of memory.

What did you find in the code? (slides 41 and 58)

We only read the code and did not run it. Thirteen issues would crash specific paths, such as every short-form run and critic training as committed. In the long-form script the usefulness score is always zero and the weights are fixed, the short-form threshold is inverted, and the training mask removes the loss between two passages. This is about the public code and does not show the reported numbers are wrong.

How has the paper influenced later work? (slide 42)

It was an oral at ICLR 2024, which the authors list as top 1%, and OpenAlex shows over a thousand citations as of November 2026. Corrective RAG adds a retrieval evaluator and is built on top of standard RAG and Self-RAG. The code and the 7B and 13B models are public, and IBM publishes a tutorial for building a Self-RAG agent with LangGraph.

## Evaluation coverage

| Rubric item | Slides |
|---|---|
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
