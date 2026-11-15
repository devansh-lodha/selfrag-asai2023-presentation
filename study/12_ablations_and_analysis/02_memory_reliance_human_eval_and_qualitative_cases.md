# Memory Reliance, Human Evaluation, and Qualitative Cases

## 1. Plain-English Intuition: Auditing How Self-RAG Uses Evidence and Self-Critiques

In Module 11 and Module 12 Part 1, we saw the aggregate benchmark scores and ablation curves of Self-RAG. To complete our study of the paper, we need to look inside individual model predictions and answer three questions (covered in Section 5.2 and Appendix C):
1. When a retrieval-augmented model answers a question correctly, did it actually read the answer from the retrieved passage (non-parametric memory), or did it ignore an irrelevant passage and guess from its pre-trained weights (parametric memory)?
2. When human annotators inspect Self-RAG's answers and its predicted `[Relevant]` and `[Fully supported]` tokens on short-form QA and long-form biographies, how often do humans agree with the model's self-reflection?
3. What do Self-RAG's failure modes look like in practice, and how do tokens like `[No support / Contradictory]` and `[Partially supported]` catch factual errors at the sentence level?

---

## 2. Parametric vs Non-Parametric Memory Reliance (Appendix C.1)

### 2.1 The Experimental Test for Non-Parametric Grounding

On open-domain QA benchmarks (TriviaQA and PopQA), the authors test how faithfully different retrieval-augmented models rely on the provided passages using a two-step procedure:
1. Collect the subset of test queries that a given model answered correctly.
2. For each correctly answered query, check whether the matched ground-truth answer string actually appears as a substring inside the retrieved passage(s) provided to the model.

Why is it a warning sign if a retrieval-augmented model frequently outputs correct answers that are NOT present in the retrieved passages?
- If the answer is absent from the retrieved passage, the passage was either irrelevant or incomplete, and the model ignored the passage to answer from its parametric weights $\theta$.
- While answering from parametric memory happens to work when the model remembers a famous entity, blindly generating confident answers when the retrieved passage is irrelevant is the exact behavior that produces unverified hallucinations whenever the model's parametric memory is wrong.

### 2.2 Comparing Unverified Parametric Reliance Across Models

Appendix C.1 reports, in prose only, the percentage of correct open-domain QA predictions where the ground-truth answer was absent from the provided passages. The same paragraph lists the evaluated models as Self-RAG 7B, Alpaca 7B, Alpaca 13B, and Llama2-Chat-13B, but then reports a number for "Alpaca 30B" (a model it never lists) and one for "Alpaca" without a size. The table keeps the paper's labels:

| Retrieval-Augmented Model (as Named in the Paper) | Share of Correct Answers Absent from Retrieved Passages | Share of Correct Answers Grounded in Retrieved Passages | Behavior When Retrieved Passages Are Irrelevant |
|---|---|---|---|
| "Alpaca 30B" (likely a typo for one of the evaluated Alpaca models) | $20\%$ | $80\%$ | Continues to generate a plausible answer from parametric weights |
| `Llama2-chat 13B` (with retrieval) | $18\%$ | $82\%$ | Continues to generate a plausible answer from parametric weights |
| "Alpaca" (size not stated) | $15\%$ | $85\%$ | Continues to generate a plausible answer from parametric weights |
| `Self-RAG 7B` | $2\%$ | $98\%$ | Predicts `[Irrelevant]` to warn that the passage does not support the answer |

In `Self-RAG 7B`, $98\%$ of correct answers come directly from the provided non-parametric evidence (only $2\%$ are absent from the passages), and whenever the retrieved passages are off-topic, Self-RAG explicitly emits `[Irrelevant]` to flag that the passage cannot ground the response.

---

## 3. Human Evaluation of Answer Support (`S&P`) and Reflection Token Accuracy (Section 5.2, Figure 4d & Table 6)

### 3.1 Human Evaluation Metrics and Quantitative Results (Figure 4d)

The paper says the authors "sampled 50 samples from PopQA and Bio results" (it does not say whether that is $50$ per task or $50$ in total, nor who the annotators were or how many there were) and asked human evaluators to grade three aspects:
1. Supported and Plausible (`S&P`): Among outputs where Self-RAG predicted `[Relevant]` and `[Fully supported]` or `[Partially supported]`, is the answer both plausible (a reasonable, on-topic response to the prompt) and supported (verifiable from the cited passage)?
2. `IsRel` agreement: Does the model's predicted `[Relevant]` / `[Irrelevant]` token match human judgment of passage relevance?
3. `IsSup` agreement: Does the model's predicted `[Fully supported]` / `[Partially supported]` / `[No support / Contradictory]` token match human judgment of passage entailment?

Figure 4d in Section 5.2 reports the human evaluation results:

| Evaluation Dimension | Short-Form `PopQA` (%) | Long-Form `Bio` (%) |
|---|---|---|
| Supported & Plausible (`S&P`) | $92.5\%$ | $70.0\%$ |
| `IsRel` Token Accuracy | $95.0\%$ | $90.0\%$ |
| `IsSup` Token Accuracy | $90.0\%$ | $85.0\%$ |

### 3.2 Walkthrough of the Four Human Evaluation Examples (Appendix Table 6)

Appendix Table 6 presents four concrete examples from this human evaluation study (two success cases and two instructive failure cases) that explain the numbers in Figure 4d:

1. Example 1 (PopQA Success Case):
   - Input: `"What genre is A Question and Answer Guide to Astronomy?"`
   - Model Output: `[Retrieve=Yes] <p>Reference [1]</p> [IsREL=Relevant] A Question and Answer Guide to Astronomy is a non-fiction book that falls under the genre of Astronomy and Cosmology. [IsSUP=Fully Supported] [IsUSE=5]`
   - Reference `[1]`: `"A Question and Answer Guide to Astronomy is a book about astronomy and cosmology, and is intended for a general audience..."`
   - Human Assessment: `S&P` is Supported and Plausible, and both `IsRel` and `IsSup` are correct.

2. Example 2 (PopQA Failure Case: Near-Miss Book Title Confusion):
   - Input: `"Who is the author of The Lie?"`
   - Model Output: `[Retrieve=Yes] <p>Reference [1]</p> [IsREL=Relevant] The author of The Lie is Sam Harris. [IsSUP=Fully Supported] [IsUSE=5]`
   - Reference `[1]`: `"Lying is a 2011 long-form essay book by American author and neuroscience expert Sam Harris..."`
   - Human Assessment: `S&P` is Plausible but not supported, and both `IsRel` and `IsSup` are incorrect!
   - Why this failed: The query asked for the author of a book titled *The Lie*, whereas the retriever returned an article about a different book with a nearly identical title, *Lying* (by Sam Harris). Fooled by the lexical similarity between *The Lie* and *Lying*, the model marked the passage `[Relevant]` and `[Fully supported]`.

3. Example 3 (Bio Success Case with `[Continue to Use Evidence]`):
   - Input: `"Tell me a bio about Ronaldinho"`
   - Model Output:
     `[Retrieve=Yes] <p>Reference [1]</p> [IsREL=Relevant] Ronaldinho is a retired professional footballer from Brazil. [IsSUP=Fully Supported] [Retrieve=Continue] He played as an attacking midfielder and was known for his technical skills, creativity, dribbling ability, and accuracy from free kicks [Retrieve=Continue] He won two FIFA World Player of the Year awards and a Ballon d'Or. [IsUSE=5]`
   - Reference `[1]`: Mentions that he is a Brazilian retired professional footballer, played as an attacking midfielder, won two FIFA World Player of the Year awards and a Ballon d'Or, and was renowned for technical skills, creativity, dribbling ability, and accuracy from free kicks.
   - Human Assessment: `S&P` is Plausible and fully supported, and both `IsRel` and `IsSup` are correct. Here, `[Continue to Use Evidence]` (`[Retrieve=Continue]`) allowed the model to extract three grounded sentences from a single rich Wikipedia paragraph without redundant retrieval calls.

4. Example 4 (Bio Failure Case: Over-Extending `[Continue to Use Evidence]`):
   - Input: `"Tell me a bio about Sergei Bodrov"`
   - Model Output:
     `[Retrieve=Yes] <p>Reference [1]</p> [IsREL=Relevant] Sergei Vladimirovich Bodrov (born June 28, 1948) is a Russian film director, screenwriter, and producer. [IsSUP=Fully Supported] [Retrieve=Continue] He was born in Moscow, Russia. [Retrieve=Continue] He is the president of the Russian Guild of Film Producers. [IsUSE=5]`
   - Reference `[1]`: `"Sergei Vladimirovich Bodrov (born June 28, 1948) is a Russian film director, screenwriter, and producer. In 2003 he was the President of the Jury at the 25th Moscow International Film Festival."`
   - Human Assessment: `IsRel` and `IsSup` on Sentence 1 are correct (Reference `[1]` is relevant and Sentence 1 is fully supported). However, overall `S&P` is rated Plausible but only partially supported!
   - Why this failed: Reference `[1]` is only two sentences long and does not mention where he was born (`"Moscow, Russia"`) or the `"Russian Guild of Film Producers"`. When the model emitted `[Retrieve=Continue]` for Sentences 2 and 3 instead of triggering a fresh `[Retrieval]` call (and without emitting a post-sentence `IsSup` check after those `[Retrieve=Continue]` sentences), it fell back on unverified parametric memory. This pattern illustrates why multi-sentence biographies can score lower on `S&P` ($70.0\%$) than PopQA answers ($92.5\%$); the paper does not break that gap down by cause.

---

## 4. Qualitative Case Studies of `Self-RAG 13B` (Appendix C.3, Table 7)

Appendix Table 7 showcases three outputs from `Self-RAG 13B` demonstrating multi-passage citation synthesis, automatic contradiction detection, and partial-support detection:

### 4.1 Case Study 1: Multi-Document Citation Synthesis on ASQA

- Input: `"When did sunday became the seventh day of the week"`
- Model Output:
  - Segment 1: `[Retrieve=Yes] <p>Reference [1]</p> [IsREL=Relevant] In 321 AD, the Roman Emperor Constantine the Great made Sunday a day of rest from labor, stating: "On the venerable Day of the Sun let the magistrates and people residing in cities rest, and let all workshops be closed". [IsSUP=Fully Supported]`
  - Segment 2: `[Retrieve=Yes] <p>Reference [2]</p> [IsREL=Relevant] The official adoption of Sunday as a day of rest by Constantine the Great in AD 321 made Sunday the seventh day of the week.[2] [IsSUP=Fully Supported] [IsUSE=5]`

Here, Segment 1 extracts the exact historical decree from Reference `[1]`, and Segment 2 retrieves a second passage (Reference `[2]`) to synthesize the conclusion and attach citation `[2]`.

### 4.2 Case Study 2: Catching Factual Contradictions with `[IsSUP=Contradictory]`

- Input: `"Tell me a bio about Rory Tapner."`
- Reference `[1]`: `"Rory Tapner is a UK businessman who, in September 2010 was appointed CEO of Coutts, the Wealth Management division of Royal Bank of Scotland Group. In February 2015 he stepped down as CEO when he was replaced in the role by Alison Rose"`
- Reference `[2]`: `"... Rory Tapner (born 30 September 1959) is a UK businessman ..."`
- Model Output:
  - Segment 1: `<p>Reference [1]</p> [IsREL=Relevant] Rory Tapner is a British businessman who has served as the Chief Executive Officer of Coutts, a private banking and wealth management company, since 2010. [IsSUP=Contradictory]`
  - Segment 2: `<p>Reference [2]</p> [IsREL=Relevant] Rory Tapner was born on 30 September 1959 in London, England. [IsSUP=Fully Supported]`

Look at what happened in Segment 1:
1. During greedy token generation, the model wrote `"has served as the Chief Executive Officer of Coutts ... since 2010"` (which implies he still serves as CEO today).
2. However, Reference `[1]` explicitly states that `"In February 2015 he stepped down as CEO"`!
3. Immediately after finishing the sentence, Self-RAG's self-reflection mechanism compared its own generated sentence against Reference `[1]` and predicted `[IsSUP=Contradictory]` (`[No support / Contradictory]`)!
4. As we saw in Module 10 Part 2, this `[No support / Contradictory]` token both lowers the branch's beam score and allows the `--ignore_cont` hard filter to automatically remove the contradictory sentence before returning the biography.

Look also at Segment 2, which the paper does not comment on: it claims Tapner was born `"in London, England"`, yet Reference `[2]` gives only the birth date, and the model still labels the segment `[IsSUP=Fully Supported]`. So even this showcased example contains a self-critique error, a useful reminder that `IsSup` predictions are not always right (consistent with the $85\%$ human agreement on biographies in Figure 4d).

### 4.3 Case Study 3: Distinguishing Partial Support with `[IsSUP=Partially Supported]`

- Input: `"Tell me a bio about G. Venugopal."`
- Reference `[1]`: `"G Venugopal (born 10 December 1960) is an Indian playback singer known for his work in Malayalam films. He started his singing career in the film 'Odaruthammaava Aalariyaam' (1984). Since then he has sung in more than 300 films; and has over 500 private albums to his credit..."`
- Reference `[2]`: `"... G Venugopal (born 10 December 1960) is an Indian playback singer known for his work in Malayalam films."`
- Model Output (Ranked 1):
  - Segment 1: `[Retrieve=Yes] <p>Reference [1]</p> [IsREL=Relevant] G. Venugopal is a popular playback singer in the Malayalam film industry. [Retrieve=Continue] He has sung over 300 songs in Malayalam, Tamil, and Hindi movies. [IsSUP=Partially Supported]`
  - Segment 2: `[Retrieve=Yes] <p>Reference [2]</p> G. Venugopal was born on December 10, 1960 in India. [IsSUP=Fully Supported]`

Look at why Segment 1 ends with `[IsSUP=Partially Supported]` while Segment 2 ends with `[IsSUP=Fully Supported]`:
- In Segment 1, Reference `[1]` verifies that G. Venugopal is a playback singer in Malayalam films, but it says he has sung in more than $300$ films (the output says $300$ songs), and it never mentions `"Tamil, and Hindi movies"`. Those details are not supported by Reference `[1]`, so `[Partially supported]` is a reasonable label. (Appendix C.3 gives a different reason, saying the statement lists "the name of the songs", which does not match the output shown in Table 7.)
- In Segment 2, the birth date `"December 10, 1960"` is stated in Reference `[2]`, and `"in India"` is inferred from `"Indian playback singer"`, so Self-RAG labels Segment 2 as `[Fully supported]`.
