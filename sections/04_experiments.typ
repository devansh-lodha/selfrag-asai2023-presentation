#import "../lib.typ": *
#let slide = slide.with(sec: [Experiments])

#section-slide(4)[What are the experimental findings?]

#slide(title: [Six tasks and how each is scored])[
  #{
    show table: set text(size: 18pt)
    set table(inset: (x: 11pt, y: 8pt))
    let tk(name, detail) = { set par(leading: 0.4em); name; linebreak(); text(size: 15.5pt, fill: mid, detail) }
    table(
      columns: (auto, auto, 1fr),
      align: left + horizon,
      toprule,
      [*Task*], [*Metric*], [*What it measures*],
      midrule,
      tk[PubHealth, ARC-Challenge][pick from fixed options], [Accuracy], tight[Share of test items answered correctly],
      lightrule,
      tk[PopQA, TriviaQA][answer a factual question], [Accuracy], tight[Correct if the output contains a gold answer],
      lightrule,
      tk[Biographies][write a bio of a named person], [FactScore], tight[Share of facts backed by a knowledge source],
      lightrule,
      tk[ASQA][long answer with citations], [Five metrics], [Correctness, fluency, and citations],
      bottomrule,
    )
  }
  #v(0.1cm)
  #cap[Every test is zero-shot, with no demonstrations. A gold answer is a correct reference answer. Higher is better on every metric.]
]

#slide(title: [Baselines])[
  #texttable(
    heads: ([Group], [Models], [What it tests]),
    widths: (0.68fr, 1.2fr, 1.12fr),
    size: 17pt,
    y: 7pt,
    ([No retrieval], [Llama2, Alpaca, Llama2-chat, ChatGPT, CoVE], [Do evidence and critique beat larger or better-tuned models?]),
    ([Standard RAG], [Llama2, Alpaca, Llama2-chat, and ChatGPT with the top passages], [Does critique add anything to retrieval?]),
    ([Same pairs before labeling], [Llama2-FT 7B, which retrieves only at test time], [Is the gain only from the training pairs?]),
    ([Other systems], [Perplexity.ai, plus published results for SAIL 7B and Toolformer 6B], [How do other retrieval methods compare?]),
  )
  #v(0.1cm)
  #cap[Retrieval models get the top 5 passages from Wikipedia. Biographies and open QA add 5 from web search, and ASQA uses a provided set for every model.]
]

#slide(
  title: [Self-RAG vs open models],
  takeaway: [Self-RAG 13B beats the best open baseline on all five tasks.],
)[
  #diverge(
    unit: 0.95cm, neg: 0.4cm, pos: 7.6cm, size: 17pt, gutter: 0.24cm,
    heads: ([Self-RAG 13B], [7B], [Best open]),
    sides: (none, [Self-RAG ahead]),
    ([ARC-Challenge accuracy], 7.3, [73.1], [67.3], [65.8]),
    ([PubHealth accuracy], 5.3, [74.5], [72.4], [69.2]),
    ([PopQA accuracy], 4.0, [55.8], [54.9], [51.8]),
    ([TriviaQA accuracy], 2.4, [69.3], [66.4], [66.9]),
    ([Biography FactScore], 0.3, [80.2], [81.2], [79.9]),
  )
  #v(0.1cm)
  #cap[Bars are Self-RAG 13B minus the best of Llama2, Llama2-chat, Llama2-FT, Alpaca, and SAIL. \ The paper attributes 7B's lead on biographies to shorter, precisely grounded outputs.]
]

#slide(
  title: [Self-RAG vs open models on ASQA],
  takeaway: [Citation precision jumps from 19.8 to 70.3, and correctness barely moves.],
)[
  #let m(name, meaning) = { set par(leading: 0.2em); name; linebreak(); text(size: 13.5pt, fill: mid, meaning) }
  #diverge(
    unit: 0.105cm, neg: 0.45cm, pos: 5.4cm, size: 16.5pt, gutter: 0.2cm, rowh: 0.95cm,
    heads: ([Self-RAG 13B], [7B], [Best open]),
    sides: (none, [Self-RAG ahead]),
    (m[Citation precision][citations that support their statement], 50.5, [70.3], [66.9], [19.8]),
    (m[Citation recall][statements fully supported by their citations], 35.2, [71.3], [67.8], [36.1]),
    (m[MAUVE][fluency, closeness to human-written text], 1.0, [71.6], [74.3], [70.6]),
    (m[ROUGE][correctness, word overlap with a reference answer], 0.3, [37.0], [35.7], [36.7]),
    (m[str-em][correctness, gold answers found word for word], -3.1, [31.7], [30.0], [34.8]),
  )
  #v(0.1cm)
  #cap[Bars are Self-RAG 13B minus the best of Llama2, Llama2-chat, Llama2-FT, and Alpaca.]
]

#slide(
  title: [Self-RAG vs ChatGPT],
  takeaway: [Self-RAG 13B wins 4 of 10 measures, and ChatGPT wins the other 6.],
)[
  #diverge(
    unit: 0.42cm, neg: 4.4cm, pos: 4.4cm, size: 15.5pt, gutter: 0.17cm,
    heads: ([Self-RAG 13B], [ChatGPT]),
    sides: ([ChatGPT ahead], [Self-RAG ahead]),
    ([Biography FactScore], 8.4, [80.2], [71.8]),
    ([ASQA citation precision], 5.2, [70.3], [65.1]),
    ([PopQA accuracy], 5.0, [55.8], [50.8]),
    ([PubHealth accuracy], 4.4, [74.5], [70.1]),
    ([ARC-Challenge accuracy], -2.2, [73.1], [75.3]),
    ([ASQA ROUGE], -2.9, [37.0], [39.9]),
    ([TriviaQA accuracy], -5.0, [69.3], [74.3]),
    ([ASQA citation recall], -5.3, [71.3], [76.6]),
    ([ASQA MAUVE], -8.1, [71.6], [79.7]),
    ([ASQA str-em], -9.0, [31.7], [40.7]),
  )
  #v(0.1cm)
  #cap[Self-RAG 13B minus the better of ChatGPT with and without retrieval. Five of the ten rows are ASQA.]
]

#slide(
  title: [Self-RAG vs Llama2-FT],
  takeaway: [Trained on the same pairs before labeling, Llama2-FT trails on 8 of 10 measures.],
)[
  #diverge(
    unit: 0.135cm, neg: 0.4cm, pos: 8.6cm, size: 15.5pt, gutter: 0.13cm,
    heads: ([Self-RAG 7B], [Llama2-FT 7B]),
    sides: (none, [Self-RAG ahead]),
    ([ASQA citation precision], 61.9, [66.9], [5.0]),
    ([ASQA citation recall], 60.3, [67.8], [7.5]),
    ([ASQA MAUVE], 23.1, [74.3], [51.2]),
    ([TriviaQA accuracy], 9.1, [66.4], [57.3]),
    ([PubHealth accuracy], 8.1, [72.4], [64.3]),
    ([PopQA accuracy], 6.2, [54.9], [48.7]),
    ([Biography FactScore], 3.0, [81.2], [78.2]),
    ([ARC-Challenge accuracy], 1.5, [67.3], [65.8]),
    ([ASQA ROUGE], -0.1, [35.7], [35.8]),
    ([ASQA str-em], -1.0, [30.0], [31.0]),
  )
  #v(0.1cm)
  #cap[Llama2-FT trains on the plain pairs and retrieves only at test time. The paper reports no significant gain for it from 50k to 150k pairs, unlike Self-RAG.]
]

// The text each training variant sees, cell by cell. A dashed cell is missing from that variant.
#let trainrow(has) = {
  let cells = (
    ([Retrieve], 2cm, false),
    ([passage], 1.9cm, true),
    ([IsRel], 1.4cm, false),
    ([$y_t$], 0.9cm, false),
    ([IsSup], 1.4cm, false),
    ([IsUse], 1.4cm, false),
  )
  grid(
    columns: cells.map(c => c.at(1)),
    column-gutter: 3pt,
    ..cells.zip(has).map(((c, on)) => box(
      width: 100%,
      height: 0.92cm,
      radius: 2pt,
      fill: if on and c.at(2) { wash } else { none },
      stroke: if on { 0.8pt + luma(45) } else { (paint: luma(190), thickness: 0.7pt, dash: "dashed") },
      align(center + horizon, text(size: 14.5pt, fill: if c.at(2) { luma(90) } else { ink }, if on { c.at(0) })),
    )),
  )
}

#slide(
  title: [Removing parts during training],
  takeaway: [Passages without tokens hurt ASQA most, and plain pairs hurt PubHealth most.],
)[
  #let hd(body) = text(size: 18pt, weight: "bold", fill: ink, body)
  #let name(a, b) = { set par(leading: 0.2em); text(size: 18pt, a); linebreak(); text(size: 13.5pt, fill: mid, b) }
  #let val(v) = text(size: 17.5pt, f1(v))
  #grid(
    columns: (auto, auto, auto, auto, auto),
    column-gutter: 0.48cm,
    row-gutter: 0.42cm,
    align: (right + horizon, left + horizon, center + horizon, center + horizon, center + horizon),
    [], hd[The text each model trains on], hd[PopQA], hd[PubHealth], hd[ASQA str-em],
    name[Full model][passages and tokens], trainrow((true, true, true, true, true, true)), val(45.5), val(73.5), val(32.1),
    name[No retriever][plain pairs, like Llama2-FT], trainrow((false, false, false, true, false, false)), val(43.6), val(67.8), val(31.0),
    name[No critic][top passage, no tokens], trainrow((false, true, false, true, false, false)), val(42.6), val(72.0), val(18.1),
  )
  #v(0.1cm)
  #cap[Three Llama2 7B models on the same 50k pairs. A dashed cell is missing from that model's training text. \ PopQA and PubHealth report accuracy, and ASQA uses 150 sampled questions.]
]

// One row per variant, bars aligned across the three tasks.
// Each row: (name, what changed, then one value per task).
#let ablation(gutter: 0.3cm, ..rows) = {
  let w = 4.2cm
  let hd(body) = text(size: 18pt, weight: "bold", fill: ink, body)
  let cell(v, dark) = if v == none {
    text(size: 16pt, fill: soft)[not run]
  } else {
    box(width: w, height: 0.52cm, fill: wash, radius: 1.5pt,
      place(left, box(width: w * v / 80, height: 0.52cm, fill: if dark { bar-strong } else { luma(170) }, radius: 1.5pt)))
    h(0.24cm)
    text(size: 17pt, weight: "bold", f1(v))
  }
  grid(
    columns: (auto, auto, auto, auto),
    column-gutter: 0.75cm,
    row-gutter: gutter,
    align: (right + horizon, left + horizon, left + horizon, left + horizon),
    [], hd[PopQA], hd[PubHealth], hd[ASQA str-em],
    ..rows.pos().enumerate().map(((i, r)) => (
      { set par(leading: 0.2em); text(size: 17.5pt, r.at(0)); linebreak(); text(size: 14pt, fill: mid, r.at(1)) },
      cell(r.at(2), i == 0), cell(r.at(3), i == 0), cell(r.at(4), i == 0),
    )).flatten(),
  )
}

#slide(
  title: [Removing parts while answering],
  takeaway: [Retrieving less hurts PopQA most, and PubHealth barely changes.],
)[
  #ablation(
    gutter: 0.3cm,
    ([Full model], [default settings], 45.5, 73.5, 32.1),
    ([No retrieval], [never retrieves], 24.7, 73.0, none),
    ([Hard constraint], [retrieves only if it writes Yes], 28.3, 72.6, none),
    ([Retrieve top 1], [always uses only the top passage], 41.8, 73.1, 28.6),
    ([Remove IsSup], [ranks without the support score], 44.1, 73.2, 30.6),
  )
  #v(0.1cm)
  #cap[The same Llama2 7B model trained on 50k pairs each time. Only the answering step changes.]
]

#slide(
  title: [Effect of the test-time settings],
  takeaway: [A larger IsSup weight barely helps. Retrieving less hurts PopQA far more than PubHealth.],
)[
  #texttable(
    heads: ([Setting], [Change], [Effect]),
    widths: (8.4cm, 6.3cm, 1fr),
    size: 18pt,
    y: 7.5pt,
    ([IsSup weight, on ASQA], [From about 0.1 to 2], [Citation precision rises about 1 point, and MAUVE falls about 10]),
    ([Threshold $delta$, on PopQA], [From 0 to 0.7], [Retrieval falls to about 20%, and normalized accuracy to about 0.45]),
    ([Threshold $delta$, on PubHealth], [From 0 to 0.7], [Retrieval falls to 0, and normalized accuracy stays about 0.99]),
  )
  #v(0.1cm)
  #cap[The IsSup row uses Self-RAG 7B. Normalized accuracy is 1 at $delta = 0$.]
]

#slide(
  title: [Are the answers grounded?],
  takeaway: [Annotators judged most answers plausible and supported, and only 2% of Self-RAG's correct answers are missing from its passages.],
)[
  #grid(
    columns: (1.04fr, 1fr),
    column-gutter: 0.8cm,
    align: center + top,
    stack(
      spacing: 0.36cm,
      head[Human evaluation (%)],
      groupbars(
        max: 100, width: 5.3cm, size: 16pt, gutter: 0.14cm,
        ([Plausible~and~supported], 92.5, 70.0),
        ([IsRel token correct], 95.0, 90.0),
        ([IsSup token correct], 90.0, 85.0),
      ),
      keys(([PopQA], bar-strong), ([Biographies], bar-weak)),
    ),
    stack(
      spacing: 0.42cm,
      head[Correct answers not in passages (%)],
      hbars(max: 100, width: 5.8cm, size: 16.5pt, fmt: v => [#v%], gutter: 0.36cm, ([Self-RAG 7B], 2), ([Alpaca], 15, luma(150)), ([Llama2-chat 13B], 18, luma(150)), ([Alpaca 30B], 20, luma(150))),
    ),
  )
  #v(0.1cm)
  #cap[Annotators reviewed 50 samples and left out answers the model marked Irrelevant or No support. Plausible means a reasonable, on-topic response, and supported means the evidence can verify the answer. The right chart uses TriviaQA and PopQA.]
]
