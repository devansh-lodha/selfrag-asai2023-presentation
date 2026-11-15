#import "../lib.typ": *
#let slide = slide.with(sec: [Problem])

#section-slide(1)[What problem does the paper solve?]

#slide(
  title: [Language models get rare facts wrong],
  takeaway: [Going from 7B to 13B changes nothing, and even ChatGPT is mostly wrong.],
)[
  #hbars(max: 100, width: 15cm, size: 22pt, gutter: 0.48cm, ([Llama2 7B], 14.7), ([Llama2 13B], 14.7), ([Alpaca 13B], 24.4), ([ChatGPT], 29.3))
  #v(0.1cm)
  #cap[Accuracy (%) on PopQA, 1,399 questions about rare entities with under 100 Wikipedia views a month. \ 7B and 13B are sizes in billions of parameters, and Alpaca is Llama2 tuned on instructions.]
]

#slide(
  title: [Retrieval-Augmented Generation (RAG)],
  takeaway: [On PopQA, RAG lifts Alpaca 13B from 24.4% to 46.1%, \ and Llama2-chat from 20.0% to 51.8%.],
)[
  #dg(
    spacing: (0.75cm, 1cm),
    node((0, 0), t[Input $x$], name: <x>),
    node((1, 0), t[Retriever #R \ #sub[searches a large collection]], name: <r>),
    node((2, 0), t[$d_1, dots, d_K$ \ #sub[the top $K$ passages]], name: <d>),
    node((3, 0), t[Language model \ #sub[reads $x$ and all $K$ passages]], name: <lm>),
    node((4, 0), t[Output $y$], name: <y>),
    edge(<x>, <r>, "-|>"),
    edge(<r>, <d>, "-|>"),
    edge(<d>, <lm>, "-|>"),
    edge(<lm>, <y>, "-|>"),
  )
  #v(1cm)
  #say[RAG searches for passages related to the input $x$ and adds the top $K$ to the prompt. The language model then writes the output $y$ from them.]
]

#slide(title: [Three problems with standard RAG])[
  #texttable(
    heads: ([Problem], [Example]),
    widths: (1fr, 1.5fr),
    y: 15pt,
    ([It retrieves even when no facts are needed], [An essay about a summer vacation still fetches passages]),
    ([It can retrieve off-topic passages], [For "How did US states get their names?" one passage is "In Texas, Emma is a popular baby name."]),
    ([Nothing makes the output agree with the passages], [It says California was named after Christopher Columbus, while its passage says a fictional island in a Spanish book]),
  )
]

#slide(
  title: [Retrieval alone is not enough],
  takeaway: [Retrieval can lower accuracy, and open models rarely cite supporting evidence.],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 0.9cm,
    align: center + top,
    stack(
      spacing: 0.3cm,
      head[PubHealth accuracy (%)],
      groupbars(
        max: 100, width: 6.9cm, size: 15.5pt, fills: (bar-weak, bar-strong), gutter: 0.15cm, bold: 1,
        ([ChatGPT], 70.1, 54.7),
        ([Alpaca 13B], 55.5, 51.1),
        ([Alpaca 7B], 49.8, 40.2),
        ([Llama2 7B], 34.2, 30.0),
      ),
      keys(([no retrieval], bar-weak), ([with retrieval], bar-strong)),
    ),
    stack(
      spacing: 0.3cm,
      head[ASQA citation precision (%)],
      hbars(
        max: 100, width: 6.4cm, size: 15pt, fmt: f1, gutter: 0.15cm,
        ([Alpaca 13B], 2.0),
        ([Llama2 13B], 2.3),
        ([Llama2 7B], 2.9),
        ([Alpaca 7B], 5.5),
        ([Llama2-chat 13B], 19.8),
        ([ChatGPT], 65.1),
      ),
    ),
  )
  #v(0.1cm)
  #cap[PubHealth checks health claims, and these four of the six baselines lose accuracy with retrieval. ASQA needs long answers with citations. Citation precision is the share of citations that support their statement.]
]

#slide(title: [Existing work fixes only one piece])[
  #{
    show table: set text(size: 17pt)
    set table(inset: (x: 12pt, y: 6pt))
    let a(name, detail) = { set par(leading: 0.36em); name; linebreak(); text(size: 14pt, fill: mid, detail) }
    let area(body) = table.cell(rowspan: 3, align: left + horizon, text(weight: "bold", fill: ink, body))
    table(
      columns: (auto, auto, 1fr),
      align: left + horizon,
      toprule,
      [*Area*], [*Approach*], [*What it leaves open*],
      midrule,
      area[Retrieval], a[Always-retrieve training][SAIL], tight[It retrieves a fixed number of passages for every input],
      lightrule,
      a[Adaptive retrieval][Toolformer and others], tight[It often loses speed, robustness to bad passages, and attribution],
      lightrule,
      a[Passage filters][using a separate model], tight[An extra model runs at test time, and the output goes unchecked],
      midrule,
      area[Critique], a[Reinforcement learning][from human feedback, or RLHF], tight[Reward models run during training, and changing behavior often means retraining],
      lightrule,
      a[Control tokens][added at the start of generation], tight[The tokens never assess the output],
      lightrule,
      a[Repeated prompting][CoVE and others], tight[Generating and revising several times makes answering slow],
      bottomrule,
    )
  }
]

#slide(title: [Motivation])[
  #rowlist(
    widths: (0.42fr, 1.25fr),
    size: 20pt,
    y: 22pt,
    ([Why it matters], [Factual errors can spread misinformation and give incorrect, dangerous advice]),
    ([The goal], [One language model that decides when to retrieve, judges what it finds, and checks what it writes]),
    ([Also needed], [Low training overhead and adjustable at test time]),
  )
]
