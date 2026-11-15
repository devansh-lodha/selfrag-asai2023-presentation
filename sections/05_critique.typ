#import "../lib.typ": *
#let slide = slide.with(sec: [Critical analysis])

#section-slide(5)[What are its limitations and impact?]

#slide(title: [Strengths and assumptions])[
  #texttable(
    heads: ([Strength], [What it assumes]),
    widths: (0.82fr, 1.25fr),
    size: 18pt,
    y: 13pt,
    ([No critic runs during training, because critique is added offline], [GPT-4's labels are good enough, though only 20 per type were checked by hand]),
    ([One model retrieves, writes, and critiques], [A 7B critic can teach it, though it matches GPT-4 on 80.2% of IsRel and 73.5% of IsUse]),
    ([Bad passages can be flagged and dropped], [The search returns useful passages, and the authors report weaker retrieval on open-ended instructions]),
    ([Behavior can be tuned without retraining], [Token probabilities work as scores, which the paper checks only through final metrics]),
  )
]

#slide(title: [Limitations and improvements])[
  #texttable(
    heads: ([Limitation], [What we would do]),
    widths: (1.02fr, 1.12fr),
    size: 19pt,
    y: 17pt,
    ([Answering cost is not reported], [Measure it, and stop a branch once the model writes Irrelevant]),
    ([Retrieval is weaker on open-ended instructions], [Train the retriever on those instructions, as the authors also suggest]),
    ([A sentence can mix supported and unsupported facts], [Judge units smaller than a sentence, which the framework allows]),
  )
]

#slide(
  title: [Are the experiments sufficient?],
  takeaway: [The coverage is broad, and none of this undoes the core idea.],
)[
  #grid(
    columns: (1fr, 1.42fr),
    column-gutter: 0.9cm,
    align: left + top,
    texttable(
      heads: ([What is missing],),
      widths: (1fr,),
      size: 18pt,
      y: 11pt,
      ([No variance or significance tests],),
      ([A human study of only 50 samples],),
      ([An output length limit stated only for the baselines on ASQA],),
    ),
    texttable(
      heads: ([What we found wrong],),
      widths: (1fr,),
      size: 18pt,
      y: 11pt,
      ([The abstract says Self-RAG beats ChatGPT on QA and reasoning, but ChatGPT wins TriviaQA and ARC-Challenge],),
      ([The paper says the critic tops 90% on most token types, but that holds for two of the four],),
      ([Reading the released code, we found over 50 issues, including 13 crashes],),
    ),
  )
]

#slide(title: [Impact since publication], sec: [Impact])[
  #rowlist(
    widths: (0.42fr, 1.25fr),
    y: 15pt,
    ([Recognition], [Oral at ICLR 2024, which the authors list as top 1%]),
    ([Citations], [Over 1,000 on OpenAlex as of November 2026]),
    ([Later research], [Corrective RAG adds a retrieval evaluator and is built on top of standard RAG and Self-RAG]),
    ([Practice], [The code and 7B and 13B models are public, and IBM publishes a Self-RAG agent tutorial using LangGraph]),
  )
]

#slide(title: [Team Contribution], sec: none)[
  #{
    show table: set text(size: 19pt)
    set table(inset: (x: 16pt, top: 6pt, bottom: 8pt))
    table(
      columns: (1fr, auto, auto, auto),
      align: (left, left, left, right),
      toprule,
      [*Student*], [*Roll number*], [*Contribution*], [*Score*],
      midrule,
      [Chavda Haarit Ravindrakumar], [23110077], [Equal], [1.0],
      lightrule,
      [Darpana Dharmesh Desai], [23110085], [Equal], [1.0],
      lightrule,
      [Devansh Lodha], [23110091], [Equal], [1.0],
      lightrule,
      [Jagtap Arnav Shrikant], [23110143], [Equal], [1.0],
      lightrule,
      [Keval Pintubhai Shah], [26120003], [Equal], [1.0],
      lightrule,
      [Ruddhi Pintubhai Shah], [26250038], [Equal], [1.0],
      lightrule,
      [Suhani], [24110358], [Equal], [1.0],
      lightrule,
      [Vedant Acharya], [23110010], [Equal], [1.0],
      bottomrule,
    )
  }
]

// Mirrors the title page.
#page(margin: (x: 0.95cm, y: 0.8cm), header: none, footer: none, {
  marker
  place(top + left, corner-text[CS 613: Natural Language Processing])
  place(top + right, corner-text[Indian Institute of Technology Gandhinagar])
  place(bottom + left, corner-text[Fall 2026])
  place(bottom + right, context corner-text(counter(page).display("1")))
  align(center + horizon, block(width: 84%, {
    set par(leading: 0.42em)
    text(size: 46pt, weight: "bold", fill: ink)[Thank you]
    v(0.3cm)
    text(size: 26pt, fill: luma(70))[Questions?]
    v(1.1cm)
    line(length: 5cm, stroke: 0.8pt + faint)
    v(0.9cm)
    text(size: 19pt, fill: mid)[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection]
    v(0.2cm)
    text(size: 18pt, fill: luma(80))[Group: Mixture of Experts]
  }))
})
