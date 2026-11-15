#import "../lib.typ": *
#let slide = slide.with(sec: [Key contribution])

#section-slide(2)[What is the key contribution?]

#slide(title: [How Self-RAG works])[
  #let H = 2cm
  #dg(
    spacing: (0.78cm, 0.95cm),
    node-inset: 10pt,
    node((0, 0), [$x$], name: <x>),
    node((1, 0), t[Need evidence for \ the next sentence?], width: 5.7cm, height: H, fill: wash, name: <a>),
    node((2, 0), t[Retrieve \ $K$ passages], width: 3.6cm, height: H, name: <b>),
    node((3, 0), t[Write one candidate \ sentence per passage], width: 6.2cm, height: H, name: <c>),
    node((3, 1), t[Critique candidates \ and keep the best], width: 6.2cm, height: H, fill: wash, name: <d>),
    node((1, 1), t[Write the sentence \ without retrieval], width: 5.7cm, height: H, name: <n>),
    node((2, 1), text(size: 15pt, fill: mid)[#set par(leading: 0.45em); Repeated until \ the answer ends], stroke: none, inset: 2pt, name: <lab>),
    node(enclose: (<a>, <b>, <c>, <d>, <n>, <lab>), stroke: (paint: luma(110), dash: "dashed", thickness: 0.9pt), fill: none, inset: 15pt, snap: false, name: <loop>),
    node((5, 0.5), t[Rate \ the answer], width: 3.55cm, height: H, fill: wash, name: <g>),
    node((6, 0.5), [$y$], name: <y>),
    edge(<x>, <a>, "-|>"),
    edge(<a>, <b>, "-|>", lbl[yes], label-side: left, label-sep: 3pt),
    edge(<b>, <c>, "-|>"),
    edge(<c>, <d>, "-|>"),
    edge(<a>, <n>, "-|>", lbl[no], label-side: left),
    edge(<loop>, <g>, "-|>"),
    edge(<g>, <y>, "-|>"),
  )
  #v(0.7cm)
  #say[Self-Reflective RAG, or Self-RAG, writes one sentence at a time. \ In each shaded box it writes a reflection token that records its decision.]
]

#slide(
  title: [Four types of reflection tokens],
  takeaway: [A token is predicted like a word, so ordinary training teaches it, and its probability can serve as a score.],
)[
  #{
    show table: set text(size: 18pt)
    set table(inset: (x: 9pt, y: 9pt))
    table(
      columns: (auto, auto, 1fr, auto),
      align: left + horizon,
      toprule,
      [], [*Token*], [*The question it answers*], [*Values*],
      midrule,
      text(weight: "bold", fill: ink)[Retrieval], [Retrieve], [Does the next sentence need retrieval?], [Yes, No, Continue],
      midrule,
      table.cell(rowspan: 3, align: left + horizon, text(weight: "bold", fill: ink)[Critique]), [IsRel], [Is this passage useful for the input?], [Relevant, Irrelevant],
      lightrule,
      [IsSup], [Does this passage support the sentence?], [Fully, Partially, No support],
      lightrule,
      [IsUse], tight[Is the answer useful, whether or not it is true?], [5 (highest) to 1],
      bottomrule,
    )
  }
  #v(0.1cm)
  #cap[Continue means keep using the passage already retrieved.]
]

#slide(
  title: [Standard RAG vs Self-RAG],
  takeaway: [Each design choice answers one of the three problems with standard RAG.],
)[
  #texttable(
    heads: ([], [Standard RAG], [Self-RAG], [Why it helps]),
    widths: (0.82fr, 0.96fr, 1.08fr, 1.36fr),
    size: 18pt,
    y: 11pt,
    ([Retrieval], [Always, once, before writing], [On demand, for each sentence], [Tasks that need no facts keep the model's creativity]),
    ([Passages], [All $K$ together in one prompt], [One candidate per passage], [A bad passage affects only its own candidate]),
    ([Quality check], [None], [Relevance, support, usefulness], [The best candidate is chosen with no extra model]),
  )
]
