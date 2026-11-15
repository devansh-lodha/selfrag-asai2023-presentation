#import "../lib.typ": *
#let slide = slide.with(sec: [Appendix])

#page(
  footer: page-footer(none),
  footer-descent: 0.35cm,
  {
    marker
    align(center + horizon, pad(bottom: 0.5cm, block(width: 78%, {
      text(size: 40pt, weight: "bold", fill: ink)[Appendix]
      v(0.75cm)
      rowlist(
        widths: (1fr, auto),
        size: 20pt,
        y: 10pt,
        ([Training details], [46 to 49]),
        ([Scoring and analysis], [50 to 54]),
        ([When tokens catch or miss mistakes], [55 and 56]),
        ([Mistakes in the paper and code], [57 and 58]),
      )
    })))
  },
)

#slide(
  title: [The training data has 145,619 input-output pairs],
  takeaway: [At this stage these pairs are plain text with no reflection tokens.],
)[
  #let dark = bar-strong
  #let lite = luma(150)
  #let bar(..rows) = hbars(max: 27000, width: 4.7cm, size: 16.5pt, fmt: commas, gutter: 0.3cm, ..rows)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 0.6cm,
    align: center + top,
    stack(spacing: 0.45cm, head[Instruction following (92,008)],
      bar(([GPT-4 Alpaca], 26168, dark), ([Stanford Alpaca], 25153, dark), ([FLAN-V2], 17817, dark), ([ShareGPT], 13406, dark), ([Open Assistant 1], 9464, dark))),
    stack(spacing: 0.45cm, head[Knowledge-intensive (53,611)],
      bar(([Wizard of Wikipedia], 17367, lite), ([Natural Questions], 15535, lite), ([FEVER], 9966, lite), ([OpenBookQA], 4699, lite), ([ASQA], 3897, lite), ([ARC-Easy], 2147, lite))),
  )
]

#slide(
  title: [How closely the critic matches GPT-4],
  takeaway: [Agreement is over 93% on Retrieve and IsSup, and lower on the other two.],
)[
  #groupbars(
    max: 100, width: 14cm, size: 15.5pt, gutter: 0.22cm,
    ([Retrieve], 93.8, 85.6),
    ([IsSup], 93.5, 73.1),
    ([IsRel], 80.2, 82.0),
    ([IsUse], 73.5, 72.1),
  )
  #v(0.35cm)
  #keys(([Llama2 7B (the critic used)], bar-strong), ([FLAN-3B (a smaller model)], bar-weak))
  #v(0.25cm)
  #cap[Share of GPT-4's labels each model reproduces on a held-out test set (%)]
]

#slide(title: [Training setup])[
  #rowlist(
    widths: (0.42fr, 1fr),
    size: 19pt,
    y: 9pt,
    ([Critic], [Llama2 7B]),
    ([Generator], [Llama2 7B or 13B]),
    ([Retriever], [Contriever-MS MARCO, off the shelf]),
    ([Training], [3 epochs, batch size 128]),
    ([Learning rate], [Peak of 2e-5, 3% warmup, then linear decay]),
    ([Maximum length], [2,048 tokens for 7B and 1,524 for 13B]),
    ([Hardware], [Four Nvidia A100 GPUs with 80GB of memory]),
  )
]

#slide(
  title: [Labeling a three-sentence output],
  takeaway: [Each Yes sentence gets its own passage, and the others get none.],
)[
  #{
    show table: set text(size: 17.5pt)
    set table(inset: (x: 14pt, y: 6.5pt))
    table(
      columns: (auto, auto, auto, 1fr),
      align: left + horizon,
      toprule,
      [*Part*], [*#C writes*], [*Search with*], [*Passage added to the text*],
      midrule,
      [Whole example], retY, [$x$ and all of $y$], tight[None. #C reads this initial passage to write Retrieve for each sentence.],
      lightrule,
      [Sentence $y_1$], retY, [$x$ and $y_1$], [The best result, just before $y_1$],
      lightrule,
      [Sentence $y_2$], retC, [No search], [None],
      lightrule,
      [Sentence $y_3$], retN, [No search], [None],
      bottomrule,
    )
  }
  #v(0.2cm)
  #text(size: 16pt, fill: mid)[Finished training text]
  #v(0.12cm)
  #text(size: 17pt)[#retY #pOpen $d$ #pClose #relY $y_1$ #supF #retC $y_2$ #retN $y_3$ #use5]
]

#slide(
  title: [A worked score example],
  takeaway: [The candidate built on $d_1$ has the highest score, so it is kept.],
)[
  #{
    show table: set text(size: 20pt, number-width: "tabular")
    set table(inset: (x: 18pt, y: 10pt))
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      toprule,
      [], [$p(y_t divides dots)$], [IsRel], [IsSup], [IsUse], [$f$],
      midrule,
      [$d_1$], [0.80], [0.90], [0.90], [0.90], [*3.05*],
      lightrule,
      [$d_3$], [0.80], [0.80], [0.55], [0.70], [2.50],
      lightrule,
      [$d_2$], [0.70], [0.05], [0.00], [−0.40], [0.55],
      bottomrule,
    )
  }
  #v(0.1cm)
  #cap[Illustrative numbers, where #box[$f$ for $d_1$ is 0.80 + 0.90 + 0.90 + 0.5 × 0.90 = 3.05]. \ $d_2$ has no IsSup token, so its support score is 0. IsUse runs from −1 to 1.]
]

#slide(
  title: [Standard RAG vs no retrieval],
  takeaway: [Retrieval helps when passages state the answer, and PubHealth mostly falls.],
)[
  #{
    show table: set text(size: 18pt)
    set table(inset: (x: 14pt, y: 8pt))
    let row(name, up, flat, down, change) = (name, marks(up, flat, down), change)
    table(
      columns: (1fr, auto, 1fr),
      align: left + horizon,
      toprule,
      [*Task*], [*Baselines*], [*Largest change*],
      midrule,
      ..row([PopQA], 6, 0, 0, [Llama2-chat 13B, 20.0 → 51.8]),
      lightrule,
      ..row([Biographies], 5, 0, 0, [Llama2 7B, 44.5 → 78.0]),
      lightrule,
      ..row([TriviaQA], 5, 0, 1, [Llama2 7B, 30.5 → 42.5]),
      lightrule,
      ..row([ARC-Challenge], 3, 1, 2, [Llama2 7B, 21.8 → 48.0]),
      lightrule,
      ..row([PubHealth], 2, 0, 4, [ChatGPT, 70.1 → 54.7]),
      bottomrule,
    )
  }
  #v(0.1cm)
  #cap[Each mark is one baseline going from no retrieval to RAG. ChatGPT has no biography score with retrieval. \ Filled means the score rose, hollow means it fell, and gray means no change.]
]

#slide(
  title: [Effect of the support weight],
  takeaway: [A larger IsSup weight barely raises precision and clearly lowers fluency.],
)[
  #let xs = (0.1, 0.5, 1, 2)
  #let ticks = ([≈ 0.1], [0.5], [1], [2])
  #grid(
    columns: (1fr, 1fr),
    align: center,
    stack(spacing: 0.65cm, head[Citation precision (about +1)],
      linechart(w: 9.6cm, h: 2.6cm, xs: xs, xticks: ticks, ymin: 64, ymax: 76, yticks: (64, 70, 76), labelw: 1cm, xtitle: [weight on IsSup],
        series: ((name: none, ys: (69.7, 70.0, 70.5, 70.5), dash: "solid"),))),
    stack(spacing: 0.65cm, head[MAUVE fluency (about −10)],
      linechart(w: 9.6cm, h: 2.6cm, xs: xs, xticks: ticks, ymin: 85, ymax: 97, yticks: (85, 91, 97), labelw: 1cm, xtitle: [weight on IsSup],
        series: ((name: none, ys: (96.4, 87.8, 86.5, 86.1), dash: "solid"),))),
  )
  #v(0.1cm)
  #cap[Self-RAG 7B on ASQA. Longer, more fluent text often contains more claims the citations do not fully support.]
]

#slide(
  title: [Effect of the retrieval threshold],
  takeaway: [Retrieving less costs little accuracy on PubHealth and a lot on PopQA.],
)[
  #let xs = (0, 0.1, 0.5, 0.7)
  #let ticks = ([0], [0.1], [0.5], [0.7])
  #let panel2(name, freq, acc, dyf: 0pt, dya: 0pt) = stack(
    spacing: 0.3cm,
    align(center, head[#name]),
    linechart(
      w: 9.2cm, h: 2.9cm, xs: xs, xticks: ticks, ymin: 0, ymax: 1,
      yticks: (0, 0.5, 1),
      labelw: 3.1cm,
      xtitle: [threshold $delta$],
      series: (
        (name: [accuracy], ys: acc, dash: "solid", dy: dya),
        (name: [frequency], ys: freq, dash: "dashed", dy: dyf),
      ),
    ),
  )
  #grid(
    columns: (1fr, 1fr),
    align: center,
    panel2([PopQA], (1, 1, 0.60, 0.20), (1, 0.89, 0.75, 0.45)),
    panel2([PubHealth], (1, 1, 0.16, 0), (1, 1, 0.985, 0.99)),
  )
  #v(0.1cm)
  #cap[Normalized accuracy starts at 1 for $delta = 0$, and frequency is how often the model retrieves. \ A higher $delta$ means less retrieval.]
]

#slide(
  title: [Effect of training data size],
  takeaway: [More data helps PopQA and ASQA citations most, and PubHealth stays flat.],
)[
  #linechart(
    w: 15cm, h: 3.4cm,
    xs: (0.699, 1, 1.301, 1.699, 2.176),
    xticks: ([5k], [10k], [20k], [50k], [150k]),
    ymin: 30, ymax: 80,
    yticks: (30, 40, 50, 60, 70, 80),
    labelw: 6.4cm,
    xtitle: [training pairs],
    series: (
      (name: [PubHealth accuracy], ys: (71.8, 70.9, 71.8, 73.5, 72.8), dash: "solid", dy: -7pt),
      (name: [ASQA citation precision], ys: (41.2, 51.6, 57.4, 58.9, 70.1), dash: "dashed", dy: 7pt),
      (name: [PopQA accuracy], ys: (33.9, 39.3, 40.8, 45.5, 54.9), dash: "solid"),
    ),
  )
  #v(0.2cm)
  #cap[Self-RAG 7B, scores in %.]
]

#let lab(body) = text(fill: mid, body)
#let cases(heads, ..rows) = {
  show table: set text(size: 16.5pt)
  set table(inset: (x: 12pt, y: 6.5pt))
  table(
    columns: (auto, 1fr, 1fr),
    align: left + horizon,
    toprule,
    [], ..heads,
    midrule,
    ..rows.pos().map(r => (lab(r.at(0)), tight(r.at(1)), tight(r.at(2)))).intersperse((lightrule,)).flatten(),
    bottomrule,
  )
}

#slide(
  title: [Tokens can catch the model's mistakes],
  takeaway: [The model writes these tokens itself, so it can flag its own mistakes.],
)[
  #cases(
    ([Asked for a bio of Rory Tapner], [Asked for a bio of G. Venugopal]),
    ([Passage], psg[... in September 2010 was appointed CEO of Coutts ... In February 2015 he stepped down as CEO ...], psg[... known for his work in Malayalam films ... he has sung in more than 300 films ...]),
    ([Model wrote], ["... has served as the Chief Executive Officer of Coutts ... since 2010."], ["He has sung over 300 songs in Malayalam, Tamil, and Hindi movies."]),
    ([Its token], supN, supP),
    ([Was the token right?], [Yes. The role ended in 2015.], [Yes. It says films, not songs, and omits Tamil and Hindi.]),
  )
]

#slide(
  title: [Tokens can also miss mistakes],
  takeaway: [Tokens make answers easier to verify, but someone must still verify them.],
)[
  #cases(
    ([Asked for the author of The Lie], [Same Tapner bio, next sentence]),
    ([Passage], psg[Lying is a 2011 long-form essay book by American author ... Sam Harris.], psg[Rory Tapner (born 30 September 1959) is a UK businessman ...]),
    ([Model wrote], ["The author of The Lie is Sam Harris."], ["Rory Tapner was born on 30 September 1959 in London, England."]),
    ([Its tokens], [#relY #supF], supF),
    ([Was the token right?], [No. _Lying_ is a different book.], [No. The printed passage does not say London.]),
  )
]

#slide(
  title: [Mistakes we noticed in the paper],
  takeaway: [None of these undo the core idea, and we also found smaller issues such as typos.],
)[
  #texttable(
    heads: ([The paper says], [What we found]),
    widths: (1fr, 1.05fr),
    size: 17pt,
    y: 7pt,
    ([Self-RAG outperforms ChatGPT on open-domain QA and reasoning (abstract)], [ChatGPT is higher on TriviaQA, 74.3 against 69.3, and on ARC-Challenge, 75.3 against 73.1]),
    ([Self-RAG is the best non-proprietary model on all tasks (Section 5.1)], [On ASQA str-em, Alpaca 13B with retrieval scores 34.8 against 31.7]),
    ([The critic agrees with GPT-4 above 90% on most token types (Section 3.2.1)], [The paper's appendix shows this for two of the four]),
    ([Citation precision checks that a claim is fully supported by its citations (Section 5.1)], [In the benchmark's definition, that is citation recall, and precision looks for irrelevant citations]),
  )
]

#slide(
  title: [Mistakes we noticed in the released code],
  takeaway: [We found over 50 issues, including 13 crashes, and cannot tell which version produced the paper's numbers.],
)[
  #texttable(
    heads: ([What the code does], [Effect]),
    widths: (1.25fr, 1fr),
    size: 18pt,
    y: 9.5pt,
    ([Long-form usefulness reads the wrong token list], [The score is always zero]),
    ([The long-form script ignores weight arguments], [Weights stay at 1, 1, and 0.5]),
    ([Short-form threshold uses raw log-probabilities], [The retrieval decision is inverted]),
    ([The loss mask runs from first passage to last], [Text between passages is not trained on]),
  )
  #v(0.1cm)
  #cap[These describe the public code and do not show that the reported numbers are wrong.]
]
