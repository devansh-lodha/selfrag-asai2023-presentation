#import "../lib.typ": *
#let slide = slide.with(sec: [Method])
#let trn(n) = [Training step #n of 4]
#let inf(n) = [Inference step #n of 5]

#section-slide(3)[How does the method work?]

#slide(
  title: [The training data we need],
  takeaway: [Training has to turn plain pairs into text like this, then teach a model to write it.],
)[
  #let lab(body) = text(size: 17.5pt, weight: "bold", fill: ink, body)
  #set text(size: 16.5pt)
  #set par(leading: 0.56em)
  #grid(
    columns: (auto, 1fr),
    align: (left + top, left + top),
    inset: (x: 11pt, y: 7pt),
    grid.hline(stroke: 1.1pt + luma(30)),
    lab[Input $x$], [How did US states get their names?],
    grid.hline(stroke: 0.4pt + luma(205)),
    lab[Before], [11 of 50 states' names come from persons. For instance, Louisiana was named after King Louis XIV, and Georgia was named after King~George~II.],
    grid.hline(stroke: 0.4pt + luma(205)),
    lab[After], [
      #retY #psg[Of the fifty states, eleven are named after an individual person.] #relY \
      11 of 50 states' names come from persons. #supF
      #v(0.05cm)
      #retY #psg[Louisiana: named in honor of Louis XIV of France.] #relY \
      For instance, Louisiana was named after King Louis XIV, and Georgia was named after King~George~II. #supP #use5
    ],
    grid.hline(stroke: 1.1pt + luma(30)),
  )
  #v(0.1cm)
  #cap[Shaded text is a retrieved passage. The second one says nothing about Georgia, so support is only partial.]
]

#slide(
  title: [Training phase],
  takeaway: [No existing data contains these tokens, so GPT-4 and the critic create them.],
)[
  #let w = 5.1cm
  #let H = 3.6cm
  #v(0.75cm)
  #dg(
    spacing: (0.95cm, 1cm),
    node-inset: 9pt,
    node((0, 0), stepcard(1, [GPT-4], width: w, height: H)[Labels a sample \ of the data], name: <a>),
    node((1, 0), stepcard(2, [Critic #C], width: w, height: H)[A Llama2 7B \ model that learns \ GPT-4's labels], name: <b>),
    node((2, 0), stepcard(3, [Labeling], width: w, height: H)[The critic and \ retriever label all \ 145,619 pairs], name: <c>),
    node((3, 0), stepcard(4, [Generator #M], width: w, height: H)[A Llama2 7B or 13B \ model that learns \ from the result], name: <d>),
    edge(<a>, <b>, "-|>"),
    edge(<b>, <c>, "-|>"),
    edge(<c>, <d>, "-|>"),
  )
  #v(0.1cm)
  #cap[Labeling every sentence by hand is too expensive, and using GPT-4 on everything raises API costs and hurts reproducibility. The 145,619 input-output pairs start as plain text.]
]

#slide(
  title: [GPT-4 labels a sample of the data],
  sec: trn(1),
  takeaway: [The authors checked 20 labels per type and agreed with at least 90%, except 80% on IsUse.],
)[
  #{
    show table: set text(size: 18pt)
    set table(inset: (x: 12pt, y: 8.5pt))
    table(
      columns: (auto, 1fr, auto, auto),
      align: (left + horizon, left + horizon, right + horizon, right + horizon),
      toprule,
      [*Token*], [*GPT-4 reads*], [*Labels collected*], [*Authors agreed*],
      midrule,
      [Retrieve], tight[Input alone, or input, earlier text, passage, and one sentence], [12,594], [95%],
      lightrule,
      [IsRel], [Input and passage], [19,317], [95%],
      lightrule,
      [IsSup], [Input, passage, and output], [11,181], [90%],
      lightrule,
      [IsUse], [Input and output], [3,831], [80%],
      bottomrule,
    )
  }
  #v(0.1cm)
  #cap[Each token type has its own prompt, with an instruction, a few demonstrations, and one sampled example.]
]

#slide(
  title: [The critic learns from GPT-4's labels],
  sec: trn(2),
  takeaway: [The critic matches GPT-4 on 93.8% of Retrieve and 93.5% of IsSup labels, but only 80.2% of IsRel and 73.5% of IsUse.],
)[
  #dg(
    spacing: (1.3cm, 1cm),
    node((0, 0), nb(8.9cm)[Input $x$, output $y$, and passage $d$ \ when the token needs one], name: <i>),
    node((1, 0), nb(3.9cm)[Critic #C \ #sub[starts as Llama2 7B]], name: <c>),
    node((2, 0), nb(6.6cm)[A predicted token, trained \ to match GPT-4's token $r$], name: <r>),
    edge(<i>, <c>, "-|>"),
    edge(<c>, <r>, "-|>"),
  )
  #v(0.6cm)
  $ max_(cal(C)) quad bb(E)_(((x, y), r) ~ cal(D)_"critic") log p_(#h(0.1em) cal(C)) (r divides x, y) $
  #v(0.5cm)
  #say[Adjust the critic so that GPT-4's token gets the highest probability, averaged over the labeled examples $cal(D)_"critic"$. This is ordinary training with one token as the target.]
]

#slide(title: [The critic labels every sentence], sec: trn(3))[
  #say(size: 19pt)[For each sentence, #C first writes Retrieve. After a Yes, #R searches with $x$ and $y_t$, \ and #C writes IsRel for each of the top $K$ passages and IsSup for each relevant one.]
  #v(0.2cm)
  #{
    show table: set text(size: 17.5pt)
    set table(inset: (x: 10pt, y: 9pt))
    let row(body) = text(fill: mid, body)
    table(
      columns: (1fr, auto, auto, auto, auto, auto),
      align: left + horizon,
      toprule,
      [*Case*], [*Decide*], [*Evidence*], [*Judge evidence*], [*Write*], [*Verify*],
      midrule,
      row[Good passage], retY, [#pOpen $d$ #pClose], relY, [$y_t$], supA,
      lightrule,
      row[Off-topic passage], retY, [#pOpen $d$ #pClose], relN, [$y_t$], nil,
      lightrule,
      row[Passage reused], retC, nil, nil, [$y_t$], nil,
      lightrule,
      row[No facts needed], retN, nil, nil, [$y_t$], nil,
      bottomrule,
    )
  }
  #v(0.1cm)
  #cap[Only one of the $K$ passages goes into the text. It is relevant, at least partially supported, and has the top retrieval score. If none qualifies, one is picked at random. One IsUse token closes the whole output.]
]

#slide(
  title: [The generator learns from the labeled data],
  sec: trn(4),
)[
  #tape(
    ([input $x$], false),
    ([Retrieve = Yes], true),
    ([\<p\> passage \</p\>], false),
    ([IsRel = Relevant], true),
    ([sentence $y_t$], true),
    ([IsSup = Fully], true),
    ([IsUse = 5], true),
  )
  #v(0.75cm)
  $ max_(cal(M)) quad bb(E)_((x, y, r) ~ cal(D)_"gen") log p_(#h(0.1em) cal(M)) (y, r divides x) $
  #v(0.5cm)
  #say[Make the sentences and tokens $r$ in the labeled data $cal(D)_"gen"$ as likely as possible. \ Passages stay in the input but are masked from the loss.]
  #v(0.1cm)
  #cap[Critique was added to the data offline, so no critic runs during this training. Llama2 7B or 13B, 3 epochs, batch size 128.]
]

#slide(
  title: [Inference phase],
  takeaway: [Inference means answering a new question, using only #M and #R.],
)[
  #let w = 4.3cm
  #let H = 3.65cm
  #let card = stepcard.with(width: w, height: H, size: 16pt, lift: 1.02cm)
  #v(0.75cm)
  #dg(
    spacing: (0.6cm, 1cm),
    node-inset: 7pt,
    node((0, 0), card(1, [Generator #M])[Decides if the \ next sentence \ needs evidence], name: <a>),
    node((1, 0), card(2, [Retriever #R])[Searches with $x$ \ and the sentence \ before this one], name: <b>),
    node((2, 0), card(3, [Generator #M])[Writes one \ candidate \ per passage], name: <c>),
    node((3, 0), card(4, [Scoring])[Turns tokens \ into scores], name: <d>),
    node((4, 0), card(5, [Ranking])[Orders the \ candidates], name: <e>),
    edge(<a>, <b>, "-|>"),
    edge(<b>, <c>, "-|>"),
    edge(<c>, <d>, "-|>"),
    edge(<d>, <e>, "-|>"),
  )
  #v(0.1cm)
  #cap[These steps repeat for every sentence. If step 1 says no, the model writes the sentence directly. \ If the model writes Continue, it keeps the earlier passage.]
]

#slide(
  title: [The model decides whether it needs evidence],
  sec: inf(1),
  takeaway: [Changing $delta$ needs no retraining. ASQA uses 0 because it needs citations. Lower it for factual tasks and raise it for open-ended ones.],
)[
  $ "retrieve if" quad p("Yes") / (p("Yes") + p("No")) > delta $
  #v(0.4cm)
  #say[#M retrieves when the share of probability $p$ on Yes is above the threshold $delta$.]
  #v(0.65cm)
  #{
    let w = 21cm
    let tick(pos, above, below) = place(top + left, dx: w * pos - 3cm, box(width: 6cm, align(center, {
      set par(leading: 0.42em)
      text(size: 17pt, above)
      v(1.15cm)
      text(size: 15.5pt, fill: mid, below)
    })))
    box(width: w, height: 2.5cm, {
      place(top + left, dy: 0.98cm, line(length: w, stroke: 1.2pt + luma(70)))
      for v in (0, 0.2, 1) {
        place(top + left, dx: w * v - 0.5pt, dy: 0.76cm, line(angle: 90deg, length: 0.44cm, stroke: 1.2pt + luma(70)))
      }
      tick(0, [$delta = 0$], [always retrieves])
      tick(0.2, [$delta = 0.2$], [most tasks])
      tick(1, [$delta = 1$], [never retrieves])
    })
  }
]

#slide(
  title: [The model retrieves passages and writes candidates],
  sec: [Inference steps 2 and 3 of 5],
  takeaway: [Each candidate uses one passage, so a bad passage spoils only its own candidate.],
)[
  #{
    show table: set text(size: 16pt)
    set table(inset: (x: 8pt, y: 8pt))
    table(
      columns: (auto, 1.2fr, auto, 1.2fr, auto),
      align: left + horizon,
      toprule,
      [], [*Passage*], [*Judge evidence*], [*Candidate sentence*], [*Verify*],
      midrule,
      [$d_1$], tight(psg[Of the fifty states, eleven are named after an individual person.]), relY, tight[11 of 50 states' names come from persons.], supF,
      lightrule,
      [$d_2$], tight(psg[Popular names by states. In Texas, Emma is a popular baby name.]), relN, tight[Texas is named after a Native American tribe.], text(fill: mid)[no token],
      lightrule,
      [$d_3$], tight(psg[California was named after a fictional island in a Spanish book.]), relY, tight[California's name has its origins in a 16th-century novel.], supP,
      bottomrule,
    )
  }
]

#slide(title: [Each critique token becomes a score], sec: inf(4))[
  #say[Each value $v$ has a probability $p(v)$, and those give a score $s_t$.]
  #v(0.75cm)
  #{
    show table: set text(size: 18pt)
    show math.equation: set text(size: 19pt)
    set table(inset: (x: 14pt, y: 13pt))
    table(
      columns: (auto, auto, 1fr),
      align: (left + horizon, center + horizon, left + horizon),
      toprule,
      [*Token*], [*Score $s_t$*], [*What it means*],
      midrule,
      [IsRel], $display(p("Relevant") / (p("Relevant") + p("Irrelevant")))$, tight[The share of probability on Relevant, from 0 to 1],
      lightrule,
      [IsSup], $display((p("Fully") + 0.5 dot p("Partially")) / (p("Fully") + p("Partially") + p("No support")))$, tight[Partial support counts half, and the score runs from 0 to 1],
      lightrule,
      [IsUse], $display((sum_(v=1)^5 u_v p(v)) / (sum_(v=1)^5 p(v)))$, tight[The average rating, with $u$ counting ratings 1 to 5 as −1, −0.5, 0, 0.5, and 1],
      bottomrule,
    )
  }
]

#slide(
  title: [Candidates are ranked by score],
  sec: inf(5),
  takeaway: [A higher weight on IsSup favors well-supported output, and a hard filter can drop any sentence marked #supN#h(0.16em). Neither needs retraining.],
)[
  $ f(y_t, d) = p(y_t divides x, d, y_(<t)) + sum_G w^G s_t^G $
  #v(0.35cm)
  #say[A good candidate is a likely sentence with high critique scores.]
  #v(0.15cm)
  #cap[$y_(<t)$ means the sentences before $y_t$, and $s_t^G$ is the score of critique $G$. \ $G$ runs over IsRel, IsSup, and IsUse, with default weights $w^G$ of 1, 1, and 0.5.]
]

#slide(title: [How the final answer is chosen])[
  #grid(
    columns: (1fr, 1.2fr),
    column-gutter: 0.9cm,
    align: horizon,
    {
      let gone = (paint: luma(170), dash: "dashed")
      let kept(pos, name) = node(pos, [], shape: circle, radius: 6pt, fill: luma(40), stroke: none, name: name)
      let cut(pos, name) = node(pos, [], shape: circle, radius: 6pt, fill: none, stroke: gone, name: name)
      align(center, stack(
        spacing: 0.45cm,
        dg(
          spacing: (1.5cm, 0.3cm),
          node-inset: 9pt,
          node((0, 2), [$y_(<t)$], name: <root>),
          node((1, 0), [3.05], stroke: 1.2pt + luma(30), name: <a>),
          node((1, 2), [2.50], stroke: 1.2pt + luma(30), name: <b>),
          node((1, 4), text(fill: luma(150))[0.55], stroke: gone, name: <c>),
          kept((2, -0.7), <a1>), kept((2, 0), <a2>), cut((2, 0.7), <a3>),
          cut((2, 1.3), <b1>), cut((2, 2), <b2>), cut((2, 2.7), <b3>),
          node((1, 5.3), note[sentence $t$], stroke: none),
          node((2, 5.3), note[sentence $t + 1$], stroke: none),
          edge(<root>, <a>, "-|>"),
          edge(<root>, <b>, "-|>"),
          edge(<root>, <c>, "-|>", stroke: gone),
          edge(<a>, <a1>, "-|>"), edge(<a>, <a2>, "-|>"), edge(<a>, <a3>, "-|>", stroke: gone),
          edge(<b>, <b1>, "-|>", stroke: gone), edge(<b>, <b2>, "-|>", stroke: gone), edge(<b>, <b3>, "-|>", stroke: gone),
        ),
        note[Numbers are candidate scores $f$. Solid paths are kept and dashed ones dropped.],
      ))
    },
    lblock(width: 100%)[
      #set text(size: 19pt)
      #set list(spacing: 0.8em)
      - #tight[The best sentence now can lead to a worse answer later.]
      - #tight[The #box[$B = 2$] best partial answers survive each sentence. This is segment-level beam search.]
      - #tight[Each one grows by one candidate per passage, and the best complete answer wins.]
    ],
  )
  #v(0.1cm)
  #cap[Words within a sentence are chosen greedily.]
]

#slide(
  title: [Training vs inference],
  takeaway: [The generator learns to predict the critic's decision before the sentence exists.],
)[
  #texttable(
    heads: ([], [Labeling training data], [Answering a new question]),
    widths: (7.7fr, 9.2fr, 9.9fr),
    size: 18pt,
    y: 8pt,
    ([The answer], [Already known], [Written one sentence at a time]),
    ([Tokens written by], [Critic #C], [Generator #M]),
    ([Retrieve decided], [After reading the sentence], [Before the sentence exists]),
    ([Search query], [$x$ and the sentence itself], [$x$ and the previous sentence]),
    ([Passages per sentence], [$K$ judged, one inserted], [$K$, each with its own candidate]),
    ([Result], [Text to train on], [The final answer]),
  )
]
