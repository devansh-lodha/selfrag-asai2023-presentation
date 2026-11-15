# Slide Layout Patterns for Touying

Practical layout patterns for structuring academic and technical slides in Touying.

## 1. Multi-Column Composition

Two equal columns:
```typst
#slide(composer: (1fr, 1fr))[
  #diagram(...)
][
  === Core Takeaways
  - First observation
  - Second observation
]
```

Asymmetric split (e.g., prominent visual + compact text):
```typst
#slide(composer: (3fr, 2fr))[
  #diagram(...)
][
  === Analysis
  - High capacity representation
  - Handles long dependencies
]
```

Three-column metric comparison:
```typst
#slide(composer: (1fr, 1fr, 1fr))[
  #block(stroke: 1pt + gray.lighten(50%), inset: 10pt, radius: 4pt)[
    *Model A* \
    BLEU: 25.9
  ]
][
  #block(stroke: 1pt + gray.lighten(50%), inset: 10pt, radius: 4pt)[
    *Model B* \
    BLEU: 28.4
  ]
][
  #block(stroke: 1.5pt + blue, fill: blue.lighten(90%), inset: 10pt, radius: 4pt)[
    *Seq2Seq (Ours)* \
    BLEU: 34.8
  ]
]
```

## 2. Card and Callout Containers

Highlight callout box:
```typst
#let callout(title, body, fill: rgb("#e8f4f8"), stroke: rgb("#1f4e79")) = {
  block(
    fill: fill,
    stroke: (left: 3pt + stroke),
    inset: (x: 12pt, y: 10pt),
    radius: (right: 4pt),
    width: 100%,
    [
      #text(weight: "bold", fill: stroke)[#title] \
      #body
    ]
  )
}
```

## 3. Diagram and Math Pairing

Pairing a Fletcher architecture diagram with mathematical state equations:
```typst
#slide(composer: (1fr, 1fr))[
  #diagram(
    node((0, 0), [Input $x_t$], shape: fletcher.shapes.pill),
    edge("-|>"),
    node((1, 0), [LSTM Layer], shape: fletcher.shapes.rect),
    edge("-|>"),
    node((2, 0), [Hidden $h_t$], shape: fletcher.shapes.pill),
  )
][
  === State Transition Equations
  $ i_t = sigma(W_(x i) x_t + W_(h i) h_(t-1) + b_i) $
  $ f_t = sigma(W_(x f) x_t + W_(h f) h_(t-1) + b_f) $
  $ o_t = sigma(W_(x o) x_t + W_(h o) h_(t-1) + b_o) $
  $ c_t = f_t dot c_(t-1) + i_t dot tanh(W_(x c) x_t + W_(h c) h_(t-1) + b_c) $
  $ h_t = o_t dot tanh(c_t) $
]
```

## 4. Progressive Animation on Layouts

Reveal points alongside a static architecture diagram:
```typst
#slide(
  repeat: 3,
  self => [
    #let (uncover, only, alternatives) = utils.methods(self)
    #grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [
        #diagram(...)
      ],
      [
        === Step-by-Step Processing
        #uncover("1-")[- Input sequence encoded into vector $v$]
        #uncover("2-")[- Decoder initialized with state $h_0 = v$]
        #uncover("3-")[- Target tokens generated sequentially until `<EOS>`]
      ]
    )
  ]
)
```
