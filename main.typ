#import "lib.typ": *
#show: deck

// Title
#page(margin: (x: 0.95cm, y: 0.8cm), header: none, footer: none, {
  marker
  place(top + left, corner-text[CS 613: Natural Language Processing])
  place(top + right, corner-text[Indian Institute of Technology Gandhinagar])
  place(bottom + left, corner-text[Fall 2026])
  place(bottom + right, context corner-text(counter(page).display("1")))
  align(center + horizon, block(width: 84%, {
    set par(leading: 0.42em)
    text(size: 44pt, weight: "bold", fill: ink)[Self-RAG]
    v(0.15cm)
    text(size: 27pt, weight: "bold", fill: luma(60))[Learning to Retrieve, Generate, and Critique \ through Self-Reflection]
    v(0.9cm)
    text(size: 17pt, fill: mid)[Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi]
    v(0.1cm)
    text(size: 15pt, fill: soft)[ICLR 2024]
    v(1.1cm)
    text(size: 20pt, fill: luma(80))[Group: Mixture of Experts]
  }))
})

#include "sections/01_problem.typ"
#include "sections/02_contribution.typ"
#include "sections/03_method.typ"
#include "sections/04_experiments.typ"
#include "sections/05_critique.typ"
#include "sections/06_appendix.typ"
