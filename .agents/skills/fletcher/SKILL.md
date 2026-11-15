---
name: fletcher
description: Draw flowcharts, commutative diagrams, finite-state machines, neural network architectures, and technical graphs in Typst using Fletcher. Use when creating, styling, or troubleshooting Fletcher diagrams, nodes, edges, anchors, coordinate systems, and custom marks in Typst.
---

# Fletcher Skill

Guide for creating and styling vector diagrams, flowcharts, architectures, and graphs in Typst using the Fletcher package.

## Quick start

Import Fletcher and optional shapes:

```typst
#import "@preview/fletcher:0.5.9" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond, pill, hexagon, house, trapezium, parallelogram

#diagram(
  node-stroke: 1pt,
  node-fill: blue.lighten(80%),
  spacing: 2cm,
  node((0, 0), [Input], shape: pill),
  edge("-|>"),
  node((1, 0), [Process], shape: diamond),
  edge("-|>"),
  node((2, 0), [Output], shape: pill),
)
```

Working examples are located in `assets/examples/`:
- `assets/examples/03-ml-architecture.typ`: Multi-layer neural network architecture
- `assets/examples/04-io-flowchart.typ`: Input/output flowchart with decision branches
- `assets/examples/06-node-groups.typ`: Grouped nodes with dynamic boundaries
- `assets/examples/07-uml-diagram.typ`: UML class and component diagram
- `assets/examples/08-tree.typ`: Hierarchical tree structure
- `assets/examples/01-commutative.typ`: Category-theoretic commutative diagram
- `assets/examples/3-state-machine.typ`: Finite-state machine with self-loops

## Coordinate systems

Fletcher supports multiple coordinate formats within the same diagram:

- Elastic coordinates: `(col, row)` grid units, such as `(0, 0)` or `(1, 2)`. Grid spacing is governed by `spacing` or `cell-size`.
- Physical coordinates: Absolute lengths, such as `(20mm, 15mm)` or `(10pt, 50pt)`.
- Relative coordinates: `(rel: (1, 0), to: (0, 0))` or `(rel: (10mm, 0mm), to: <node-name>)`.
- Polar coordinates: `(dir: 45deg, dist: 3cm)` or `(rel: (dir: 90deg, dist: 15mm), to: <node-name>)`.
- Named node anchors: `<node-name>` or `<node-name.north>`, `<node-name.south>`, `<node-name.east>`, `<node-name.west>`, `<node-name.top-left>`, etc.
- Shorthand directions: `"r"` (right), `"l"` (left), `"u"` (up), `"d"` (down), `"ur"` (up-right), `"dl"` (down-left). Can be sequenced: `"r,u,r"`.

## Nodes API

Syntax:
```typst
node(pos, label, name: <id>, shape: rect, fill: none, stroke: 1pt, ..options)
```

Key node parameters:
- `pos`: Position tuple or label.
- `label`: Content or string displayed inside the node.
- `name`: Unique label for referencing in edges or enclosures (e.g., `name: <encoder>`).
- `shape`: `rect`, `circle`, `diamond`, `pill`, `hexagon`, `house`, `trapezium`, `parallelogram`, `octagon`, `isosceles-triangle`, `cylinder`, `brace`, `bracket`, `paren`.
- `fill`: Background color or gradient.
- `stroke`: Border stroke thickness and color, or `none`.
- `width`, `height`, `radius`: Explicit sizing overrides.
- `corner-radius`: Radius for rounded corners on box-like shapes.
- `extrude`: Numerical tuple (e.g. `(0, 3)`) to draw duplicate extruded strokes (double-outline effect).
- `enclose`: List of node labels to dynamically encompass (e.g., `enclose: (<n1>, <n2>)`).
- `layer`: Z-index integer controlling drawing order (higher values draw on top).
- `align`: Content alignment within the node (e.g. `center`, `top + left`).
- `inset`, `outset`: Inner padding and outer boundary offsets.

Enclosing node groups:
```typst
#diagram(
  node((0, 0), [A], name: <a>),
  node((1, 0), [B], name: <b>),
  node(enclose: (<a>, <b>), stroke: 0.5pt + gray, fill: rgb(0, 0, 0, 10), inset: 5pt),
)
```

## Edges API

Syntax:
```typst
edge(from, to, label, marks, ..options)
```

Key edge parameters:
- `from`, `to`: Start and end endpoints. If omitted, `from` defaults to previous node and `to` defaults to subsequent node or direction shorthand.
- `marks`: Arrowhead string shorthand or mark array.
  - Simple: `"->"`, `"<-"`, `"<->"`, `"-"`
  - Solid: `"-|>"`, `"<|-"`, `"<|-|>"`
  - Double: `"--|>"`
  - Line styles: `"dashed"`, `"dotted"`, `"wave"`, `"zigzag"`, `"coil"`
  - Crow's foot: `"n"`, `"1"`, `"n!"`, `"1?"`
  - Brackets: `"["`, `"]"`, `"("`, `")"`
- `bend`: Angle for curved arcs (positive curves clockwise, negative counterclockwise), e.g. `bend: 30deg`.
- `corner`: Right-angled edge routing, values `left` or `right`.
- `shift`: Offset perpendicular to edge trajectory, as length (`shift: 3pt`) or coordinate delta.
- `label`: Annotation content.
- `label-pos`: Position along edge from 0.0 (start) to 1.0 (end).
- `label-side`: Side of edge for label (`left`, `right`, `center`).
- `label-angle`: `auto` (parallel to edge), `0deg` (horizontal), or fixed angle.
- `snap-to`: Node identifiers to snap edge ends to, or `none` to disable boundary snapping.
- `layer`: Drawing layer z-index.

Multi-segment edge example:
```typst
edge((0, 0), "r,u,r", "-|>", [Step], label-pos: 0.5)
```

## Math mode diagrams

Fletcher diagrams can be defined with math grid syntax using `&` column separators and `\` line breaks:

```typst
#diagram(cell-size: 15mm, $
  G edge(f, ->) edge("d", pi, ->>) & im(f) \
  G slash ker(f) edge("ur", tilde(f), "hook-->")
$)
```

## Touying presentation integration

To animate Fletcher diagrams across Touying presentation slides, wrap `diagram` with `touying-reducer`:

```typst
#import "@preview/touying:0.6.1": *
#import "@preview/fletcher:0.5.9" as fletcher: diagram, node, edge

#let diagram = touying-reducer.with(
  reduce: fletcher.diagram,
  cover: fletcher.hide,
)

#slide(repeat: 3, self => {
  let (uncover, only, alternatives) = utils.methods(self)
  diagram(
    node((0, 0), [Input], name: <a>),
    pause,
    edge("->"),
    node((1, 0), [Hidden], name: <b>),
    pause,
    edge("->"),
    node((2, 0), [Output], name: <c>),
    only("3", edge(<a>, "~", <c>, bend: 40deg, stroke: red)),
  )
})
```

## CeTZ canvas integration

For custom geometry or complex paths, use the `render` callback hook:

```typst
#diagram(
  node((0, 0), $A$, name: <a>),
  node((2, 1), $B$, name: <b>),
  render: (grid, nodes, edges, options) => {
    fletcher.cetz.canvas({
      fletcher.draw-diagram(grid, nodes, edges, debug: options.debug)
      let n1 = fletcher.find-node-at(nodes, <a>)
      let n2 = fletcher.find-node-at(nodes, <b>)
      let p1 = fletcher.get-node-anchor(n1, 0deg)
      let p2 = fletcher.get-node-anchor(n2, 180deg)
      fletcher.cetz.draw.bezier(p1, p2, (rel: (0deg, 20pt), to: p1), (rel: (180deg, 20pt), to: p2))
    })
  }
)
```

## Documentation references

Consult the reference files in `references/`:
- `references/manual.typ`: Complete package manual and API reference
- `references/README.md`: Quick overview, release notes, and version compatibility matrix
- `assets/examples/`: Runnable diagram examples for machine learning architectures, flowcharts, UML diagrams, trees, and state machines
