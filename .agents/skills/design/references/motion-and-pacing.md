# Motion, Transitions, and Slide Animation Principles

Guidelines for meaningful visual pacing and dynamic reveals in presentations.

## 1. Motion as Cognitive Explanation

Motion and progressive reveals must explain state changes, sequence logic, and relationships:
- Never add animation solely for decorative spectacle.
- Every dynamic reveal or transition must either:
  - Direct audience attention to a specific step in an algorithmic process.
  - Decompose a complex mathematical derivation into understandable phases.
  - Reveal architecture components progressively to explain data flow.

## 2. Progressive Reveals in Touying Slides

- Space-Reserving Reveals (uncover):
  - Use `uncover("2-")[...]` when the hidden content must reserve its physical space to prevent layout shifts or jumping text.
  - Essential for equations, tables, and side-by-side diagrams where layout stability is mandatory.
- Space-Collapsing Reveals (only):
  - Use `only("2")[...]` when content alternates or collapses without leaving empty gaps.
- Sequential Reveals (pause / meanwhile):
  - Use `#pause` for linear bullet reveals.
  - Use `#meanwhile` for simultaneous reveals in multiple columns or diagram nodes.

## 3. Timing and Easing Rules

- Duration ranges:
  - Immediate feedback and highlights: 100ms to 150ms.
  - Routine state transitions and reveals: 200ms to 300ms.
  - Complex architectural shifts: 300ms to 500ms.
- Asymmetry in transitions: Exits should be faster than entrances.
- Easing curves: Use natural deceleration curves (such as cubic-bezier ease-out); never use bounce or elastic curves by reflex.
- Slide Pacing: In live presentations, pace reveals so each subslide corresponds to 15-30 seconds of spoken explanation.
