# Study Guidelines

## Pedagogy
- Follow the logical sequence defined in study/plan.md rather than paper chronology, synthesizing concepts across the paper and codebase so material builds up from first principles without forward references or redundant recaps.
- Introduce every concept, token, and algorithm with clear, plain-language intuition explaining what problem it solves before presenting formal notation or implementation details.
- Derive mathematical formulations from first principles, showing all intermediate algebraic steps and explaining why each term and operation exists.
- Define every symbol, subscript, parameter, and tensor dimension immediately after each equation, and explain how changing each variable affects the output.
- State assumptions, boundary conditions, selection criteria, failure modes, and trade-offs for each method, making it clear when the method applies and when it fails.
- Ground all theoretical explanations in primary paper references (reference/paper/) and code implementations (reference/code/), verifying tensor shapes, loss masking, and inference logic.
- Stay strictly within the scope of the current part in study/plan.md with zero repetition across parts.

## Formatting
- Rely on pure Markdown and LaTeX math ($...$ for inline math, $$...$$ for display math).
- Do not create ASCII art, ASCII box diagrams, or fragile ASCII flowcharts.
- Do not wrap prose, derivations, or tables inside code blocks. Reserve code blocks strictly for actual executable code or raw data snippets.
- In LaTeX math mode, do not use double straight quotes or unicode quotes like \text{"word"}. Use proper LaTeX quotation syntax \text{``word''}.
- In Markdown table cells, never use raw unescaped pipe characters (| or \|) inside LaTeX math. Strictly use \vert, \Vert, \mid, or text notation to avoid breaking table column separation.

## Plan Maintenance
- Maintain study/plan.md as an evolving roadmap organized into numbered modules and narrowly scoped part files ordered by prerequisite dependencies rather than raw paper sections.
- For each part in study/plan.md, list the target file, source mappings (paper sections, codebase files), and the ordered sequence of concepts, derivations, and code walkthroughs to cover.
- Split or refine parts in study/plan.md whenever needed so each part can be written in full depth in a single turn without compression.

## Workflow
- Read study/plan.md and corresponding source files in reference/paper/ and reference/code/ before writing.
- Write directly to files in study/; never dump large file contents into the chat output.
- Write exactly one part file per turn into its designated path in study/, then mark it complete ([x]) in study/plan.md.
- Reply in chat with a concise status summary only, without pasting file contents, and wait for user review before starting the next part.
- Update relevant files in study/ directly whenever questions, revisions, or doubts arise.
- Use study/memory.md as an active scratchpad for continuous understanding, discoveries, pending reminders, mathematical nuances, and edge cases.
