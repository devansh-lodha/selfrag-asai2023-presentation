# Agent Guidelines

## General Principles
- Be concise, precise, direct, information-dense, and technical.
- Keep language simple, clear, minimal, and easy to understand, ensuring ideas build up logically with zero redundancy.
- Do not use emojis, em-dashes, fluff, filler, or marketing language.
- Avoid fancy bullet formatting such as bold prefixes. Use simple, plain bullet points.
- Terminology: Use inclusive, modern language at all times.
- Focus on thoroughly understanding and applying underlying core concepts rather than relying on surface jargon.
- Stick strictly to concepts and materials explicitly provided, without introducing outside theories or uncovered material.
- Aim for the shortest possible output that completely satisfies all requirements and achieves the highest evaluation standard.
- Do not cite sources, drop academic names, or include reference lists unless explicitly requested, ensuring writing sounds like an authentic, independent human analysis.
- Keep this file lean. Only add new guidelines or project instructions with explicit user permission.
- Keep .gitignore minimal. Only add entries when required and do not bloat it.
- Use conventional commits for all git commit messages.

## Project Scope
- Typst presentation for "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2023 / ICLR 2024).
- Reference materials (paper, code, guidelines) are located in reference/.
- Skills to use Typst are located in .agents/skills/.

## Temporary Testing Workflow
- Use `.tmp/` inside the project root for local test builds, snippet evaluation, and rendering PNG page previews.
- Never write temporary scratch files outside the project directory.
- Clean up all temporary artifacts in `.tmp/` immediately after inspection and testing.
