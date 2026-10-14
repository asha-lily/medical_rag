---
name: progress-check-in
description: Review progress on this project against the plan in plan.md, give honest feedback on where things stand, and recommend what to do next. Use when the user asks for a progress check-in, a review of where they're at, or what to work on next.
argument-hint: [optional: what you've been working on, or a question to focus on]
disable-model-invocation: true
---

# Progress check-in

Review my progress against the plan in [plan.md](plan.md) and tell me where I'm at and what to do next.

Extra context from me, if any: $ARGUMENTS

## Steps

1. Read [plan.md](plan.md).
2. Gather evidence from the repo:
   - `git log --format='%ad %s' --date=short | head -30` for recent work, and `git status` for uncommitted work.
   - The README roadmap, `docs/evaluation.md`, `docs/decisions.md` and `docs/rag_pipeline_design_notes.md`.
   - The runs in `results/` and the code in `rag_pipeline/` and `tests/`, where needed to judge whether a goal is met.
3. For each phase, give a status of **Done**, **In progress** or **Not started**. Judge each goal against its "Done when" criteria and name the evidence (a file, a commit, a result) for every status.
4. Show me the check-in in this format:

   ```
   ## Progress check-in: <date>

   **Summary:** <two or three sentences on where the project stands>

   | Phase | Status | Evidence |
   |---|---|---|

   **What's going well:** <short bullets>

   **Concerns:** <short bullets: gaps and risks>

   **Recommended next steps:**
   1. <the single most useful next task, with why and a concrete first action>
   2. ...
   ```

## Rules

- Don't run anything except the git commands above: no tests, indexing, RAG pipeline or evaluation. Judge from files only. Where a goal depends on something you can't check this way (for example, whether tests pass), say so.
- Don't create or edit any files. The check-in goes in your reply only.
- Every status and claim must point to evidence in the repo. If you can't tell whether something is done, say so and ask me. Don't guess.
- Be honest and direct. Point out work that is uncommitted, untested or undocumented.
- Phases don't have to be done in order. Working on a later phase before an earlier one is finished is fine, so don't flag it as a concern.
- Recommend no more than three next steps, in priority order. Make each one small enough to start today.
- Keep the project's priorities in mind: evaluation, refusing when the documents don't answer the question, and citations.
- If the README roadmap and plan.md disagree, point it out.
- Use plain, direct sentences. Keep the check-in under about 400 words.
