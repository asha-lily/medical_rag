---
name: decision-log
description: Write a new entry in docs/decisions.md recording an ML or engineering decision, with the options considered, the evidence and the consequences. Use when the user asks to log, record or write up a decision.
argument-hint: [short description of the decision]
disable-model-invocation: true
---

# Write a decision log entry

Add a new entry to `docs/decisions.md` for this decision: $ARGUMENTS

## Steps

1. Read `docs/decisions.md`. If it doesn't exist, create it with the heading `# Decision log` and a one-line description of the project.
2. Find the highest existing ADR number and use the next one. Number with three digits (ADR 001, ADR 002...).
3. Gather the facts for the entry from what the user has said in this conversation, and from code, configs, experiment logs and evaluation results in the repo when the user points to them.
4. Before writing, check that you have: the context, at least two options considered, the evidence, and the decision. If any of these is missing, ask the user for it in one short list of questions. Do not guess.
5. Write the entry using the template in [template.md](template.md), appended at the end of the file. Newest entries go last.
6. If this decision replaces an earlier one, set this entry's status to "Accepted (supersedes ADR NNN)" and change the earlier entry's status to "Superseded by ADR NNN". Change nothing else in the earlier entry.
7. Show the user the new entry and ask whether anything needs correcting.

## Rules

- Never invent numbers, metrics, results or reasons. Every figure must come from the user or from a file in the repo. If a figure is missing, write `TODO: <what is needed>` in its place and point it out to the user.
- Write in plain, direct sentences. Use "I" rather than "we" (this is a solo project).
- Keep each entry under about 300 words. The table can hold most of the evidence.
- Include negative consequences, not only positive ones.
- Only include the "Surprises" line if something genuinely turned out differently from what was expected; otherwise leave it out.
- Use the date of the decision if the user gives one; otherwise use today's date.
