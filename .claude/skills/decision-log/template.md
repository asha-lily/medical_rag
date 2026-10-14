# Decision log entry template

Copy this structure for each entry. Section notes in brackets explain what goes there; do not include the notes in the entry.

---

### ADR NNN: [Short noun phrase naming the decision, e.g. "Add a cross-encoder reranker after hybrid retrieval"]

**Date:** YYYY-MM-DD  **Status:** [Proposed | Accepted | Accepted (supersedes ADR NNN) | Superseded by ADR NNN]

**Context**
[2-4 sentences. The problem, why it matters (link it to a failure mode, user impact or constraint), and the constraints that apply: latency, cost, data, time. Describe facts neutrally.]

**Options considered**
1. [Option, with a few words on its main appeal or drawback]
2. [Option]
3. [Option, if there was one]

**Evidence**
[How the options were compared: which dataset or eval, which metrics. Use a table when there are numbers.]

| Option | [Metric 1] | [Metric 2] | [Latency / cost] |
|---|---|---|---|
| [Baseline] | | | |
| [Option 1] | | | |

**Surprises** *(optional)*
[Anything that turned out differently from what was expected, and the likely reason.]

**Decision**
[One or two sentences, active voice: "I will..."]

**Consequences**
- Positive: [...]
- Negative: [...]
- Revisit if: [the condition that would make this decision worth reopening]
