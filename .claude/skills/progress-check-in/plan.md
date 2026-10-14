# Project plan

Drawn from the README roadmap, `docs/project_intro.md`, `docs/evaluation.md` and `docs/rag_pipeline_design_notes.md` (October 2026). Edit this file when the plan changes.

The goal is to learn and to explore the challenges of RAG and agents in a health setting, not to ship a production system. The priorities throughout are evaluation, refusing when the documents don't answer the question, and citations.

Each phase lists its goals and what "done" means. Status is not recorded here; the check-in works it out from the repo each time.

## Phase 1: Baseline RAG pipeline

- Parse the 10 PILs with layout-aware parsing, keeping metadata (medicine, source, heading, page).
- Chunk by section, splitting only oversized sections.
- Embed with BGE-small and store in Chroma.
- Answer a question with retrieval and an LLM, with citations and a refusal instruction.

**Done when:** `run_indexing` and `run_rag` work end to end and the design choices are written up in `docs/rag_pipeline_design_notes.md`.

## Phase 2: Baseline evaluation

- Build a small hand-labelled ground truth set covering the categories in `docs/evaluation.md`, including unanswerable questions.
- Retrieval metrics: Recall@k and MRR at the fact level.
- Refusal metrics: refusal rate, missed refusal rate, false refusal rate.
- RAGAS faithfulness and answer relevancy with a local judge from outside the Llama family.

**Done when:** a baseline run is saved in `results/`, visualised, and written up in `docs/evaluation.md` and the README.

## Phase 3: Engineering foundations

- Tests for the evaluation code (retrieval metrics, refusal metrics, ground truth loading), with pytest declared as a dev dependency.
- Prompt tracking: each prompt version is stored and linked to the runs that used it.
- Experiment tracking: each run records its config (chunking, embedding, k, LLM, prompt version, judge) next to its results, so runs can be compared.
- A decision log (`docs/decisions.md`) for significant choices.

**Done when:** tests pass and are committed, and any run in `results/` can be traced back to the exact config and prompt that produced it.

## Phase 4: Trustworthy evaluation

- Grow the ground truth set beyond 13 questions, with coverage across medicines and categories, and split it into a training set (used to iterate) and a held-out test set (not used to make changes).
- Measure how well the RAGAS judge agrees with my own labels on faithfulness and answer relevancy.
- Compare at least two judge LLMs on that agreement.
- Consider using the Inspect framework

**Done when:** I have a judge-agreement figure I can quote, a held-out test set exists, and I know how far to trust the RAGAS scores.

## Phase 5: Iterative improvements

Each change is measured against the previous run on the training set and logged as an ADR. Candidates found by the baseline:

- **False refusals:** 4 answerable questions were refused. Revise the prompt (a second version is drafted in `prompts.md`) and check that the missed refusal rate stays at 0.
- **Retrieval recall:** recall barely rises from k=3 to k=5, which suggests whole-section chunks hide small facts. Try smaller chunks or splitting list items.
- **Wrong-medicine retrieval:** classify which medicine the query is about and filter retrieval by `medicine_name`, with brand-name mapping and fuzzy matching for misspellings.
- **Other ideas from the design notes:** re-ranking, query rewriting.

Consider trying DSPy for prompt tuning.

**Done when:** at least one change has been shown to help (or not) with evidence, and the final system is scored once on the held-out test set.

## Phase 6: Deployment

Docker, a FastAPI service, deployed to Cloud Run on GCP, with GitHub Actions running tests and your evaluation suite on every change. Add tracing (Langfuse or OpenTelemetry) to track latency, token usage and cost per query.

## Future work (after Phase 6)

- A second retriever over clinical guidelines (e.g. SIGN, WHO).
- An agent that routes between the PIL and guideline retrievers, with routing accuracy measured.
- More agentic behaviour: tools, clarifying questions, sub-query planning.
