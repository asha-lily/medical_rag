# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG pipeline that answers questions about UK patient information leaflets (PILs), a set of 10 over-the-counter medicine leaflet PDFs in `data/PILs/`. It is an exploratory learning project, not a production system. The focus is on evaluation and on getting the model to admit when it doesn't know, rather than hallucinating. The roadmap (README.md) goes on to add a second retriever over clinical guidelines, then agentic routing between the two.

## Commands

Everything runs through `uv`; run the modules from the repo root with `-m`.

```bash
uv sync                      # install deps (Python 3.12)
pre-commit install           # one-time: ruff lint+format and nbstripout on commit

# Requires a running Ollama server (`ollama serve`) with `llama3.2` (generation)
# and `qwen3:8b` (RAGAS judge) pulled.
uv run python -m rag_pipeline.run_indexing                 # parse → chunk → embed → Chroma
uv run python -m rag_pipeline.run_rag "<question>"
uv run python -m rag_pipeline.run_evaluation --label <name> --collect-only  # writes results/run_<name>.json with "refusal": null
# hand-label every record's "refusal" as true/false, then:
uv run python -m rag_pipeline.run_evaluation --from-run <name>              # retrieval, refusal and RAGAS metrics
uv run python -m rag_pipeline.run_evaluation --from-run <name> --skip-ragas # retrieval + refusal metrics only

uv run ruff check . && uv run ruff format .
uv run --with pytest pytest tests/                                   # pytest is not yet a declared dependency
uv run --with pytest pytest tests/test_retrieval_metrics.py::test_name
```

## Architecture

`rag_pipeline/` holds library modules plus thin `run_*.py` entry points. All tunable settings (chunk size, embedding model, k, LLM names, RAGAS judge settings) are dataclasses in `rag_pipeline/config.py`.

Indexing flow (`run_indexing.py`):
1. `parse_documents.py`: Docling converts each PDF and splits it into **one LangChain `Document` per section**, using layout-detected headings. Metadata: `medicine_name` (taken from the filename, e.g. `ibuprofen_PIL.pdf` → "Ibuprofen"), `source` (the filename), `heading`, `page`, `section_index`. Docling is forced onto the CPU because MPS doesn't support float64.
2. `chunking.py`: a section that fits in `chunk_size` stays as one chunk; only oversized sections are split with `RecursiveCharacterTextSplitter`. Each chunk gets a `chunk_id` of `{source}#{section_index}.{chunk_index}`, and `Medicine: …` / `Section: …` lines are **prepended to `page_content`**, so they get embedded.
3. `vector_store.py`: BGE-small embeddings (normalized) go into a persistent Chroma store at `PIL_docs_vector_store/` in the repo root, using `chunk_id` as the Chroma id. `run_indexing` **does not rebuild an existing store**. After changing parsing, chunking or embedding, delete `PIL_docs_vector_store/` and re-index. (`chroma_db/` is an older, gitignored store.)

Query flow:
- `retrieval.py`: BGE needs `EmbeddingModelConfig.query_instruction` prepended to **queries only, not documents**. `retrieve()` handles this, and `DocumentRetriever` wraps it as a LangChain `BaseRetriever` for use in LCEL chains.
- `generation.py`: an LCEL chain returning `{docs, question, answer}`. The context is formatted as numbered `[i] Source: <file>, page <n>` blocks so the model can cite them. The system prompt tells the model to refuse when the context is insufficient.

Evaluation:
- **Ground truth set**: `data/ground_truth_dataset.yaml` (gitignored), loaded and validated by `ground_truth_data_set.py`. Each question has `answerable`, and answerable questions have `evidence` items of `{source, quote}`. Entries whose quotes start with `TODO` are skipped.
- **Retrieval metrics** (`retrieval_metrics.py`) work at the **fact level, not the chunk level**: an evidence item counts as retrieved if a chunk from the same `source` contains its quote after `normalise()`. Recall@k and MRR are all derived from each item's first-match rank. This avoids double-counting quotes duplicated by chunk overlap. `collect_run` checks this against the index and records unfindable ids in the run, which are then excluded from the metrics.
- **Generation metrics** (`ragas_evaluation.py`): RAGAS Faithfulness and AnswerRelevancy, using the local Ollama judge and the same HF embeddings. `max_workers` is kept low for laptop hardware. Only questions that are answerable **and** hand-labelled `refusal: false` are sent to RAGAS.
- **Refusal metrics** (`run_evaluation.py`): from the hand-labelled `refusal` field. Refusal rate = refused / unanswerable, missed refusal rate = 1 − that, false refusal rate = refused / answerable. Labels are validated (strict bools) before refusal metrics and RAGAS run.

`notebooks/` holds exploratory work for each pipeline stage, and `docs/` holds design notes and evaluation write-ups. nbstripout clears notebook outputs on commit.

`openfda_drug_labels/` is a separate script set for fetching US openFDA drug labels into `raw_labels/`, a possible future data source. It isn't wired into the pipeline yet. It uses bare imports (`from config import …`), so run it from inside that directory.