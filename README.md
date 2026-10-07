This project is a work in progress.

This project builds a RAG pipeline for UK medicine leaflets

### Disclaimer

*It's crucial to note that the aim of this project is not to build an accurate, safe system that can be relied on, but rather to explore the challenges associated with applying RAG and agentic behaviour to a relatively simple and constrained health-related problem space.*

# About this project

I'm interested in how AI is being used in healthcare, and how it might be used in the future.

One such application is chatbots designed to answer health and medicine-related questions. These already exist.

This use-case initially struck me as very risky, and in need of very thorough evaluation and strict guardrails to minimise potential harm to users. Returning the wrong dose, or hallucinating a seemingly minor detail could be deadly. As chatbots become agentic, with access to tools and the ability to autonomously carry out multiple actions, the risks multiply.

My curiosity about the potential benefits of using AI in healthcare, and the challenges of doing this safely, motivates this project. 

I'd like to build an agentic RAG system in which the retrieval step has access to patient information leaflets (PILs). These are the leaflets inside every pack of medicine sold in the UK.

A system like this could be useful to the average person who may have over-the-counter medicines at home or in their bag but without the original information leaflet.

# Project roadmap

I'm setting out with the following plan:

- [ ] Build a simple RAG pipeline that can answer questions by retrieving relevant information from patient information leaflets.
- [ ] Evaluate this pipeline on basic metrics, including using RAGAS with an LLM judge to evaluate faithfulness and answer relevancy.
- [ ] Build a second retriever with access to a vector store of medical guideline documents. 
    - Where PILs address the question of 'how to take a medicine safely', guidelines address 'what's the right treatment or care'
    - Providers of guidelines include [sign](https://www.sign.ac.uk/guidelines/pharmacological-management-of-migraine/) & the [WHO](https://www.who.int/publications/who-guidelines)
- [ ] Give an agent access to both retrievers (PIL & guidelines) and let it choose which to use for a given question (i.e routing). Measure routing accuracy.
    - The user could then ask a question such as "My doctor prescribed amlodipine for high blood pressure. Why this medicine, and how should I take it?" The "why" is in the guideline; the "how" is in the PIL.
- [ ] Add more complex agentic behaviour, e.g provide access to tools, enable the agent to ask clarifying questions, plan sub-queries etc.

A key thing to measure will be whether the agent admits when it doesn't have the relevant information to answer the question, since hallucinating an answer puts the user at great risk in this domain. The system should also return citations with every answer.

# About the dataset

See `data/PILs`. I'm starting off with a set of 10 PILs for 10 common over-the-counter medicines.

An important note at this stage is that there's a lot of variety in the format of PILs from different providers. To build a robust RAG system I would want to sample from a diverse range of formats and make sure the parsing step works for all of them. For the MVP stage of this project I'm intentionally using a small dataset so that I can inspect them manually, and while I've tried to select a range of PIL formats, the small dataset size means that this diversity will be limited.

For more details on the dataset and the plan for this project, see `notebooks/project_intro.md`.

# Setup

1. Install dependencies (creates the virtual environment):

       uv sync

2. Install the pre-commit hooks (one-time, per clone):

       pre-commit install

Ruff now runs automatically on staged files at every commit.

3. Install [Ollama](https://ollama.com/download) (e.g. `brew install ollama` on macOS) and pull the models used for generation and evaluation:

       ollama pull llama3.2

       ollama pull gpt-oss:20b

Make sure the Ollama server is running (open the Ollama app, or run `ollama serve`) before running the RAG pipeline or evaluation.



## Running Indexing and question-answering

`uv run python -m rag_pipeline.run_indexing`

Once you've built the vector store of document chunks, you can generate an answer to your query:

`uv run python -m rag_pipeline.run_rag "<question>"`