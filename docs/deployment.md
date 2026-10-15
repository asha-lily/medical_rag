So far, I've built a basic RAG pipeline script that runs locally in my terminal.

Currently, `rag_pipeline/run_rag.py` takes a question as input, loads the vector store, embedding model, retriever, and LCEL chain, then invokes the LCEL chain with the question. The result contains `docs`, `question` and `answer`. 

Now we want to deploy it (just the question-answering part, not indexing). This involves the following:

- Enabling other programs / people to use it (currently it only runs via my terminal) -> wrap the pipeline in an API service
- Making sure it runs the same everywhere (e.g on a different laptop) -> containerise it
- Tracking how the system is behaving -> add observability (e.g using langfuse)


## 1. Wrap the pipeline in a FastAPI service

FastAPI is a python web framework that handles HTTP requests to / from a service.

A service is an application (in this case the RAG pipeline) that runs continuously and waits for requests. The interface the service exposes is an API.

In the past I've used Flask, but FastAPI is now more widely used, and has the benefits of adding validation and documentation automatically.

### Build a `/health` endpoint

Sending a `GET` request to the `/health` endpoint indicates whether the service is up and running. A response of `{"status": "ok"}` is returned when this is the case.

As well as installing `fastapi`, we need to install `uvicorn` as the server.


FastAPI is built on pydantic.

```
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)

class Source(BaseModel):
    source: str
    page: int
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
```

```
rag_pipeline/
  api/
    __init__.py
    schemas.py   ← the Pydantic models above (the contract)
    main.py      ← the FastAPI app: lifespan loading + /query, /health, /ready
  config.py
  retrieval.py
  generation.py
  ...
```






## 2. Containerise



## 3. Add Langfuse with docker compose

Langfuse is an LLM observability tool which records what happens inside each request and displays it in a web dashboard. This will enable us to monitor latency and token usage, as well as the prompt and retrieved chunks for each question.

Langfuse has the benefit of being open source (unlike LangSmith) and is easier to set up than something like OpenTelemetry + Prometheus / Grafana (although this is what a lot of production systems use).

As well as being a python library, Langfuse sends traces, so a web application needs to receive, store and display them. That web application needs to be run by someone, and needs databases behind it. We could use Langfuse cloud, or self-host.

Just like with the choice between a hosted and local LLM, self-hosting might be preferred for systems working with sensitive data.


## 4. Add CI with GitHub actions



## 5. Next steps

When you're ready to start step 1, I can either write it for you or set you smaller tasks (e.g. "get /health returning {"status": "ok"}") and review what you write. Given your learning goal, I'd suggest the second.

- Add streamlit UI?


## Questions

Should I leave my existing code as it is, e.g run_rag.py?

how to use hosted langfuse: I've signed up and got my API keys -- where do they go?

