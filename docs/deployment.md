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

Sending a `GET` request to the `/health` endpoint indicates whether the service is up and running; `{"status": "ok"}` should be returned. If the service is not running, checking the health (e.g by running `curl http://localhost:8000/health`) will return a `connection refused` error. If the service is running but something is broken, the request will timeout.

As well as installing `fastapi`, we need to install `uvicorn` as the server.

### Build a `/query` endpoint

In `schemas.py` we define the `QueryRequest` pydantic model which defines what a request must look like. We also define `QueryResponse` which defines what a response from the service will look like. FastAPI is built on pydantic, and validates that both the request and response bodies adhere to their pydantic models. 

I say 'bodies' here because requests and responses also contain headers, which contain metadata about the message. Since our query request body is a dictionary (`{"question": "<question>"}`), we need to add a `Content-Type: application/json` header so that FastAPI reads the body as JSON. Without this header in the request, the response contains this message: `Input should be a valid dictionary or object to extract fields from`. Other headers get added to the request automatically, e.g `Host: localhost:8000`.

A question can be send to the `/query` endpoint with a curl request like the following:

```
curl -v -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I take ibuprofen with paracetamol?"}'
```

For the hardcoded `post_query()` function, the following is returned as expected:

```
{"answer":"This is a sample answer.","sources":[{"source":"sample source","page":1,"chunk_id":"sample_chunk_id"}]}
```

If the question is left empty, the response says `string should have at least 1 character`. This is because in the `schemas.py` query request model we've required the question to have `min_length=1`.







## 2. Containerise



## 3. Add Langfuse with docker compose

Langfuse is an LLM observability tool which records what happens inside each request and displays it in a web dashboard. This will enable us to monitor latency and token usage, as well as the prompt and retrieved chunks for each question.

Langfuse has the benefit of being open source (unlike LangSmith) and is easier to set up than something like OpenTelemetry + Prometheus / Grafana (although this is what a lot of production systems use).

As well as being a python library, Langfuse sends traces, so a web application needs to receive, store and display them. That web application needs to be run by someone, and needs databases behind it. We could use Langfuse cloud, or self-host.

Just like with the choice between a hosted and local LLM, self-hosting might be preferred for systems working with sensitive data.


## 4. Add CI with GitHub actions



## 5. Next steps

- Add streamlit UI?


## Questions

Should I leave my existing code as it is, e.g run_rag.py?

how to use hosted langfuse: I've signed up and got my API keys -- where do they go?

