from fastapi import FastAPI
from rag_pipeline.api.schemas import QueryRequest, QueryResponse, Source

app = FastAPI()


@app.get("/health")
def get_health():
    return {"status": "ok"}


@app.post("/query")
def post_query(query_request: QueryRequest) -> QueryResponse:
    return QueryResponse(
        answer="This is a sample answer.",
        sources=[Source(source="sample source", page=1, chunk_id="sample_chunk_id")],
    )
