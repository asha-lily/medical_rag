from fastapi import FastAPI
from schemas import QueryRequest, QueryResponse

app = FastAPI()


@app.get("/health")
def get_health():
    return {"status": "ok"}


@app.post("/query")
def post_query(query_request: QueryRequest) -> QueryResponse:
    result = {"answer": "This is a sample answer.", "sources": []}
    return QueryResponse(**result)
