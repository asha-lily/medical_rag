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
