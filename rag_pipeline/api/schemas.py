from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(strip_whitespace=True, min_length=1)


class Source(BaseModel):
    source: str
    page: int | None
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
