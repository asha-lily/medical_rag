from typing import Annotated
from pydantic import BaseModel, StringConstraints


class QueryRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Source(BaseModel):
    source: str
    page: int | None
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
