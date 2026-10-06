from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.config import ChunkingConfig

config = ChunkingConfig()


def chunk_documents(
    documents: list[Document],
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = config.chunk_overlap,
) -> list[Document]:
    """Turn section-level Documents into chunks.

    Each section becomes exactly one chunk if it fits within `chunk_size`;
    oversized sections are split further with `RecursiveCharacterTextSplitter`.
    Each chunk inherits its source section's metadata (`source`, `heading`, ...)
    plus a `chunk_index` marking its position within that section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for document in documents:
        sub_documents = (
            [document]
            if len(document.page_content) <= chunk_size
            else splitter.split_documents([document])
        )
        for chunk_index, chunk in enumerate(sub_documents):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata['source']}#{chunk.metadata['section_index']}.{chunk_index}"
            )
            prefix_lines = []
            if medicine := chunk.metadata.get("medicine_name"):
                prefix_lines.append(f"Medicine: {medicine}")
            if heading := chunk.metadata.get("heading"):
                prefix_lines.append(f"Section: {heading}")
            if prefix_lines:
                chunk.page_content = (
                    "\n".join(prefix_lines) + "\n\n" + chunk.page_content
                )
            chunks.append(chunk)

    return chunks
