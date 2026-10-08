import logging

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from rag_pipeline.config import GenerationConfig
from rag_pipeline.retrieval import DocumentRetriever

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about over-the-counter medicines using extracts from UK patient information leaflets (PILs).

Answer the question using only the context provided below. If the context does not contain enough information to answer the question confidently, say "I don't have enough information to answer that question based on the available documents."

Do not make up information or draw on knowledge outside the provided context."""

_HUMAN_TEMPLATE = """Context:
{context}

Question: {question}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_TEMPLATE),
    ]
)


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a context string with source citations."""
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] Source: {source}, page {page}\n{doc.page_content}")
    return "\n\n".join(parts)


def create_answer_chain(config: GenerationConfig | None = None):
    """Return an LCEL chain: already-retrieved docs → prompt → LLM → string.

    Invoke it with {"question": str, "docs": list[Document]}. Use this when
    retrieval happens separately, e.g. in evaluation.
    """
    if config is None:
        config = GenerationConfig()

    llm = ChatOllama(
        model=config.model_name,
        num_predict=config.max_tokens,
        temperature=config.temperature,
    )

    return (
        RunnablePassthrough.assign(context=lambda x: _format_context(x["docs"]))
        | _PROMPT
        | llm
        | StrOutputParser()
    )


def create_rag_chain(
    retriever: DocumentRetriever, config: GenerationConfig | None = None
):
    """Return an LCEL chain: retriever → prompt → LLM → string.

    Invoke it with a question string; it returns {docs, question, answer}.
    """
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(
        answer=create_answer_chain(config)
    )


def generate(
    question: str,
    retriever: DocumentRetriever,
    config: GenerationConfig | None = None,
) -> dict:
    """Run a single RAG query.

    Returns a dict with:
        question (str)
        answer   (str)
        docs     (list[Document]) — the chunks the answer was generated from
    """
    chain = create_rag_chain(retriever, config)
    result = chain.invoke(question)

    log.info("Q: %s", question)
    log.info("A: %s", result["answer"])
    log.info("Sources:")
    for i, doc in enumerate(result["docs"], start=1):
        log.info(
            "  [%d] %s, page %s (%s)",
            i,
            doc.metadata.get("source", "unknown"),
            doc.metadata.get("page", "?"),
            doc.metadata.get("heading", "no heading"),
        )
    return result
