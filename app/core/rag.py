"""
Two entry points:

  ingest_document(text, source)
    1. Ask Claude to summarise the full document.
    2. Embed the summary and store it in Chroma (keyed by a new UUID).
    3. Persist the full document in LocalFileStore under the same UUID.

  query_rag(question)
    1. Embed the question and search Chroma for the most relevant summaries.
    2. MultiVectorRetriever resolves each summary to its full document.
    3. Pass all full documents as context to Claude and return its answer.
"""

import hashlib
import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.core.vectorstore import get_retriever
from langsmith import traceable


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model_name,
        anthropic_api_key=settings.anthropic_api_key,
    )


_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at summarising documents for retrieval. "
        "Produce a concise but comprehensive summary that captures all key "
        "information, concepts, entities, and arguments in the document.",
    ),
    ("human", "Summarise the following document:\n\n{document}"),
])

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question using only "
        "the provided context documents. If the context does not contain "
        "enough information to answer, say so clearly.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])

@traceable
async def ingest_document(text: str, source: str = "uploaded") -> str:
    retriever = get_retriever()

    # Compute a content fingerprint and check Chroma for an existing match.
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    results = retriever.vectorstore.get(where={"content_hash": content_hash})
    if results["ids"]:
        existing_doc_id = results["metadatas"][0]["doc_id"]
        return existing_doc_id  # Already stored — skip summarisation and embedding.

    llm = _get_llm()

    doc_id = str(uuid.uuid4())
    full_doc = Document(page_content=text, metadata={"source": source, "doc_id": doc_id})

    summary_chain = _SUMMARIZE_PROMPT | llm | StrOutputParser()
    summary = await summary_chain.ainvoke({"document": text})

    summary_doc = Document(
        page_content=summary,
        metadata={"doc_id": doc_id, "source": source, "content_hash": content_hash},
    )

    # Persist: full document in the file store, summary embedding in Chroma.
    retriever.docstore.mset([(doc_id, full_doc)])
    retriever.vectorstore.add_documents([summary_doc])

    return doc_id


@traceable(run_type="tool")
async def query_rag(question: str) -> str:
    retriever = get_retriever()
    llm = _get_llm()

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return await chain.ainvoke(question)
