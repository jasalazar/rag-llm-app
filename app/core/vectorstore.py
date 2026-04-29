"""
Builds and returns a singleton MultiVectorRetriever wiring together:
  - Chroma (summary embeddings, for similarity search)
  - LocalFileStore (full documents, for context retrieval)

The retriever links the two stores via a shared "doc_id" metadata key.
At query time, Chroma finds the most relevant summaries and the retriever
fetches the corresponding full documents from disk.

Note: the voyageai SDK is not used because all of its releases are gated to
Python <3.13 or <3.14. Instead, VoyageEmbeddings calls the Voyage AI REST API
directly via httpx, which has no Python version restrictions.
"""

from functools import lru_cache

import httpx
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import settings


class VoyageEmbeddings(Embeddings):
    """
    Thin LangChain Embeddings wrapper that calls the Voyage AI REST API.
    Implements the two methods LangChain requires: embed_documents and embed_query.
    """

    _API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            self._API_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model},
            timeout=60.0,
        )
        response.raise_for_status()
        items = response.json()["data"]
        # Voyage AI returns results in order, but sort by index to be safe.
        return [item["embedding"] for item in sorted(items, key=lambda x: x["index"])]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_retriever() -> MultiVectorRetriever:
    embeddings = VoyageEmbeddings(
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
    )

    vectorstore = Chroma(
        collection_name="doc_summaries",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

    # LocalFileStore persists serialised Document objects to disk as bytes.
    # MultiVectorRetriever wraps it transparently via its byte_store parameter.
    docstore = LocalFileStore(settings.full_docs_store_dir)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=docstore,
        id_key="doc_id",
    )

    return retriever
