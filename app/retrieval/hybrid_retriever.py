"""
hybrid_retriever.py
────────────────────────────────────────────────────────────────────────────────
Production-grade Ensemble Retriever for RAG pipelines.

Combines:
  • Dense vector retrieval  — FAISS (local) or Chroma (persistent)
  • Sparse BM25 retrieval   — rank-bm25 via LangChain's BM25Retriever
  • Reciprocal Rank Fusion  — merges both ranked lists into a single ranking

Install dependencies:
    pip install langchain langchain-community langchain-openai \
                faiss-cpu rank-bm25 chromadb tiktoken
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class VectorStoreBackend(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"


@dataclass
class EnsembleRetrieverConfig:
    """
    Central configuration for the hybrid retriever.

    Attributes:
        vector_k:          Top-K docs fetched from the vector store.
        bm25_k:            Top-K docs fetched from BM25.
        vector_weight:     Weight assigned to vector scores during RRF (0–1).
        bm25_weight:       Weight assigned to BM25 scores during RRF (0–1).
                           Must sum to 1.0 with vector_weight.
        rrf_k:             RRF smoothing constant (default 60 per the paper).
        vector_backend:    Which vector store to use — FAISS or Chroma.
        chroma_collection: Collection name (only used when backend=CHROMA).
        chroma_persist_dir: Persistence directory for Chroma (optional).
    """
    vector_k: int = 10
    bm25_k: int = 10
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60
    vector_backend: VectorStoreBackend = VectorStoreBackend.FAISS
    chroma_collection: str = "ask_my_docs"
    chroma_persist_dir: str | None = None

    def __post_init__(self) -> None:
        total = round(self.vector_weight + self.bm25_weight, 6)
        if total != 1.0:
            raise ValueError(
                f"vector_weight + bm25_weight must equal 1.0, got {total}"
            )
        if self.vector_k < 1 or self.bm25_k < 1:
            raise ValueError("vector_k and bm25_k must be >= 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")


# ──────────────────────────────────────────────────────────────────────────────
# Scored result container (used for custom RRF if needed)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class ScoredChunk:
    """A retrieved document chunk with its RRF-fused score."""
    score: float
    document: Document = field(compare=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class HybridEnsembleRetriever:
    """
    Hybrid retriever that fuses BM25 (sparse) and vector search (dense)
    using Reciprocal Rank Fusion (RRF).

    How RRF works
    ─────────────
    Given two ranked lists L₁ (BM25) and L₂ (vector), the fused score for
    document d is:

        RRF(d) = Σᵢ  weight_i / (k + rank_i(d))

    where rank_i(d) is d's 1-based position in list i, and k is a smoothing
    constant (typically 60) that prevents very high scores for top-ranked
    documents dominating entirely.

    This approach is:
    - Parameter-light (only k and per-list weights).
    - Robust: works even when the two retrievers use incomparable score scales.
    - Empirically strong: often outperforms more complex fusion methods.

    Usage
    ─────
        docs = [...]  # list of langchain Document objects
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        config = EnsembleRetrieverConfig(vector_weight=0.6, bm25_weight=0.4)

        retriever = HybridEnsembleRetriever(
            documents=docs,
            embeddings=embeddings,
            config=config,
        )
        results = retriever.retrieve("What is retrieval-augmented generation?")
    """

    def __init__(
        self,
        documents: list[Document],
        embeddings: Embeddings | None = None,
        config: EnsembleRetrieverConfig | None = None,
    ) -> None:
        """
        Args:
            documents:  Corpus of documents to index. Must be non-empty.
            embeddings: LangChain-compatible embedding model. Defaults to
                        OllamaEmbeddings(model="nomic-embed-text") if not provided.
            config:     Retriever configuration. Uses sensible defaults if
                        not provided.

        Raises:
            ValueError: If documents list is empty.
            RuntimeError: If vector store or BM25 index cannot be built.
        """
        if not documents:
            raise ValueError("documents list must be non-empty.")

        self.config = config or EnsembleRetrieverConfig()
        self.embeddings = embeddings or OllamaEmbeddings(model="nomic-embed-text")
        self._documents = documents
        self._ensemble: EnsembleRetriever | None = None

        logger.info(
            "Initialising HybridEnsembleRetriever | backend=%s | docs=%d",
            self.config.vector_backend.value,
            len(documents),
        )
        self._build()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Build both sub-retrievers and compose the EnsembleRetriever."""
        vector_retriever = self._build_vector_retriever()
        bm25_retriever = self._build_bm25_retriever()

        self._ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[self.config.bm25_weight, self.config.vector_weight],
            # LangChain's EnsembleRetriever uses RRF internally with c=rrf_k
            c=self.config.rrf_k,
        )
        logger.info("EnsembleRetriever built successfully.")

    def _build_vector_retriever(self) -> BaseRetriever:
        """Construct and return the dense vector retriever."""
        try:
            if self.config.vector_backend == VectorStoreBackend.FAISS:
                logger.debug("Building FAISS index for %d docs…", len(self._documents))
                vector_store = FAISS.from_documents(
                    documents=self._documents,
                    embedding=self.embeddings,
                )
            else:
                logger.debug("Building Chroma index — collection=%s", self.config.chroma_collection)
                kwargs: dict[str, Any] = {
                    "documents": self._documents,
                    "embedding": self.embeddings,
                    "collection_name": self.config.chroma_collection,
                }
                if self.config.chroma_persist_dir:
                    kwargs["persist_directory"] = self.config.chroma_persist_dir
                vector_store = Chroma.from_documents(**kwargs)

            return vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.vector_k},
            )
        except Exception as exc:
            logger.exception("Failed to build vector retriever: %s", exc)
            raise RuntimeError(f"Vector store construction failed: {exc}") from exc

    def _build_bm25_retriever(self) -> BM25Retriever:
        """Construct and return the sparse BM25 retriever."""
        try:
            logger.debug("Building BM25 index for %d docs…", len(self._documents))
            retriever = BM25Retriever.from_documents(self._documents)
            retriever.k = self.config.bm25_k
            return retriever
        except Exception as exc:
            logger.exception("Failed to build BM25 retriever: %s", exc)
            raise RuntimeError(f"BM25 index construction failed: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[Document]:
        """
        Run hybrid retrieval for a query.

        Args:
            query: The user's natural-language query string.

        Returns:
            Deduplicated list of Documents ranked by RRF score (highest first).

        Raises:
            ValueError: If the query string is empty.
            RuntimeError: If retrieval fails.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if self._ensemble is None:
            raise RuntimeError("Retriever not initialised — call _build() first.")

        logger.info("Retrieving for query: %r", query[:80])
        try:
            results = self._ensemble.invoke(query)
            logger.info("Retrieved %d documents after RRF fusion.", len(results))
            return results
        except Exception as exc:
            logger.exception("Retrieval failed for query %r: %s", query[:80], exc)
            raise RuntimeError(f"Hybrid retrieval failed: {exc}") from exc

    def add_documents(self, new_documents: list[Document]) -> None:
        """
        Incrementally add new documents and rebuild both indexes.

        Note: For large corpora, prefer rebuilding offline and swapping
        the instance rather than live updates.

        Args:
            new_documents: Additional documents to include in the index.

        Raises:
            ValueError: If new_documents is empty.
        """
        if not new_documents:
            raise ValueError("new_documents must be non-empty.")

        logger.info("Adding %d new documents and rebuilding indexes…", len(new_documents))
        self._documents.extend(new_documents)
        self._build()

    # ──────────────────────────────────────────────────────────────────────────
    # Optional: manual RRF for inspection / testing
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_rrf_scores(
        ranked_lists: list[list[Document]],
        weights: list[float],
        k: int = 60,
    ) -> list[ScoredChunk]:
        """
        Standalone RRF implementation — useful for unit testing and debugging.

        Args:
            ranked_lists: One list of ranked Documents per retriever.
            weights:      Per-list weights. Must have same length as ranked_lists.
            k:            RRF smoothing constant.

        Returns:
            List of ScoredChunks sorted by descending RRF score, deduplicated
            by page_content.

        Example:
            bm25_results   = [doc_A, doc_B, doc_C]
            vector_results = [doc_C, doc_A, doc_D]
            fused = HybridEnsembleRetriever.compute_rrf_scores(
                ranked_lists=[bm25_results, vector_results],
                weights=[0.5, 0.5],
            )
        """
        if len(ranked_lists) != len(weights):
            raise ValueError("ranked_lists and weights must have equal length.")
        if not all(0 <= w <= 1 for w in weights):
            raise ValueError("All weights must be in [0, 1].")

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, doc in enumerate(ranked_list, start=1):
                key = doc.page_content  # deduplicate by content
                rrf_contribution = weight / (k + rank)
                scores[key] = scores.get(key, 0.0) + rrf_contribution
                doc_map[key] = doc

        return sorted(
            [ScoredChunk(score=score, document=doc_map[key]) for key, score in scores.items()],
            key=lambda sc: sc.score,
            reverse=True,
        )

    def __repr__(self) -> str:
        return (
            f"HybridEnsembleRetriever("
            f"backend={self.config.vector_backend.value!r}, "
            f"docs={len(self._documents)}, "
            f"weights=(bm25={self.config.bm25_weight}, vec={self.config.vector_weight}), "
            f"rrf_k={self.config.rrf_k})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_docs = [
        Document(page_content="Retrieval-Augmented Generation grounds LLMs in external knowledge.", metadata={"source": "intro.txt"}),
        Document(page_content="BM25 is a probabilistic ranking function based on term frequency.", metadata={"source": "bm25.txt"}),
        Document(page_content="Dense vector search uses cosine similarity in embedding space.", metadata={"source": "vector.txt"}),
        Document(page_content="Cross-encoder rerankers score query-document pairs jointly.", metadata={"source": "rerank.txt"}),
        Document(page_content="Reciprocal Rank Fusion combines multiple ranked lists robustly.", metadata={"source": "rrf.txt"}),
    ]

    config = EnsembleRetrieverConfig(
        vector_k=3,
        bm25_k=3,
        vector_weight=0.6,
        bm25_weight=0.4,
        vector_backend=VectorStoreBackend.FAISS,
    )

    # NOTE: requires OPENAI_API_KEY in environment
    retriever = HybridEnsembleRetriever(
        documents=sample_docs,
        config=config,
    )
    print(retriever)

    results = retriever.retrieve("How does BM25 rank documents?")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content[:80]}")
