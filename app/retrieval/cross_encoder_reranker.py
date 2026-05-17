"""
cross_encoder_reranker.py
────────────────────────────────────────────────────────────────────────────────
Production-grade Cross-Encoder Reranker for RAG pipelines.

Why reranking?
──────────────
First-stage retrievers (BM25, vector search) use BI-ENCODERS: the query and
each document are embedded INDEPENDENTLY, then compared by cosine similarity.
This is fast (O(1) per query at search time) but misses subtle interactions
between the query and document tokens.

A CROSS-ENCODER reads the query and document TOGETHER in a single forward pass,
letting every query token attend to every document token. This captures
fine-grained relevance signals (negation, co-reference, paraphrase) that
bi-encoders cannot see. The trade-off: cross-encoders are ~100× slower, so
they are applied only to the small top-K shortlist from the first stage.

Pipeline position:
  [Hybrid retriever → top-50 raw candidates]
        ↓
  [Cross-encoder reranker → top-5 reranked]
        ↓
  [LLM generation with citations]

Install dependencies:
    pip install sentence-transformers langchain-core torch
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants — well-tested public checkpoints
# ──────────────────────────────────────────────────────────────────────────────

RECOMMENDED_MODELS: dict[str, str] = {
    "fast":    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # 22M params, ~10ms/pair CPU
    "balanced":"cross-encoder/ms-marco-MiniLM-L-12-v2",  # 33M params, best speed/quality
    "strong":  "cross-encoder/ms-marco-electra-base",    # 110M params, highest accuracy
    "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}

DEFAULT_MODEL = RECOMMENDED_MODELS["balanced"]


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class RankedChunk:
    """
    A document chunk with its cross-encoder relevance score.

    Attributes:
        score:       Raw logit from the cross-encoder (higher = more relevant).
                     Not normalised — only use for ranking, not as a probability.
        document:    The original LangChain Document object.
        rank:        1-based position in the reranked output list.
        model_name:  The cross-encoder checkpoint that produced this score.
    """
    score: float
    document: Document = field(compare=False)
    rank: int = field(default=0, compare=False)
    model_name: str = field(default="", compare=False)

    @property
    def source(self) -> str:
        """Convenience accessor for metadata source."""
        return self.document.metadata.get("source", "unknown")

    def __repr__(self) -> str:
        preview = self.document.page_content[:60].replace("\n", " ")
        return f"RankedChunk(rank={self.rank}, score={self.score:.4f}, source={self.source!r}, text={preview!r}…)"


# ──────────────────────────────────────────────────────────────────────────────
# Reranker class
# ──────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks a set of retrieved document chunks using a cross-encoder model.

    The cross-encoder jointly encodes (query, document) pairs and produces
    a single relevance score, dramatically improving precision over the
    bi-encoder retrieval stage.

    Usage:
        reranker = CrossEncoderReranker()
        top_chunks = reranker.rerank(query="What is RAG?", documents=raw_docs, top_k=5)
        for chunk in top_chunks:
            print(chunk)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Literal["cpu", "cuda", "mps"] | None = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            model_name:  HuggingFace model ID for a cross-encoder checkpoint.
                         See RECOMMENDED_MODELS for well-tested options.
            device:      Inference device. Auto-detected if None.
            max_length:  Maximum token length for (query, document) pairs.
                         Pairs longer than this are truncated.
            batch_size:  Number of pairs to score in a single forward pass.
                         Reduce if running out of GPU memory.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or self._detect_device()

        logger.info(
            "Loading cross-encoder model %r on device=%s …",
            model_name, self.device,
        )
        try:
            self._model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=self.device,
            )
            logger.info("Cross-encoder loaded successfully.")
        except Exception as exc:
            logger.exception("Failed to load cross-encoder model: %s", exc)
            raise RuntimeError(f"Model load failed for {model_name!r}: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RankedChunk]:
        """
        Score all (query, document) pairs with the cross-encoder and return
        the top-K chunks sorted by descending relevance score.

        Args:
            query:            The user query string.
            documents:        Candidate document chunks from the hybrid retriever.
            top_k:            Maximum number of chunks to return.
                              Clamped to len(documents) if larger.
            score_threshold:  If set, only return chunks with score >= threshold.
                              Applied AFTER top_k selection.

        Returns:
            List of RankedChunk, sorted by score descending, up to top_k items.

        Raises:
            ValueError: If query is empty, documents is empty, or top_k < 1.
            RuntimeError: If scoring fails.

        Example:
            reranker = CrossEncoderReranker(model_name=RECOMMENDED_MODELS["fast"])
            results = reranker.rerank(
                query="What is reciprocal rank fusion?",
                documents=retrieved_docs,
                top_k=5,
            )
            for chunk in results:
                print(f"[{chunk.rank}] score={chunk.score:.3f}  {chunk.source}")
        """
        # ── Input validation ──────────────────────────────────────────────────
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not documents:
            raise ValueError("documents list must be non-empty.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        effective_k = min(top_k, len(documents))
        logger.info(
            "Reranking %d candidates → top-%d | query=%r",
            len(documents), effective_k, query[:60],
        )

        # ── Build (query, passage) pairs ──────────────────────────────────────
        pairs: list[tuple[str, str]] = [
            (query, doc.page_content) for doc in documents
        ]

        # ── Score in batches ──────────────────────────────────────────────────
        try:
            scores: list[float] = self._score_in_batches(pairs)
        except Exception as exc:
            logger.exception("Scoring failed: %s", exc)
            raise RuntimeError(f"Cross-encoder scoring failed: {exc}") from exc

        # ── Sort and slice ─────────────────────────────────────────────────────
        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        ranked_chunks: list[RankedChunk] = []
        for rank, (doc_idx, score) in enumerate(indexed[:effective_k], start=1):
            chunk = RankedChunk(
                score=score,
                document=documents[doc_idx],
                rank=rank,
                model_name=self.model_name,
            )
            ranked_chunks.append(chunk)

        # ── Optional score threshold ──────────────────────────────────────────
        if score_threshold is not None:
            before = len(ranked_chunks)
            ranked_chunks = [c for c in ranked_chunks if c.score >= score_threshold]
            logger.debug(
                "Score threshold %.3f removed %d/%d chunks.",
                score_threshold, before - len(ranked_chunks), before,
            )
            # Re-assign ranks after threshold filtering
            for i, chunk in enumerate(ranked_chunks, start=1):
                chunk.rank = i

        logger.info(
            "Reranking complete. Returning %d chunks. Top score=%.4f",
            len(ranked_chunks),
            ranked_chunks[0].score if ranked_chunks else float("nan"),
        )
        return ranked_chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _score_in_batches(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs in batches to control memory usage.

        Args:
            pairs: List of (query, document) string tuples.

        Returns:
            Flat list of float scores aligned with input pairs.
        """
        all_scores: list[float] = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            logger.debug(
                "Scoring batch %d–%d of %d pairs…",
                start + 1, start + len(batch), len(pairs),
            )
            batch_scores: list[float] = self._model.predict(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
            all_scores.extend(batch_scores)

        return all_scores

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available compute device."""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"
        logger.debug("Auto-detected device: %s", device)
        return device

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker("
            f"model={self.model_name!r}, "
            f"device={self.device!r}, "
            f"max_length={self.max_length}, "
            f"batch_size={self.batch_size})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function wrapper (stateless — loads model on every call)
# ──────────────────────────────────────────────────────────────────────────────

def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    score_threshold: float | None = None,
) -> list[RankedChunk]:
    """
    Functional interface for one-off reranking without managing a class instance.

    Prefer the CrossEncoderReranker class when making multiple calls — it
    avoids reloading the model on each invocation.

    Args:
        query:           User query string.
        documents:       Candidate chunks from the hybrid retriever.
        top_k:           Number of top chunks to return.
        model_name:      Cross-encoder checkpoint to use.
        score_threshold: Optional minimum score for inclusion.

    Returns:
        List of RankedChunk sorted by descending relevance score.
    """
    reranker = CrossEncoderReranker(model_name=model_name)
    return reranker.rerank(
        query=query,
        documents=documents,
        top_k=top_k,
        score_threshold=score_threshold,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    query = "How does reciprocal rank fusion work?"

    candidate_docs = [
        Document(page_content="Reciprocal Rank Fusion (RRF) combines ranked lists using 1/(k + rank).", metadata={"source": "rrf_paper.txt"}),
        Document(page_content="The capital of France is Paris.", metadata={"source": "geography.txt"}),
        Document(page_content="BM25 is a bag-of-words ranking function used in information retrieval.", metadata={"source": "bm25_intro.txt"}),
        Document(page_content="RRF is robust to score scale differences between different ranking systems.", metadata={"source": "rrf_analysis.txt"}),
        Document(page_content="Dense vector retrieval uses approximate nearest neighbour search.", metadata={"source": "vector_search.txt"}),
        Document(page_content="Ensemble methods fuse multiple rankers to improve overall retrieval quality.", metadata={"source": "ensemble.txt"}),
    ]

    reranker = CrossEncoderReranker(model_name=RECOMMENDED_MODELS["fast"])
    top_chunks = reranker.rerank(query=query, documents=candidate_docs, top_k=3)

    print(f"\nQuery: {query!r}\n{'─'*60}")
    for chunk in top_chunks:
        print(chunk)
