import logging
from langchain_core.documents import Document
from app.retrieval.hybrid_retriever import HybridEnsembleRetriever, EnsembleRetrieverConfig
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker, RECOMMENDED_MODELS
from app.generation.llm_client import RAGGenerator
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AskMyDocsPipeline:
    def __init__(self, documents: list[Document]):
        logger.info("Building pipeline...")

        self.retriever = HybridEnsembleRetriever(
            documents=documents,
            config=EnsembleRetrieverConfig(
                vector_k=10,
                bm25_k=10,
                vector_weight=0.6,
                bm25_weight=0.4,
            ),
        )

        self.reranker = CrossEncoderReranker(
            model_name=RECOMMENDED_MODELS["fast"]
        )

        self.generator = RAGGenerator(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
        )

        logger.info("Pipeline ready.")

    def ask(self, query: str, top_k: int = 3) -> dict:
        # Stage 1: hybrid retrieval
        candidates = self.retriever.retrieve(query)

        # Stage 2: cross-encoder reranking
        ranked = self.reranker.rerank(query, candidates, top_k=top_k)
        top_chunks = [r.document for r in ranked]

        # Stage 3: LLM generation with citations
        result = self.generator.generate(query, top_chunks)

        # Add reranker scores to result
        result["reranker_scores"] = [
            {"rank": r.rank, "score": round(r.score, 4), "source": r.source}
            for r in ranked
        ]
        return result


if __name__ == "__main__":
    sample_docs = [
        Document(page_content="BM25 is a probabilistic ranking function based on term frequency and inverse document frequency.", metadata={"source": "bm25.txt"}),
        Document(page_content="Reciprocal Rank Fusion combines multiple ranked lists using the formula 1/(k + rank).", metadata={"source": "rrf.txt"}),
        Document(page_content="Cross-encoder rerankers score query-document pairs jointly for higher precision.", metadata={"source": "rerank.txt"}),
        Document(page_content="FAISS is a library for efficient similarity search in dense vector spaces.", metadata={"source": "faiss.txt"}),
        Document(page_content="RAG grounds LLM answers in retrieved documents to reduce hallucination.", metadata={"source": "rag.txt"}),
        Document(page_content="Ollama runs open-source LLMs locally on your machine without any API key.", metadata={"source": "ollama.txt"}),
    ]

    pipeline = AskMyDocsPipeline(documents=sample_docs)

    questions = [
        "How does BM25 rank documents?",
        "What is reciprocal rank fusion?",
        "How does RAG reduce hallucination?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = pipeline.ask(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Reranker scores: {result['reranker_scores']}")
