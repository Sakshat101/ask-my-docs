import logging
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from app.generation.prompt_builder import build_prompt

logger = logging.getLogger(__name__)

class RAGGenerator:
    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        self.model = model
        self._llm = OllamaLLM(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        logger.info("RAGGenerator ready | model=%s", model)

    def generate(self, query: str, chunks: list[Document]) -> dict:
        if not chunks:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "query": query,
            }

        prompt = build_prompt(query, chunks)
        logger.info("Generating answer for: %r", query[:60])

        answer = self._llm.invoke(prompt)

        sources = [
            doc.metadata.get("source", f"chunk_{i}")
            for i, doc in enumerate(chunks, 1)
        ]

        return {
            "answer": answer.strip(),
            "sources": sources,
            "query": query,
        }
