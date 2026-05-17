from langchain_core.documents import Document

def build_prompt(query: str, chunks: list[Document]) -> str:
    context = ""
    for i, doc in enumerate(chunks, 1):
        source = doc.metadata.get("source", f"chunk_{i}")
        context += f"[{i}][{source}]\n{doc.page_content}\n\n"

    return f"""You are a precise question-answering assistant.
Answer the question using ONLY the context below.
For every fact you state, cite the source like this: [1], [2], etc.
If the answer is not in the context, say "I don't have enough information."

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with citations):"""
