"""LLM service — sends grounded prompts to Ollama and returns responses."""

from __future__ import annotations

from langchain_ollama import OllamaLLM

from app.config import settings

_llm: OllamaLLM | None = None

SYSTEM_PROMPT = (
    "You are a helpful, professional research support assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain enough information to answer, respond with: "
    "'I don't have enough information in the uploaded documents to answer that.' "
    "Be concise, accurate, and professional.\n\n"
)


def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )
    return _llm


def generate_answer(question: str, chunks: list[str], history: str = "") -> str:
    """Build a grounded prompt and call the local Ollama model."""
    if not chunks:
        return "I don't have enough information in the uploaded documents to answer that."

    context = "\n\n---\n\n".join(chunks)[: settings.max_context_chars]

    prompt_parts = [SYSTEM_PROMPT]
    if history:
        prompt_parts.append(f"Conversation history:\n{history}\n\n")
    prompt_parts.append(f"Context from documents:\n{context}\n\n")
    prompt_parts.append(f"Question: {question}\n\nAnswer:")

    full_prompt = "".join(prompt_parts)

    llm = _get_llm()
    return llm.invoke(full_prompt)
