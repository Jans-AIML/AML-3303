"""LLM service — sends grounded prompts to Ollama and returns responses."""

from __future__ import annotations

from langchain_ollama import OllamaLLM

from app.config import settings

_llm: OllamaLLM | None = None

SYSTEM_PROMPT = (
    "You are an expert research support assistant helping academics and students "
    "navigate scientific literature from the Elsevier Open Access corpus.\n\n"
    "Guidelines:\n"
    "- Answer directly first, then support with evidence from the provided context.\n"
    "- Cite sources by mentioning authors and year when available in the context.\n"
    "- Keep answers concise and grounded only in the retrieved text.\n"
    "- If the context is insufficient, say so clearly.\n"
    "- Do NOT fabricate data, citations, or findings.\n\n"
)


def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=min(settings.llm_max_tokens, 384),
        )
    return _llm


def _format_history(history: list) -> str:
    if not history:
        return ""
    compact_turns = []
    for msg in history[-4:]:
        content = " ".join(str(msg.content).split())[:300]
        compact_turns.append(f"{msg.role.capitalize()}: {content}")
    return "\n".join(compact_turns)


def _prepare_context(chunks: list[str]) -> str:
    compact_chunks: list[str] = []
    for chunk in chunks[:3]:
        compact = " ".join(chunk.split())
        compact_chunks.append(compact[:1200])
    return "\n\n---\n\n".join(compact_chunks)[: min(settings.max_context_chars, 4000)]


def generate_answer(question: str, chunks: list[str], history: list | None = None) -> str:
    if not chunks:
        return "I don't have enough information in the uploaded documents to answer that."

    context = _prepare_context(chunks)
    prompt_parts = [SYSTEM_PROMPT]

    formatted_history = _format_history(history or [])
    if formatted_history:
        prompt_parts.append(f"Recent conversation:\n{formatted_history}\n\n")

    prompt_parts.append(f"Context from documents:\n{context}\n\n")
    prompt_parts.append(f"Question: {question.strip()}\n\n")
    prompt_parts.append("Answer in 2-5 concise paragraphs:\n")

    llm = _get_llm()
    return llm.invoke("".join(prompt_parts)).strip()