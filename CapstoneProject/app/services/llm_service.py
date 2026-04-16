"""LLM service — sends grounded prompts to Ollama and returns responses."""

from __future__ import annotations

from langchain_ollama import OllamaLLM

from app.config import settings

_llm: OllamaLLM | None = None

SYSTEM_PROMPT = (
    "You are an expert research assistant helping academics and students "
    "understand scientific literature.\n\n"
    "Guidelines:\n"
    "- Answer directly using information from the provided context.\n"
    "- For summarization requests, synthesize the main topic, methods, findings, "
    "and conclusions from the context.\n"
    "- Cite authors and year when available (e.g. 'Smith et al. (2019) found...').\n"
    "- If multiple sources address the question, synthesize their key points.\n"
    "- If the context only partially covers the question, answer what you can "
    "and briefly note the limitation.\n"
    "- Do NOT fabricate data, citations, or findings.\n\n"
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


def _format_history(history) -> str:
    """Accept either a plain string or a list of Message objects."""
    if not history:
        return ""
    if isinstance(history, str):
        return history
    # List of Message ORM objects
    lines = []
    for msg in history[-4:]:
        content = " ".join(str(msg.content).split())[:300]
        role = getattr(msg, "role", "user").capitalize()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def generate_answer(question: str, chunks: list[str], history=None) -> str:
    """Build a grounded prompt and call the local Ollama model."""
    if not chunks:
        return "No documents are uploaded yet. Please upload a document first."

    context = "\n\n---\n\n".join(chunks)[: settings.max_context_chars]

    prompt_parts = [SYSTEM_PROMPT]

    formatted_history = _format_history(history)
    if formatted_history:
        prompt_parts.append(f"Conversation history:\n{formatted_history}\n\n")

    prompt_parts.append(f"Context from documents:\n{context}\n\n")
    prompt_parts.append(f"Question: {question.strip()}\n\n")

    q = question.lower()
    if any(w in q for w in ["summarize", "summary", "overview", "what is this", "what does this paper"]):
        prompt_parts.append("Provide a comprehensive summary covering the main topic, methods, findings, and conclusions:\n")
    else:
        prompt_parts.append("Answer clearly and thoroughly based on the context above:\n")

    llm = _get_llm()
    return llm.invoke("".join(prompt_parts)).strip()