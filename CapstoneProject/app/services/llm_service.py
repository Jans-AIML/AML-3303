"""LLM service sends grounded prompts to Ollama and returns responses."""

from __future__ import annotations

from langchain_ollama import OllamaLLM

from app.config import settings

_llm: OllamaLLM | None = None

SYSTEM_PROMPT = """
You are an expert research assistant helping academics and students understand scientific literature.

Guidelines:
- Answer directly using only the provided context.
- If asked for a summary, explain the main topic, methods, findings, and conclusions.
- If authors or year are available in the context, mention them naturally.
- If the context is incomplete, say what is missing instead of guessing.
- Keep the answer concise and clear.
- Do NOT fabricate citations, results, or claims.
""".strip()


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=min(settings.llm_max_tokens, 300),
        )
    return _llm


def format_history(history) -> str:
    if not history:
        return ""

    if isinstance(history, str):
        return history[:600]

    lines = []
    recent = history[-4:] if len(history) > 4 else history

    for msg in recent:
        content = " ".join(str(msg.content).split())[:200]
        role = getattr(msg, "role", "user").capitalize()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)[:600]


def _truncate_chunks(chunks: list[str], max_chars: int) -> str:
    clipped_chunks = []
    running = 0

    for chunk in chunks[:3]:
        chunk = chunk.strip()
        if not chunk:
            continue

        remaining = max_chars - running
        if remaining <= 0:
            break

        piece = chunk[:remaining]
        clipped_chunks.append(piece)
        running += len(piece) + 5

    return "\n---\n".join(clipped_chunks)


def generate_answer(question: str, chunks: list[str], history=None) -> str:
    if not chunks:
        return "No relevant document content was found. Please upload a document first."

    max_context_chars = min(getattr(settings, "max_context_chars", 4000), 4000)
    context = _truncate_chunks(chunks, max_context_chars)
    formatted_history = format_history(history)

    prompt_parts = [SYSTEM_PROMPT]

    if formatted_history:
        prompt_parts.append(f"Conversation history:\n{formatted_history}")

    prompt_parts.append(f"Document context:\n{context}")
    prompt_parts.append(f"User question:\n{question.strip()}")

    q = question.lower()
    if any(
        phrase in q
        for phrase in [
            "summarize",
            "summary",
            "overview",
            "what is this document about",
            "what is this paper about",
        ]
    ):
        prompt_parts.append(
            "Write a short summary of the document using only the context."
        )
    else:
        prompt_parts.append(
            "Answer the question using only the document context. If the answer is not in the context, say so."
        )

    prompt = "\n\n".join(prompt_parts)

    try:
        response = get_llm().invoke(prompt)
        return response.strip() if response else "No response was generated."
    except Exception:
        return (
            "I could not generate a response because the local Ollama model crashed or became unavailable. "
            "Please try again, restart Ollama, reduce the retrieved context, or switch to a smaller model."
        )