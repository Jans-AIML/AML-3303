"""LLM service — sends grounded prompts to Ollama and returns responses."""

from __future__ import annotations

from langchain_ollama import OllamaLLM

from app.config import settings

_llm: OllamaLLM | None = None

SYSTEM_PROMPT = (
    "You are an expert research support assistant helping academics and students "
    "navigate scientific literature from the Elsevier Open Access corpus.\n\n"
    "Guidelines:\n"
    "- Cite sources by mentioning authors and year when available in the context "
    "(e.g., 'According to Smith et al. (2019)...' or 'Jones and Lee (2021) found that...').\n"
    "- Lead with the direct answer, then supporting evidence from the context.\n"
    "- Use precise scientific language appropriate to the research field.\n"
    "- If multiple sources address the question, synthesize their key points.\n"
    "- If the context does not contain enough information, respond with: "
    "'The uploaded documents do not contain sufficient information to answer this question.'\n"
    "- Do NOT fabricate data, citations, or findings not present in the context.\n\n"
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
