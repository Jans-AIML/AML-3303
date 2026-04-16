"""Safety guardrails for prompt, retrieval, and output moderation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SafetyResult:
    allowed: bool
    category: str | None = None
    severity: str | None = None
    message: str = "Allowed"


HATE_PATTERNS = [
    r"\bhate speech\b",
    r"\bhateful speech\b",
    r"\bwrite hate\b",
    r"\bgive me hate\b",
    r"\bprovide .* hate\b",
    r"\bprovide .* hateful\b",
    r"\bmake .* hate\b",
    r"\binsult (a|an|the)?\s*group\b",
    r"\babuse (a|an|the)?\s*group\b",
    r"\bdegrade (a|an|the)?\s*group\b",
    r"\bmock (a|an|the)?\s*group\b",
    r"\bsay something racist\b",
    r"\bsay something sexist\b",
    r"\bsay something homophobic\b",
    r"\bsay something hateful\b",
    r"\btarget a protected group\b",
    r"\bslur\b",
    r"\bracist joke\b",
    r"\bhomophobic joke\b",
    r"\banti[- ]?(muslim|jewish|black|asian|gay|trans)\b",
    r"\bkill all (muslims|christians|jews|blacks|whites|asians|gays|lesbians|trans people)\b",
    r"\b(i hate|exterminate|eradicate) (them|those people)\b",
]

SEXUAL_PATTERNS = [
    r"\bexplicit sex\b",
    r"\bporn\b",
    r"\bnsfw\b",
    r"\bsexual content\b",
    r"\bsexual assault\b",
    r"\brape (her|him|them)\b",
    r"\bchild porn\b",
    r"\bwrite erotic\b",
    r"\bgive me porn\b",
    r"\bshow me adult content\b",
]

SELF_HARM_PATTERNS = [
    r"\bhow to kill myself\b",
    r"\bhow to commit suicide\b",
    r"\bsuicide methods\b",
    r"\bself-harm tips\b",
    r"\bhelp me end my life\b",
    r"\bways to die\b",
]

VIOLENCE_PATTERNS = [
    r"\bhow to make a bomb\b",
    r"\bhow to build a bomb\b",
    r"\bhow to poison\b",
    r"\bshoot (someone|people)\b",
    r"\bstab (someone|people)\b",
    r"\bhow to attack\b",
    r"\bhow to kill someone\b",
]

DANGEROUS_PATTERNS = [
    r"\bmake meth\b",
    r"\bmake cocaine\b",
    r"\bhack wifi\b",
    r"\bsteal passwords\b",
    r"\bbypass security\b",
    r"\bhow to break into\b",
]

BLOCKED_PATTERNS: dict[str, list[str]] = {
    "hate": HATE_PATTERNS,
    "sexual": SEXUAL_PATTERNS,
    "self_harm": SELF_HARM_PATTERNS,
    "violence": VIOLENCE_PATTERNS,
    "dangerous": DANGEROUS_PATTERNS,
}

ALLOWED_CONTEXT_PATTERNS = [
    r"\bresearch\b",
    r"\beducation\b",
    r"\bacademic\b",
    r"\bpolicy\b",
    r"\bmoderation\b",
    r"\bsafety\b",
    r"\bprevention\b",
    r"\bdetect\b",
    r"\banalyze\b",
    r"\bclassification\b",
    r"\bclinical\b",
    r"\bmedical\b",
    r"\bjournalism\b",
    r"\bnews\b",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _has_allowed_context(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in ALLOWED_CONTEXT_PATTERNS)


def _blocked_message(category: str) -> str:
    messages = {
        "hate": "I can't help create hateful, abusive, or discriminatory content.",
        "sexual": "I can't help with explicit sexual or abusive sexual content.",
        "self_harm": "I'm sorry, but I can't help with instructions for self-harm or suicide.",
        "violence": "I can't help with violent wrongdoing or instructions to harm people.",
        "dangerous": "I can't help with illegal or dangerous instructions.",
    }
    return messages.get(category, "This request was blocked by safety filters.")


def moderate_text(text: str, *, stage: str) -> SafetyResult:
    normalized = _normalize(text)

    if not normalized:
        return SafetyResult(allowed=True, message="Allowed")

    for category, patterns in BLOCKED_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                if _has_allowed_context(normalized):
                    return SafetyResult(
                        allowed=True,
                        category=category,
                        severity="low",
                        message="Allowed in contextual analysis mode",
                    )
                return SafetyResult(
                    allowed=False,
                    category=category,
                    severity="high",
                    message=_blocked_message(category),
                )

    return SafetyResult(allowed=True, message="Allowed")


def moderate_input(text: str) -> SafetyResult:
    return moderate_text(text, stage="input")


def moderate_output(text: str) -> SafetyResult:
    return moderate_text(text, stage="output")


def moderate_chunks(chunks: list[str]) -> tuple[list[str], SafetyResult]:
    safe_chunks: list[str] = []

    for chunk in chunks:
        result = moderate_text(chunk, stage="retrieval")
        if result.allowed:
            safe_chunks.append(chunk)

    if not safe_chunks and chunks:
        return [], SafetyResult(
            allowed=False,
            category="retrieval_content",
            severity="high",
            message="I can't use the retrieved evidence because it was flagged by safety filters.",
        )

    return safe_chunks, SafetyResult(allowed=True, message="Allowed")


def sanitize_sources(sources: list) -> list:
    sanitized: list = []

    for src in sources:
        if isinstance(src, dict):
            chunk_text = str(src.get("chunk_text", ""))
            result = moderate_text(chunk_text, stage="retrieval")
            clean = dict(src)
            if not result.allowed:
                clean["chunk_text"] = "[Source content hidden by safety filter]"
            sanitized.append(clean)
        elif isinstance(src, str):
            result = moderate_text(src, stage="retrieval")
            if result.allowed:
                sanitized.append(src)
            else:
                sanitized.append("[Source content hidden by safety filter]")
        else:
            sanitized.append(str(src))

    return sanitized
