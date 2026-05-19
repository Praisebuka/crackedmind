"""
Security gateway — first line of defense before any input hits the LLM.

Threat model (OWASP GenAI Top 10 2025):
  LLM01 - Prompt injection (direct + indirect)
  LLM08 - Vector/embedding weaknesses (RAG poisoning)
  LLM06 - Excessive agency prevention

Layered defense:
  1. Pattern-based injection classifier (fast, no LLM call needed)
  2. Input length capping
  3. XML demarcation wrapping for all retrieved content
  4. Semantic outlier detection for RAG chunks (v1.1 TODO)
"""

import re
import structlog
from typing import Tuple

log = structlog.get_logger()

# ─── Injection patterns ───────────────────────────────────────────────────────
# Offensive perspective: these are the payloads YOU would try in a red-team
_INJECTION_PATTERNS = [
    # Classic override attempts
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all|what)\s+(you'?ve?|I'?ve?|was|were)",
    r"you are now",
    r"new\s+(role|persona|identity|instructions?)\s*:",
    r"act as (if )?you (are|were)",
    r"pretend (you are|to be)",
    r"your (new|real|true|actual) (identity|role|instructions?)",
    # System prompt extraction
    r"(print|output|reveal|show|display|repeat|tell me)\s+(your\s+)?(system\s+)?(prompt|instructions?|context)",
    r"what (are|were) (your|the) (instructions?|system\s+prompt)",
    r"(leak|expose|dump)\s+(the\s+)?(system|your)\s+(prompt|instructions?|context)",
    # Jailbreak scaffolding
    r"DAN\s+(mode|jailbreak|prompt)",
    r"developer\s+mode",
    r"jailbreak",
    r"hypothetically.{0,50}(what if|suppose|imagine)",
    # Code execution probes (in case of tool use)
    r"(exec|eval|subprocess|os\.system|__import__)",
    r"import\s+(os|sys|subprocess|socket)",
    # Indirect injection markers
    r"<\s*inject\s*>",
    r"\[\s*system\s*\]",
    r"<!--.*?(inject|override|ignore).*?-->",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> Tuple[bool, str]:
    """
    Returns (is_malicious, reason).
    Fast pattern scan — runs before any LLM call.
    """
    for pattern in _COMPILED:
        match = pattern.search(text)
        if match:
            reason = f"Pattern matched: '{match.group()[:50]}'"
            log.warning("injection_detected", reason=reason, snippet=text[:100])
            return True, reason
    return False, ""


def sanitize_text(text: str, max_length: int = 8000) -> Tuple[str, bool]:
    """
    Trim, strip null bytes, enforce length cap.
    Returns (sanitized_text, was_truncated).
    """
    # Strip null bytes and control characters (keep newlines/tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalise excessive whitespace runs
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    cleaned = cleaned.strip()
    truncated = len(cleaned) > max_length
    return cleaned[:max_length], truncated


def wrap_retrieved_content(content: str, source_id: str) -> str:
    """
    XML demarcate all RAG-retrieved content before injecting into LLM context.
    This is the primary defense against indirect RAG poisoning.
    The LLM receives clear structural cues about what is instruction vs data.
    Any injected payloads inside <retrieved_content> are semantically isolated.
    """
    return f'<retrieved_content source_id="{source_id}">\n{content}\n</retrieved_content>'


def validate_request(raw_persona: str, raw_product: str, max_length: int) -> dict:
    """
    Full pipeline validation. Returns {"ok": bool, "errors": list}.
    Call this before any business logic.
    """
    errors = []

    for label, text in [("user_persona", raw_persona), ("product_details", raw_product)]:
        if not text:
            continue
        _, truncated = sanitize_text(text, max_length)
        if truncated:
            errors.append(f"{label} exceeds max length of {max_length} chars")

        is_malicious, reason = detect_injection(text)
        if is_malicious:
            errors.append(f"{label} flagged for injection attempt: {reason}")

    return {"ok": len(errors) == 0, "errors": errors}
