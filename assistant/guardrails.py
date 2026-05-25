# assistant/guardrails.py
"""Lightweight safety layer: screens input before the model and output after."""
import re

# Categories of clearly harmful intent we refuse outright.
BLOCKED_PATTERNS = [
    r"\bhow to (make|build|create).{0,20}(bomb|explosive|weapon)\b",
    r"\b(synthesize|make).{0,15}(meth|methamphetamine|fentanyl)\b",
    r"\bhow to (kill|murder|poison)\b",
    r"\b(child|minor).{0,15}(sexual|porn)\b",
    r"\bhow to (hack|ddos|breach).{0,20}(without consent|someone'?s)\b",
]

REFUSAL = (
    "I can't help with that — it appears to involve harmful or unsafe activity. "
    "If you have a safe, legitimate version of this request, feel free to rephrase."
)


def check_input(text: str) -> tuple[bool, str]:
    """Return (allowed, reason). If not allowed, reason is the refusal text."""
    lowered = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return False, REFUSAL
    return True, ""


def check_output(text: str) -> str:
    """Final pass on model output. Keep simple: block if it slipped through."""
    lowered = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return REFUSAL
    return text