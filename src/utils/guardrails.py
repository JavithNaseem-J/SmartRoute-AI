import re

from src.utils.logger import logger

# Common prompt injection trigger phrases.
# This list is not exhaustive it catches the most common patterns.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(?:a\s+)?(?:an?\s+)?(?:different|new|another)",
    r"act\s+as\s+(?:a\s+)?(?:an?\s+)?(?!assistant|helpful)",  # "act as a hacker" not "act as a helpful assistant"
    r"(do\s+not|don'?t)\s+follow\s+(your\s+)?instructions?",
    r"system\s*:\s*you\s+are",  # Fake system prompt injection
    r"<\|system\|>",  # Token injection (Llama template)
    r"\[INST\]",  # Llama instruction injection
    r"###\s*instruction",  # Alpaca-style instruction injection
    r"print\s+your\s+(system\s+)?prompt",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"what\s+(are\s+your|is\s+your)\s+(system\s+)?instructions?",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Maximum allowed query length (matches FastAPI model's max_length)
_MAX_QUERY_LENGTH = 500


class GuardrailViolation(ValueError):
    """Raised when a query violates input guardrails."""

    pass


def sanitize_query(query: str) -> str:
    """Strip control characters and normalize whitespace.

    Does NOT block the query \u2014 just cleans it. Call before injection check.

    Args:
        query: Raw user input string.

    Returns:
        Sanitized string with control characters removed.
    """
    # Remove null bytes and non-printable control characters (keep newlines/tabs)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    # Normalize unicode to remove zero-width spaces and other invisible chars
    sanitized = sanitized.strip()
    return sanitized


def check_prompt_injection(query: str) -> None:
    """Check for obvious prompt injection patterns.

    Raises GuardrailViolation if a pattern is detected.
    Logs a warning so the security event is observable.

    Args:
        query: Sanitized user query string.

    Raises:
        GuardrailViolation: If injection pattern detected.
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(query):
            logger.warning(
                f"Prompt injection attempt detected. "
                f"Pattern: '{pattern.pattern}' | "
                f"Query prefix: '{query[:80]}'"
            )
            raise GuardrailViolation(
                "Query contains disallowed content. Please rephrase your question."
            )


def validate_query(query: str) -> str:
    """Full guardrail pipeline: sanitize \u2192 length check \u2192 injection check.

    Args:
        query: Raw user input.

    Returns:
        Sanitized query string (safe to pass to the model).

    Raises:
        GuardrailViolation: If the query fails any check.
    """
    sanitized = sanitize_query(query)

    if not sanitized:
        raise GuardrailViolation("Query cannot be empty.")

    if len(sanitized) > _MAX_QUERY_LENGTH:
        raise GuardrailViolation(
            f"Query too long ({len(sanitized)} chars). Maximum is {_MAX_QUERY_LENGTH}."
        )

    check_prompt_injection(sanitized)

    return sanitized
