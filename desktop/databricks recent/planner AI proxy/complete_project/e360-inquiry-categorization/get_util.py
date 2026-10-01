"""
get_util.py — Utility helpers for e360 Inquiry Categorization
==============================================================
General-purpose utilities: env loading, retry logic, JSON parsing,
response validation.
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
from typing import Any, Dict, Optional, TypeVar, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════════════════
# ENV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Environment variable safely lo.

    Args:
        key:      Env variable name
        default:  Default value if not set
        required: True → raise if missing
    """
    value = os.environ.get(key, default)
    if required and not value:
        raise EnvironmentError(f"Required env variable '{key}' is not set.")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Boolean env variable lo."""
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Integer env variable lo."""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# JSON HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_parse_json(text: str, fallback: Any = None) -> Any:
    """
    JSON safely parse karo — backticks aur extra whitespace bhi handle karo.

    LLM responses mein kabhi kabhi ```json ... ``` aata hai.
    """
    if not text:
        return fallback

    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting first JSON object/array
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    logger.warning(f"JSON parse failed for text: {text[:100]!r}")
    return fallback


def to_json_str(obj: Any, indent: Optional[int] = None) -> str:
    """Object ko JSON string mein convert karo."""
    return json.dumps(obj, default=str, indent=indent, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# RETRY LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def retry(
    func: Callable[..., T],
    *args,
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    **kwargs,
) -> T:
    """
    Function ko retry karo on failure.

    Args:
        func:           Function to call
        max_attempts:   Total attempts
        delay_seconds:  Initial delay
        backoff_factor: Multiply delay on each retry
        exceptions:     Which exceptions to catch

    Returns:
        Function result on success

    Raises:
        Last exception if all attempts fail
    """
    last_exc: Optional[Exception] = None
    delay = delay_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed: {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_attempts} attempts failed. Last error: {exc}")

    raise last_exc


# ══════════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Text ko max_length pe truncate karo."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Extra whitespace aur special chars clean karo."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_emails(text: str) -> list[str]:
    """Text se email addresses extract karo."""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    return list(set(re.findall(pattern, text)))


def extract_hcid(text: str) -> list[str]:
    """
    Text se HCID numbers extract karo.
    Pattern: alphanumeric 8-12 chars (common healthcare ID format).
    """
    pattern = r"\b[A-Z]{2,4}\d{6,10}\b"
    return list(set(re.findall(pattern, text.upper())))


def extract_phone_numbers(text: str) -> list[str]:
    """Phone numbers extract karo."""
    pattern = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    return list(set(re.findall(pattern, text)))


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST ID HELPER
# ══════════════════════════════════════════════════════════════════════════════

def generate_request_id(prefix: str = "e360") -> str:
    """Unique request ID generate karo."""
    import uuid
    return f"{prefix}-{uuid.uuid4()}"


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

VALID_CATEGORIES = [
    "ID Card Change Request",
    "Address Change Request",
    "General Enquiry",
    "Billing Inquiry",
    "Claims Inquiry",
    "Benefits Inquiry",
    "Provider Inquiry",
    "Other",
]

VALID_PRIORITIES = ["Standard", "Access to Care", "Urgent"]

SLA_MAP: Dict[str, int] = {
    "Urgent":          4,
    "Access to Care":  8,
    "Standard":       24,
}

ROUTING_MAP: Dict[str, str] = {
    "ID Card Change Request": "PEGA_ID_CARD_QUEUE",
    "Address Change Request": "PEGA_ADDRESS_QUEUE",
    "General Enquiry":        "PEGA_INQUIRY_QUEUE",
    "Billing Inquiry":        "PEGA_BILLING_QUEUE",
    "Claims Inquiry":         "PEGA_CLAIMS_QUEUE",
    "Benefits Inquiry":       "PEGA_BENEFITS_QUEUE",
    "Provider Inquiry":       "PEGA_PROVIDER_QUEUE",
    "Other":                  "PEGA_GENERAL_QUEUE",
}


def validate_category(category: str) -> str:
    """Category validate karo — invalid toh 'Other' return karo."""
    if category in VALID_CATEGORIES:
        return category
    logger.warning(f"Invalid category '{category}', defaulting to 'Other'")
    return "Other"


def validate_priority(priority: str) -> str:
    """Priority validate karo — invalid toh 'Standard' return karo."""
    if priority in VALID_PRIORITIES:
        return priority
    logger.warning(f"Invalid priority '{priority}', defaulting to 'Standard'")
    return "Standard"


def get_sla_hours(priority: str) -> int:
    """Priority ke liye SLA hours lo."""
    return SLA_MAP.get(priority, 24)


def get_pega_queue(category: str) -> str:
    """Category ke liye PEGA queue name lo."""
    return ROUTING_MAP.get(category, "PEGA_GENERAL_QUEUE")
