from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

DEFAULT_PRIORITY_LEVELS: List[str] = [
    "Standard",
    "Access to Care",
    "Urgent",
]


def _parse_priority_levels(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_PRIORITY_LEVELS.copy()
    levels = [item.strip() for item in raw.split(",") if item.strip()]
    return levels or DEFAULT_PRIORITY_LEVELS.copy()


@dataclass
class AppConfig:
    """Configuration values for the email intelligence service."""

    model_provider: str = field(
        default_factory=lambda: os.getenv("MODEL_PROVIDER", "none").lower()
    )
    model_api_key: str | None = field(
        default_factory=lambda: os.getenv("MODEL_API_KEY")
    )
    priority_levels: List[str] = field(
        default_factory=lambda: _parse_priority_levels(os.getenv("PRIORITY_LEVELS"))
    )
    include_debug_info: bool = field(
        default_factory=lambda: os.getenv("INCLUDE_DEBUG_INFO", "")
        .strip()
        .lower()
        == "true"
    )
    max_summary_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUMMARY_LENGTH", "400"))
    )
    enable_sentiment_integration: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SENTIMENT_INTEGRATION", "true")
        .strip()
        .lower()
        == "true"
    )


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return a cached AppConfig instance initialized from environment variables."""

    global _config
    if _config is None:
        _config = AppConfig()
    return _config
