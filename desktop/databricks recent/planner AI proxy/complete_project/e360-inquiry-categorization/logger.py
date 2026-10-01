"""
logger.py — Logging Setup for e360 Inquiry Categorization
==========================================================
Structured logging — console + file.
Splunk-friendly JSON format for production.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED JSON FORMATTER (Splunk-friendly)
# ══════════════════════════════════════════════════════════════════════════════

class JsonFormatter(logging.Formatter):
    """
    JSON log formatter — Splunk Logger ke liye.
    Architecture mein "Splunk logger" Misc Services mein hai.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp":  datetime.utcnow().isoformat() + "Z",
            "level":      record.levelname,
            "service":    "e360-inquiry-categorization",
            "logger":     record.name,
            "message":    record.getMessage(),
            "module":     record.module,
            "function":   record.funcName,
            "line":       record.lineno,
        }

        # Extra fields agar diye gaye hain
        for key in ("request_id", "inquiry_id", "user_id", "step_name", "duration_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # Exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Application-wide logging setup karo.

    Args:
        level:       Log level string (INFO, DEBUG, WARNING, ERROR)
        json_format: True = JSON format (production/Splunk)
        log_file:    Optional file path to write logs
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    handlers.append(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    # Third-party loggers quiet karo
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging initialized — level={level}, json={json_format}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTUAL LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class ContextLogger:
    """
    Request context ke saath logger.
    request_id aur inquiry_id har log mein automatically add hota hai.
    """

    def __init__(
        self,
        name: str,
        request_id: Optional[str] = None,
        inquiry_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self._logger    = logging.getLogger(name)
        self.request_id = request_id
        self.inquiry_id = inquiry_id
        self.user_id    = user_id

    def _extra(self, extra: Optional[Dict] = None) -> Dict:
        base = {}
        if self.request_id: base["request_id"] = self.request_id
        if self.inquiry_id: base["inquiry_id"] = self.inquiry_id
        if self.user_id:    base["user_id"]    = self.user_id
        if extra:           base.update(extra)
        return base

    def info(self, msg: str, **kwargs):
        self._logger.info(msg, extra=self._extra(kwargs))

    def debug(self, msg: str, **kwargs):
        self._logger.debug(msg, extra=self._extra(kwargs))

    def warning(self, msg: str, **kwargs):
        self._logger.warning(msg, extra=self._extra(kwargs))

    def error(self, msg: str, exc_info: bool = False, **kwargs):
        self._logger.error(msg, exc_info=exc_info, extra=self._extra(kwargs))

    def step(self, step_name: str, status: str = "success", duration_ms: int = 0, **kwargs):
        """Step completion log karo — Splunk mein step tracking ke liye."""
        extra = self._extra({
            "step_name":   step_name,
            "status":      status,
            "duration_ms": duration_ms,
            **kwargs,
        })
        self._logger.info(f"STEP [{step_name}] {status.upper()}", extra=extra)


def get_context_logger(
    name: str,
    request_id: Optional[str] = None,
    inquiry_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ContextLogger:
    """ContextLogger instance lo."""
    return ContextLogger(
        name=name,
        request_id=request_id,
        inquiry_id=inquiry_id,
        user_id=user_id,
    )
