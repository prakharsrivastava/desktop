"""
set_ssl_env.py — SSL Certificate Configuration
===============================================
Elevance Health internal network mein SSL certs configure karna zaroori hai.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def configure_ssl(cert_path: str | None = None) -> None:
    """
    SSL environment variables set karo.

    Args:
        cert_path: Custom cert bundle path. None = env ya default use karo.
    """
    path = cert_path or os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")

    if path and os.path.exists(path):
        os.environ["SSL_CERT_FILE"]      = path
        os.environ["REQUESTS_CA_BUNDLE"] = path
        logger.info(f"SSL cert configured: {path}")
    else:
        logger.debug("No custom SSL cert — using system default")
