"""
horizon_langchain_obj.py — Horizon LLM LangChain Wrapper
==========================================================
HorizonLlmChat aur TokenManager ka e360 project mein use.

Project A mein directly `from horizon_langchain import HorizonLlmChat, TokenManager`
kiya gaya. Yeh file local implementation provide karta hai
jab horizon_langchain package available nahi ho.
"""

from __future__ import annotations

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class TokenManager:
    """
    OAuth token manager for Horizon API.
    Client credentials flow — token cache + auto-refresh.
    """

    TOKEN_URL = "https://api.horizon.elevancehealth.com/oauth/token"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        """Get valid token — refresh if expired."""
        if self._token and time.time() < self._expires_at - 30:
            return self._token
        return self._refresh_token()

    def _refresh_token(self) -> str:
        """Fetch new token from Horizon OAuth endpoint."""
        try:
            resp = requests.post(
                self.TOKEN_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token      = data["access_token"]
            expires_in       = data.get("expires_in", 3600)
            self._expires_at = time.time() + expires_in
            logger.info("Horizon token refreshed successfully")
            return self._token
        except Exception as exc:
            logger.error(f"Token refresh failed: {exc}")
            raise RuntimeError(f"Failed to get Horizon token: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES (LangChain compatible)
# ══════════════════════════════════════════════════════════════════════════════

class AIMessage:
    """LangChain AIMessage compatible response wrapper."""
    def __init__(self, content: str):
        self.content = content

    def __repr__(self):
        return f"AIMessage(content={self.content[:80]!r}...)"


# ══════════════════════════════════════════════════════════════════════════════
# HORIZON LLM CHAT
# ══════════════════════════════════════════════════════════════════════════════

class HorizonLlmChat:
    """
    LangChain-style wrapper for Horizon Elevance Health LLM API.

    Usage:
        token_manager = TokenManager(client_id, client_secret)
        llm = HorizonLlmChat(
            api_endpoint="https://api.horizon.elevancehealth.com/v2/text/chats",
            token_manager=token_manager
        )
        response = llm.invoke("Summarize this email: ...")
        print(response.content)
    """

    DEFAULT_MODEL    = "gpt-4o"
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.1    # Low temp for consistent structured output

    def __init__(
        self,
        api_endpoint: str,
        token_manager: TokenManager,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = 60,
    ):
        self.api_endpoint  = api_endpoint
        self.token_manager = token_manager
        self.model         = model
        self.max_tokens    = max_tokens
        self.temperature   = temperature
        self.timeout       = timeout

    def invoke(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system_prompt: Optional[str] = None,
    ) -> AIMessage:
        """
        LLM call karo.

        Args:
            prompt: String ya messages list
            system_prompt: Optional system message

        Returns:
            AIMessage with .content (string)
        """
        messages = self._build_messages(prompt, system_prompt)
        token    = self.token_manager.get_token()

        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

        try:
            resp = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data    = resp.json()
            content = self._extract_content(data)
            return AIMessage(content=content)

        except requests.HTTPError as exc:
            logger.error(f"Horizon API HTTP error: {exc} — {exc.response.text[:200]}")
            raise
        except Exception as exc:
            logger.error(f"Horizon API call failed: {exc}")
            raise

    def _build_messages(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        """Build messages array for the API."""
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            # Already formatted list
            messages.extend(prompt)

        return messages

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """Extract text from API response."""
        try:
            # Standard OpenAI-compatible format
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            pass
        try:
            # Alternative format
            return data["content"]
        except KeyError:
            pass
        # Fallback
        return json.dumps(data)
