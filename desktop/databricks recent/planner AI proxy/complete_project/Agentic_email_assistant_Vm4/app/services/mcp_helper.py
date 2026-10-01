"""
MCP Helper Functions

Provides synchronous wrappers for MCP client operations to use in FastAPI endpoints.
"""

import asyncio
import logging
from typing import Dict, Any

from app.services.mcp_client import get_mcp_client, MCPSentimentClient

logger = logging.getLogger(__name__)


def analyze_sentiment_sync(
    text: str,
    context: str = "email",
    fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for MCP sentiment analysis.

    Args:
        text: Text to analyze
        context: Context for analysis
        fallback_on_error: Return default sentiment on error instead of raising

    Returns:
        Sentiment analysis results
    """
    client = get_mcp_client()

    try:
        # Run async code in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _analyze():
                async with client.connect():
                    return await client.analyze_text_sentiment(text, context)

            result = loop.run_until_complete(_analyze())
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"MCP sentiment analysis failed: {e}")
        if fallback_on_error:
            return MCPSentimentClient._get_default_sentiment()
        raise


def analyze_email_sentiment_sync(
    subject: str = "",
    body: str = "",
    context: str = "email",
    fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for MCP email sentiment analysis.

    Args:
        subject: Email subject line
        body: Email body content
        context: Context for analysis
        fallback_on_error: Return default sentiment on error instead of raising

    Returns:
        Sentiment analysis results
    """
    client = get_mcp_client()

    try:
        # Run async code in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _analyze():
                async with client.connect():
                    return await client.analyze_email_sentiment(subject, body, context)

            result = loop.run_until_complete(_analyze())
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"MCP email sentiment analysis failed: {e}")
        if fallback_on_error:
            return MCPSentimentClient._get_default_sentiment()
        raise


async def analyze_sentiment_async(
    text: str,
    context: str = "email",
    fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Async wrapper for MCP sentiment analysis (for async FastAPI endpoints).

    Args:
        text: Text to analyze
        context: Context for analysis
        fallback_on_error: Return default sentiment on error instead of raising

    Returns:
        Sentiment analysis results
    """
    client = get_mcp_client()

    try:
        async with client.connect():
            result = await client.analyze_text_sentiment(text, context)
            return result
    except Exception as e:
        logger.error(f"MCP sentiment analysis failed: {e}")
        if fallback_on_error:
            return MCPSentimentClient._get_default_sentiment()
        raise


async def analyze_email_sentiment_async(
    subject: str = "",
    body: str = "",
    context: str = "email",
    fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Async wrapper for MCP email sentiment analysis (for async FastAPI endpoints).

    Args:
        subject: Email subject line
        body: Email body content
        context: Context for analysis
        fallback_on_error: Return default sentiment on error instead of raising

    Returns:
        Sentiment analysis results
    """
    client = get_mcp_client()

    try:
        async with client.connect():
            result = await client.analyze_email_sentiment(subject, body, context)
            return result
    except Exception as e:
        logger.error(f"MCP email sentiment analysis failed: {e}")
        if fallback_on_error:
            return MCPSentimentClient._get_default_sentiment()
        raise
