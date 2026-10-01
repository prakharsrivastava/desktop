"""
MCP Client for Sentiment Analysis

This module provides a proper MCP client implementation to communicate
with the sentiment analysis MCP server using the stdio transport.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPSentimentClient:
    """MCP client for sentiment analysis server."""

    def __init__(self, server_script_path: str = "mcp_sentiment_server.py"):
        """
        Initialize MCP sentiment client.

        Args:
            server_script_path: Path to the MCP server script
        """
        self.server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env={"PYTHONPATH": "."}
        )
        self._session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None

    @asynccontextmanager
    async def connect(self):
        """Context manager to establish MCP connection."""
        try:
            async with stdio_client(self.server_params) as (read, write):
                self._read_stream = read
                self._write_stream = write

                async with ClientSession(read, write) as session:
                    self._session = session

                    # Initialize the session
                    await session.initialize()
                    logger.info("MCP sentiment client connected and initialized")

                    yield self

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
        finally:
            self._session = None
            self._read_stream = None
            self._write_stream = None
            logger.info("MCP sentiment client disconnected")

    async def analyze_email_sentiment(
        self,
        subject: str = "",
        body: str = "",
        context: str = "email"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of email content using MCP server.

        Args:
            subject: Email subject line
            body: Email body content
            context: Optional context about the email

        Returns:
            Dictionary containing sentiment analysis results
        """
        if not self._session:
            raise RuntimeError("MCP client not connected. Use 'async with client.connect()' first.")

        try:
            result = await self._session.call_tool(
                "analyze_email_sentiment",
                arguments={
                    "subject": subject,
                    "body": body,
                    "context": context
                }
            )

            # Parse the result - MCP returns TextContent
            if result.content and len(result.content) > 0:
                text_content = result.content[0].text
                return json.loads(text_content)
            else:
                logger.warning("Empty response from MCP server")
                return self._get_default_sentiment()

        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            raise

    async def analyze_text_sentiment(
        self,
        text: str,
        context: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of any text content using MCP server.

        Args:
            text: Text content to analyze
            context: Optional context about the text

        Returns:
            Dictionary containing sentiment analysis results
        """
        if not self._session:
            raise RuntimeError("MCP client not connected. Use 'async with client.connect()' first.")

        try:
            result = await self._session.call_tool(
                "analyze_text_sentiment",
                arguments={
                    "text": text,
                    "context": context
                }
            )

            # Parse the result
            if result.content and len(result.content) > 0:
                text_content = result.content[0].text
                return json.loads(text_content)
            else:
                logger.warning("Empty response from MCP server")
                return self._get_default_sentiment()

        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            raise

    async def batch_sentiment_analysis(
        self,
        items: list[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for multiple text items at once.

        Args:
            items: List of items with 'id', 'text', and optional 'context'

        Returns:
            Dictionary containing batch sentiment analysis results
        """
        if not self._session:
            raise RuntimeError("MCP client not connected. Use 'async with client.connect()' first.")

        try:
            result = await self._session.call_tool(
                "batch_sentiment_analysis",
                arguments={"items": items}
            )

            # Parse the result
            if result.content and len(result.content) > 0:
                text_content = result.content[0].text
                return json.loads(text_content)
            else:
                logger.warning("Empty response from MCP server")
                return {"results": []}

        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            raise

    async def get_resources(self) -> list:
        """Get available resources from MCP server."""
        if not self._session:
            raise RuntimeError("MCP client not connected. Use 'async with client.connect()' first.")

        try:
            resources = await self._session.list_resources()
            return resources.resources
        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            raise

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource from MCP server.

        Args:
            uri: Resource URI (e.g., "sentiment://help")

        Returns:
            Resource content as string
        """
        if not self._session:
            raise RuntimeError("MCP client not connected. Use 'async with client.connect()' first.")

        try:
            result = await self._session.read_resource(uri)
            if result.contents and len(result.contents) > 0:
                return result.contents[0].text
            return ""
        except Exception as e:
            logger.error(f"Error reading resource: {e}")
            raise

    @staticmethod
    def _get_default_sentiment() -> Dict[str, Any]:
        """Return default neutral sentiment."""
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "emotional_tone": "neutral",
            "urgency_level": "low",
            "key_indicators": [],
            "healthcare_flags": {
                "medical_urgency": False,
                "insurance_frustration": False,
                "health_anxiety": False,
                "service_appreciation": False
            }
        }


# Singleton instance for reuse
_mcp_client_instance: Optional[MCPSentimentClient] = None


def get_mcp_client(server_script_path: str = "mcp_sentiment_server.py") -> MCPSentimentClient:
    """
    Get or create MCP sentiment client instance.

    Args:
        server_script_path: Path to the MCP server script

    Returns:
        MCPSentimentClient instance
    """
    global _mcp_client_instance
    if _mcp_client_instance is None:
        _mcp_client_instance = MCPSentimentClient(server_script_path)
    return _mcp_client_instance


async def analyze_sentiment_with_mcp(
    text: str,
    context: str = "email",
    fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze sentiment using MCP client.

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
