"""
mcp_sentiment_server.py — MCP Sentiment Analysis Server
=========================================================
Architecture mein "Misc Services" layer ka hissa.
MCPSentimentClient (mcp_client.py) yahan connect karta hai.

Tools exposed:
    analyze_email_sentiment  — full email (subject + body)
    analyze_text_sentiment   — any text
    batch_sentiment_analysis — multiple items at once

Resources:
    sentiment://help         — usage guide

Run:
    python mcp_sentiment_server.py
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── MCP Server instance ───────────────────────────────────────────────────────
server = Server("sentiment-analysis-server")


# ══════════════════════════════════════════════════════════════════════════════
# CORE SENTIMENT LOGIC
# (HorizonLLM nahi hai yahan — pattern-based + keyword rules)
# ══════════════════════════════════════════════════════════════════════════════

# Healthcare-specific keyword sets
URGENT_KEYWORDS = [
    "urgent", "asap", "emergency", "critical", "immediate",
    "life-threatening", "danger", "sos", "help", "dying",
]
FRUSTRATED_KEYWORDS = [
    "frustrated", "angry", "disappointed", "unacceptable",
    "terrible", "horrible", "worst", "useless", "ridiculous",
    "complaint", "demand", "refuse", "denied",
]
POSITIVE_KEYWORDS = [
    "thank", "appreciate", "helpful", "excellent", "great",
    "satisfied", "pleased", "wonderful", "amazing", "resolved",
]
MEDICAL_URGENCY_KEYWORDS = [
    "surgery", "hospital", "prescription", "medication", "doctor",
    "appointment", "rx", "procedure", "diagnosis", "treatment",
    "access to care", "specialist",
]
INSURANCE_FRUSTRATION_KEYWORDS = [
    "claim denied", "not covered", "out of pocket", "premium",
    "deductible", "appeal", "grievance", "overpaid", "billing error",
]


def _analyze_sentiment_logic(text: str, context: str = "general") -> Dict[str, Any]:
    """
    Core sentiment analysis — keyword-based + confidence scoring.

    Real implementation mein yahan HorizonLLM call hoga.
    Yeh version pattern matching use karta hai as baseline.
    """
    text_lower = text.lower()
    tokens = set(text_lower.split())

    # ── Urgency check ─────────────────────────────────────────────────────────
    urgent_matches   = [kw for kw in URGENT_KEYWORDS if kw in text_lower]
    positive_matches = [kw for kw in POSITIVE_KEYWORDS if kw in text_lower]
    frustrated_matches = [kw for kw in FRUSTRATED_KEYWORDS if kw in text_lower]

    # ── Healthcare flags ──────────────────────────────────────────────────────
    medical_urgency        = any(kw in text_lower for kw in MEDICAL_URGENCY_KEYWORDS)
    insurance_frustration  = any(kw in text_lower for kw in INSURANCE_FRUSTRATION_KEYWORDS)
    health_anxiety         = medical_urgency and any(
        kw in text_lower for kw in ["worried", "scared", "anxious", "afraid", "nervous"]
    )
    service_appreciation   = len(positive_matches) > 0

    # ── Sentiment scoring ─────────────────────────────────────────────────────
    urgency_score    = min(len(urgent_matches) * 0.3, 1.0)
    frustration_score = min(len(frustrated_matches) * 0.25, 1.0)
    positive_score   = min(len(positive_matches) * 0.25, 1.0)

    if urgency_score > 0.3:
        sentiment       = "urgent"
        emotional_tone  = "distressed"
        confidence      = 0.7 + urgency_score * 0.2
        urgency_level   = "high"
    elif frustration_score > 0.3:
        sentiment       = "negative"
        emotional_tone  = "frustrated"
        confidence      = 0.65 + frustration_score * 0.2
        urgency_level   = "medium"
    elif positive_score > 0.3:
        sentiment       = "positive"
        emotional_tone  = "appreciative"
        confidence      = 0.70 + positive_score * 0.2
        urgency_level   = "low"
    else:
        sentiment       = "neutral"
        emotional_tone  = "neutral"
        confidence      = 0.6
        urgency_level   = "low"

    # Collect all matched indicators
    key_indicators = urgent_matches + frustrated_matches + positive_matches

    return {
        "sentiment":       sentiment,
        "confidence":      round(min(confidence, 0.99), 2),
        "emotional_tone":  emotional_tone,
        "urgency_level":   urgency_level,
        "key_indicators":  key_indicators[:10],   # top 10
        "healthcare_flags": {
            "medical_urgency":       medical_urgency,
            "insurance_frustration": insurance_frustration,
            "health_anxiety":        health_anxiety,
            "service_appreciation":  service_appreciation,
        },
        "context": context,
    }


def _analyze_email_sentiment_logic(
    subject: str,
    body: str,
    context: str = "email",
) -> Dict[str, Any]:
    """Combine subject + body for email-specific analysis."""
    combined = f"{subject} {body}".strip()
    result   = _analyze_sentiment_logic(combined, context)

    # Subject line zyada weight deta hai urgency ke liye
    subject_lower = subject.lower()
    if any(kw in subject_lower for kw in URGENT_KEYWORDS):
        result["urgency_level"] = "high"
        result["confidence"]    = min(result["confidence"] + 0.1, 0.99)

    result["subject_sentiment"] = _analyze_sentiment_logic(subject, "subject")["sentiment"]
    result["body_length"]       = len(body)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """MCP client ko available tools batao."""
    return [
        types.Tool(
            name="analyze_email_sentiment",
            description=(
                "Analyze sentiment of an email (subject + body). "
                "Returns sentiment, confidence, urgency level, and healthcare-specific flags."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content",
                    },
                    "context": {
                        "type": "string",
                        "description": "Context hint (default: 'email')",
                        "default": "email",
                    },
                },
                "required": ["subject", "body"],
            },
        ),
        types.Tool(
            name="analyze_text_sentiment",
            description=(
                "Analyze sentiment of any text. "
                "Returns sentiment, emotional tone, and urgency indicators."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze",
                    },
                    "context": {
                        "type": "string",
                        "description": "Context hint (default: 'general')",
                        "default": "general",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="batch_sentiment_analysis",
            description="Analyze sentiment of multiple text items at once.",
            inputSchema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":      {"type": "string"},
                                "text":    {"type": "string"},
                                "context": {"type": "string"},
                            },
                            "required": ["id", "text"],
                        },
                        "description": "List of items to analyze",
                    },
                },
                "required": ["items"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: Dict[str, Any],
) -> List[types.TextContent]:
    """Tool calls handle karo."""

    try:
        if name == "analyze_email_sentiment":
            subject = arguments.get("subject", "")
            body    = arguments.get("body", "")
            context = arguments.get("context", "email")
            result  = _analyze_email_sentiment_logic(subject, body, context)

        elif name == "analyze_text_sentiment":
            text    = arguments.get("text", "")
            context = arguments.get("context", "general")
            result  = _analyze_sentiment_logic(text, context)

        elif name == "batch_sentiment_analysis":
            items = arguments.get("items", [])
            batch_results = []
            for item in items:
                item_result = _analyze_sentiment_logic(
                    item.get("text", ""),
                    item.get("context", "general"),
                )
                item_result["id"] = item.get("id", "unknown")
                batch_results.append(item_result)
            result = {"results": batch_results, "total": len(batch_results)}

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [types.TextContent(type="text", text=json.dumps(result))]

    except Exception as exc:
        logger.error(f"Tool '{name}' failed: {exc}", exc_info=True)
        error_result = {
            "error":    str(exc),
            "tool":     name,
            "sentiment": "neutral",
            "confidence": 0.5,
            "emotional_tone": "neutral",
            "urgency_level": "low",
            "key_indicators": [],
            "healthcare_flags": {
                "medical_urgency":       False,
                "insurance_frustration": False,
                "health_anxiety":        False,
                "service_appreciation":  False,
            },
        }
        return [types.TextContent(type="text", text=json.dumps(error_result))]


# ══════════════════════════════════════════════════════════════════════════════
# MCP RESOURCE — Help documentation
# ══════════════════════════════════════════════════════════════════════════════

@server.list_resources()
async def list_resources() -> List[types.Resource]:
    return [
        types.Resource(
            uri="sentiment://help",
            name="Sentiment Analysis Help",
            description="How to use the sentiment analysis MCP server",
            mimeType="text/plain",
        ),
        types.Resource(
            uri="sentiment://healthcare-keywords",
            name="Healthcare Keywords",
            description="List of healthcare-specific keywords used for analysis",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    if str(uri) == "sentiment://help":
        return """
Sentiment Analysis MCP Server — Usage Guide
============================================

TOOLS:
  1. analyze_email_sentiment
     Input:  subject (str), body (str), context (str, optional)
     Output: sentiment, confidence, urgency_level, healthcare_flags

  2. analyze_text_sentiment
     Input:  text (str), context (str, optional)
     Output: sentiment, confidence, emotional_tone, key_indicators

  3. batch_sentiment_analysis
     Input:  items: [{id, text, context}]
     Output: {results: [...], total: N}

SENTIMENT VALUES:
  urgent, negative, neutral, positive

URGENCY LEVELS:
  low, medium, high

HEALTHCARE FLAGS:
  medical_urgency, insurance_frustration, health_anxiety, service_appreciation
"""
    elif str(uri) == "sentiment://healthcare-keywords":
        return json.dumps({
            "urgent_keywords":     URGENT_KEYWORDS,
            "frustrated_keywords": FRUSTRATED_KEYWORDS,
            "positive_keywords":   POSITIVE_KEYWORDS,
            "medical_urgency":     MEDICAL_URGENCY_KEYWORDS,
            "insurance_frustration": INSURANCE_FRUSTRATION_KEYWORDS,
        }, indent=2)
    else:
        raise ValueError(f"Unknown resource URI: {uri}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    logger.info("Starting MCP Sentiment Analysis Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
