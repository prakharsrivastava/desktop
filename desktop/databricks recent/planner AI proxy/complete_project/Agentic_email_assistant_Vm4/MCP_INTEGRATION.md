# MCP Integration Guide

## Overview

This project uses the **Model Context Protocol (MCP)** to integrate a dedicated
sentiment analysis server into the email intelligence pipeline.

The MCP server (`mcp_sentiment_server.py`) runs as a separate process and communicates
with the main application via **stdio transport**.

---

## Architecture

```
FastAPI App (main.py)
      ↓
email_intelligence.py
      ↓  (optional sentiment enrichment)
mcp_helper.py  ←→  MCPSentimentClient (mcp_client.py)
                         ↓ stdio
               mcp_sentiment_server.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install mcp
# or with uv:
uv add mcp
```

### 2. Configure MCP server path

Edit `mcp_config.json`:

```json
{
  "mcpServers": {
    "sentiment-analysis": {
      "command": "python",
      "args": ["mcp_sentiment_server.py"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

### 3. Enable sentiment integration

In `config.json`:

```json
{
  "enable_sentiment_integration": true
}
```

Or via environment variable:

```bash
export ENABLE_SENTIMENT_INTEGRATION=true
```

---

## MCP Tools Available

### `analyze_email_sentiment`

Analyzes sentiment of a full email (subject + body).

**Input:**
```json
{
  "subject": "string",
  "body": "string",
  "context": "email"
}
```

**Output:**
```json
{
  "sentiment": "urgent | negative | neutral | positive",
  "confidence": 0.0-1.0,
  "emotional_tone": "distressed | frustrated | neutral | appreciative",
  "urgency_level": "low | medium | high",
  "key_indicators": ["keyword1", "keyword2"],
  "healthcare_flags": {
    "medical_urgency": false,
    "insurance_frustration": false,
    "health_anxiety": false,
    "service_appreciation": false
  }
}
```

### `analyze_text_sentiment`

Analyzes sentiment of any text string.

### `batch_sentiment_analysis`

Analyzes multiple items in one call.

---

## Usage in Code

```python
from app.services.mcp_helper import analyze_email_sentiment_sync

result = analyze_email_sentiment_sync(
    subject="URGENT - Need ID card",
    body="Please send my replacement card ASAP",
    context="email",
    fallback_on_error=True,
)

print(result["sentiment"])       # "urgent"
print(result["urgency_level"])   # "high"
print(result["healthcare_flags"]["medical_urgency"])  # False
```

---

## Running the MCP Server Standalone

```bash
python mcp_sentiment_server.py
```

The server reads from stdin and writes to stdout (MCP stdio transport).

---

## Resources

- MCP help: `sentiment://help`
- Healthcare keywords: `sentiment://healthcare-keywords`

---

## Troubleshooting

**Server not starting:**
- Check Python path in `mcp_config.json`
- Ensure `mcp` package is installed: `pip install mcp`

**Sentiment always returning neutral:**
- Check `enable_sentiment_integration` is `true` in config
- Check MCP server process is running

**Timeout errors:**
- Default timeout is 10s per call
- Increase via `MCP_TIMEOUT` environment variable
