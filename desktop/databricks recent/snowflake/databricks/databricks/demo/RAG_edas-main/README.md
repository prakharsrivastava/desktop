# RAG EDAS System - Configuration and Setup

## Overview
RAG EDAS is a FastAPI-based Retrieval-Augmented Generation (RAG) system with emotion and intent detection.

## System Architecture

### Core Components:
1. **FastAPI App** (app.py) - Main application entry point
2. **Routes**:
   - `/documents/upload` - Upload and process PDF documents
   - `/query` - Query the RAG system with intent/emotion detection
3. **Services**:
   - `rag_service.py` - PDF processing and text embedding
   - `vector_service.py` - Vector similarity search
   - `llm_service.py` - LLM response generation
   - `intent_service.py` - Intent detection (zero-shot classification)
   - `emotion_service.py` - Emotion detection
   - `memory_service.py` - Session management with Redis
   - `mlflow_service.py` - Metrics logging

## Setup Instructions

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Start Redis Server
```bash
redis-server
```

### 4. Run the Application
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Upload Document
```bash
POST /documents/upload
Parameters:
  - userId: str
  - documentId: str
  - file: UploadFile (PDF)

Response:
{
  "status": "success",
  "chunksStored": 15,
  "documentId": "doc-123"
}
```

### Query the System
```bash
POST /query
Body:
{
  "userId": "user-123",
  "sessionId": "session-456",
  "query": "What is the loan status?"
}

Response:
{
  "intent": {
    "intent": "Question",
    "confidence": 0.95
  },
  "emotion": {
    "label": "neutral",
    "score": 0.92
  },
  "answer": "Based on the documents provided..."
}
```

## Key Fixes Applied

1. ✅ **query_api.py** - Fixed import organization, added actual latency calculation
2. ✅ **llm_service.py** - Added API key handling and error handling
3. ✅ **memory_service.py** - Added type hints, error handling, and Redis connection verification
4. ✅ **intent_service.py** - Added error handling and consistent return format
5. ✅ **emotion_service.py** - Added error handling and standardized output
6. ✅ **rag_service.py** - Added error handling, logging, and empty document checks
7. ✅ **vector_service.py** - Added error handling and better empty result handling
8. ✅ **upload_api.py** - Added error handling and type hints
9. ✅ **mlflow_service.py** - Added type hints, error handling, and logging
10. ✅ **prompt_template.py** - Created missing prompt template module

## Testing

Use the Swagger UI at `http://localhost:8000/docs` to test the endpoints.
