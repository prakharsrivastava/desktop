# Installation and Running Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-api-key-here
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Step 3: Start Redis Server

Open a new terminal and start Redis:

```bash
# On Linux/Mac
redis-server

# On Windows (if installed)
redis-server.exe

# Or use Docker
docker run -d -p 6379:6379 redis:latest
```

## Step 4: Run the Application

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start at: **http://localhost:8000**

## Step 5: Test the API

### Open Swagger UI:
Visit **http://localhost:8000/docs** to test endpoints interactively

### Upload a Document:
```bash
curl -X POST "http://localhost:8000/documents/upload?userId=user1&documentId=doc1" \
  -H "accept: application/json" \
  -F "file=@sample.pdf"
```

### Query the System:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user1",
    "sessionId": "session1",
    "query": "What is the main topic?"
  }'
```

## Troubleshooting

### Redis Connection Error
- Ensure Redis is running on `localhost:6379`
- Check with: `redis-cli ping` (should return "PONG")

### OpenAI API Error
- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check your API key is valid on https://platform.openai.com/account/api-keys

### Model Loading Error
- First run downloads the models (~2GB)
- Ensure good internet connection
- Check disk space for model cache

### Port Already in Use
- Change port in command: `uvicorn app:app --reload --port 8001`

## Next Steps

1. Upload your PDF documents using `/documents/upload`
2. Query the system using `/query`
3. Monitor metrics in MLflow
4. Check logs for debugging

Good to go! 🚀
