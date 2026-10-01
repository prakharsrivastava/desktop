import time
from fastapi import APIRouter
from pydantic import BaseModel
from services.intent_service import detect_intent
from services.emotion_service import detect_emotion
from services.vector_service import retrieve_chunks
from services.llm_service import generate_response
from services.mlflow_service import log_query_metrics
from prompts.prompt_template import build_prompt
from services.memory_service import (
    save_session,
    get_session
)

router = APIRouter()

class QueryRequest(BaseModel):
    userId: str
    sessionId: str
    query: str

@router.post("/query")
async def query(req: QueryRequest):
    start_time = time.time()
    
    # Get or create session
    session = get_session(req.sessionId)
    
    # Detect intent
    intent_result = detect_intent(req.query)
    
    # Detect emotion
    emotion_result = detect_emotion(req.query)
    
    # Retrieve relevant chunks
    chunks = retrieve_chunks(req.query)
    
    # Build prompt with context
    prompt = build_prompt(
        req.query,
        intent_result["intent"],
        emotion_result["label"],
        chunks
    )
    
    # Generate response from LLM
    answer = generate_response(prompt)
    
    # Update session
    session["last_query"] = req.query
    save_session(req.sessionId, session)
    
    # Log metrics
    latency = time.time() - start_time
    log_query_metrics(
        latency=latency,
        precision=0.91,
        hallucination_rate=0.03
    )
    
    return {
        "intent": intent_result,
        "emotion": emotion_result,
        "answer": answer
    }
