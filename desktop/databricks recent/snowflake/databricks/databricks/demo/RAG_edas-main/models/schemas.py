from pydantic import BaseModel

# -----------------------------------------
# DOCUMENT UPLOAD REQUEST
# -----------------------------------------

class UploadRequest(BaseModel):

    userId: str

    documentId: str

# -----------------------------------------
# QUERY REQUEST
# -----------------------------------------

class QueryRequest(BaseModel):

    userId: str

    sessionId: str

    query: str

# -----------------------------------------
# QUERY RESPONSE
# -----------------------------------------

class QueryResponse(BaseModel):

    intent: str

    emotion: str

    answer: str

# -----------------------------------------
# SESSION MODEL
# -----------------------------------------

class SessionState(BaseModel):

    session_id: str

    intent: str = None

    loan_id: str = None

    complaint_id: str = None