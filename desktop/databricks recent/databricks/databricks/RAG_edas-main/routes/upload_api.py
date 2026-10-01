from fastapi import APIRouter, UploadFile

from services.rag_service import process_document

router = APIRouter()

@router.post("/documents/upload")

async def upload_document(

    userId: str,
    documentId: str,
    file: UploadFile

):

    result = await process_document(

        userId,
        documentId,
        file
    )

    return result