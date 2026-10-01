from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import chromadb

from PyPDF2 import PdfReader

embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)

client = chromadb.Client()

collection = client.get_or_create_collection(

    "documents"

)

async def process_document(

    userId,
    documentId,
    file

):


    pdf = PdfReader(file.file)

    text = ""

    for page in pdf.pages:

        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50

    )

    chunks = splitter.split_text(text)


    embeddings = embedding_model.encode(chunks)


    for idx, chunk in enumerate(chunks):

        collection.add(

            documents=[chunk],

            embeddings=[embeddings[idx].tolist()],

            ids=[f"{documentId}_{idx}"],

            metadatas=[{

                "userId": userId,

                "documentId": documentId

            }]
        )

    return {

        "status": "success",

        "chunksStored": len(chunks)

    }