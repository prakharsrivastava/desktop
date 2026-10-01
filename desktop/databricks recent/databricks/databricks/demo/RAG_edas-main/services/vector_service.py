from services.rag_service import collection

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)

def retrieve_chunks(

    query,
    top_k=3

):

    query_embedding = embedding_model.encode(

        query

    ).tolist()

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=top_k

    )

    return results["documents"][0]