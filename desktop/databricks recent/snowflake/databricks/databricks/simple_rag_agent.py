
import mlflow
from mlflow.pyfunc import PythonModel
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

class SimpleRAGAgent(PythonModel):
    def predict(self, context, model_input, params=None):
        # Handle both dict and DataFrame inputs
        if hasattr(model_input, "to_dict"):
            model_input = model_input.to_dict(orient="records")[0]
        
        # Get user question
        messages = model_input.get("input", model_input.get("messages", []))
        if isinstance(messages, str):
            user_message = messages
        else:
            user_message = messages[-1]["content"] if messages else ""
        
        # Step 1: Search vector index
        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name="rikku_vs_endpoint",
            index_name="agents.default.rikku_knowledge_base1"
        )
        
        results = index.similarity_search(
            query_text=user_message,
            columns=["content", "source"],
            num_results=3
        )
        
        # Build context from results
        docs = results.get("result", {}).get("data_array", [])
        if docs:
            context_text = "\n\n".join([f"[{doc[1]}]: {doc[0]}" for doc in docs])
        else:
            context_text = "No relevant information found."
        
        # Step 2: Generate answer with LLM
        prompt = f"""Answer based only on the context below. If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question: {user_message}

Answer:"""
        
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name="databricks-meta-llama-3.1-405b-instruct",
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
            temperature=0.1,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return answer
