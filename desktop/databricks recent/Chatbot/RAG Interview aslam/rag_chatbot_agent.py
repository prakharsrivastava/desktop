import uuid
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse


class RAGChatbotAgent(ChatAgent):
    """RAG chatbot agent: retrieves chunks via Vector Search, generates answers with Foundation Model API."""

    def load_context(self, context):
        from databricks.sdk import WorkspaceClient
        import mlflow.deployments

        self.w = WorkspaceClient()
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

        self.vs_index = "workspace.default.prak_pdf_chunks_index"
        self.llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
        self.top_k = 3
        self.content_filter = '{"chunk_type": ["text", "table", "table_subchunk", "caption"]}'

    def predict(self, messages, context=None, custom_inputs=None):
        """Input: list of ChatAgentMessage. Output: ChatAgentResponse with assistant reply."""
        # Extract the latest user question from the message history
        user_question = ""
        for msg in reversed(messages):
            if msg.role == "user" and msg.content:
                user_question = msg.content
                break

        if not user_question:
            return ChatAgentResponse(
                messages=[ChatAgentMessage(role="assistant", content="Please ask a question about the document.", id=str(uuid.uuid4()))]
            )

        # Retrieve chunks from Vector Search
        results = self.w.vector_search_indexes.query_index(
            index_name=self.vs_index,
            columns=["chunk_id", "chunk_text", "chunk_type", "page", "section_id", "file_name"],
            query_text=user_question,
            num_results=self.top_k,
            filters_json=self.content_filter,
        )
        chunks = results.result.data_array

        if not chunks:
            return ChatAgentResponse(
                messages=[ChatAgentMessage(role="assistant", content="I don't have enough information to answer this question.", id=str(uuid.uuid4()))]
            )

        # Build context from retrieved chunks
        context_parts = []
        for i, row in enumerate(chunks):
            context_parts.append(f"[Source {i+1}] (page {row[3]}, section {row[4]})\n{row[1]}")
        context_str = "\n\n".join(context_parts)

        # Generate answer via Foundation Model API
        system_prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context from PDF documents.

Context:
{context_str}

Instructions:
- Answer the question using ONLY the context above.
- If the context does not contain enough information, say: "I don't have enough information to answer this question."
- Cite sources by referring to page numbers and section IDs.
- Be concise and accurate."""

        response = self.deploy_client.predict(
            endpoint=self.llm_endpoint,
            inputs={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question},
                ],
                "max_tokens": 512,
                "temperature": 0.1,
            },
        )
        answer = response["choices"][0]["message"]["content"]

        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=answer, id=str(uuid.uuid4()))]
        )


import mlflow.models
mlflow.models.set_model(RAGChatbotAgent())