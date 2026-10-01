# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,MongoDB Conversation Memory Saver
# MAGIC %md
# MAGIC # MongoDB Conversation Memory Saver
# MAGIC
# MAGIC Stores all RAG conversations in MongoDB keyed by `thread_id` + `user_id` so that
# MAGIC each user has isolated memory. Conversations persist across sessions and can be
# MAGIC retrieved by thread, user, or time range.
# MAGIC
# MAGIC **Architecture:**
# MAGIC - MongoDB collection `conversations` stores message documents
# MAGIC - Composite key: `{user_id, thread_id}` ensures per-user isolation
# MAGIC - Each message includes role, content, timestamp, and optional RAG metadata
# MAGIC - Integrates with the existing Databricks Vector Search index for retrieval
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - MongoDB instance (Atlas or self-hosted) accessible from Databricks
# MAGIC - Connection string stored in Databricks Secrets (scope: `mongodb`, key: `connection_string`)
# MAGIC - Vector Search index `workspace.default.rag_chunks_index` from the RAG pipeline notebook

# COMMAND ----------

# DBTITLE 1,Install & Configure
# ── Install pymongo and set up configuration ──────────────────────
%pip install pymongo -q

# ── Configuration ───────────────────────────────────────────────────
import os

# MongoDB connection string — store in Databricks Secrets
# To set up: dbutils.secrets.put(scope="mongodb", key="connection_string")
# Or replace with your Atlas connection string directly for testing
MONGO_CONNECTION_STRING = dbutils.secrets.get(scope="mongodb", key="connection_string") \
    if dbutils.secrets.listScopes() and any(s.name == "mongodb" for s in dbutils.secrets.listScopes()) \
    else os.environ.get("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")

MONGO_DB_NAME = "rag_conversations"
MONGO_COLLECTION_NAME = "conversations"

# Vector Search index (from the RAG pipeline notebook)
VS_INDEX_NAME = "workspace.default.rag_chunks_index"
VS_ENDPOINT_NAME = "rag-chunking-endpoint"
EMBEDDING_MODEL = "databricks-gte-large-en"
LLM_MODEL = "databricks-claude-sonnet-4"

print(f"MongoDB DB : {MONGO_DB_NAME}")
print(f"Collection : {MONGO_COLLECTION_NAME}")
print(f"VS Index   : {VS_INDEX_NAME}")

# COMMAND ----------

# DBTITLE 1,MongoDBConversationSaver Class
# ── MongoDBConversationSaver: per-user, per-thread memory ─────────────
# Each conversation is isolated by (user_id, thread_id).
# Different users → different memories. Same user, different thread → different context.

from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId
import json


class MongoDBConversationSaver:
    """Stores and retrieves conversation messages from MongoDB.

    Each document in the collection represents one message in a conversation,
    keyed by user_id + thread_id for full isolation between users.
    """

    def __init__(self, connection_string, db_name, collection_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Ensure indexes for fast lookups
        self.collection.create_index([("user_id", 1), ("thread_id", 1), ("timestamp", 1)])
        self.collection.create_index([("user_id", 1), ("timestamp", -1)])
        self.collection.create_index([("thread_id", 1), ("timestamp", -1)])

    def add_message(self, user_id, thread_id, role, content, metadata=None):
        """Add a single message to the conversation.

        Args:
            user_id:    Unique user identifier (e.g. 'user_001')
            thread_id:  Conversation thread ID (e.g. 'thread_abc123')
            role:       'user', 'assistant', or 'system'
            content:    Message text
            metadata:   Optional dict with RAG context, scores, tokens, etc.
        """
        doc = {
            "user_id": user_id,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc),
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def add_user_message(self, user_id, thread_id, content, metadata=None):
        return self.add_message(user_id, thread_id, "user", content, metadata)

    def add_assistant_message(self, user_id, thread_id, content, metadata=None):
        return self.add_message(user_id, thread_id, "assistant", content, metadata)

    def get_history(self, user_id, thread_id, limit=None):
        """Retrieve conversation history for a specific user + thread.

        Returns messages in chronological order.
        """
        query = {"user_id": user_id, "thread_id": thread_id}
        cursor = self.collection.find(query).sort("timestamp", 1)
        if limit:
            cursor = cursor.limit(limit)
        messages = []
        for doc in cursor:
            messages.append({
                "role": doc["role"],
                "content": doc["content"],
                "timestamp": doc["timestamp"],
                "metadata": doc.get("metadata", {}),
            })
        return messages

    def get_history_as_strings(self, user_id, thread_id, limit=None):
        """Return conversation history as formatted strings for LLM context."""
        messages = self.get_history(user_id, thread_id, limit)
        formatted = []
        for msg in messages:
            prefix = msg["role"].upper()
            formatted.append(f"{prefix}: {msg['content']}")
        return formatted

    def get_history_as_chat_messages(self, user_id, thread_id, limit=None):
        """Return history in the format expected by LLM chat APIs: [{role, content}]."""
        messages = self.get_history(user_id, thread_id, limit)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def clear_thread(self, user_id, thread_id):
        """Delete all messages for a specific user + thread."""
        result = self.collection.delete_many({"user_id": user_id, "thread_id": thread_id})
        return result.deleted_count

    def list_threads(self, user_id):
        """List all thread IDs for a user, with message counts and last activity."""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$thread_id",
                "message_count": {"$sum": 1},
                "last_message": {"$max": "$timestamp"},
            }},
            {"$sort": {"last_message": -1}},
        ]
        return list(self.collection.aggregate(pipeline))

    def list_all_users(self):
        """List all unique user IDs in the collection."""
        return self.collection.distinct("user_id")

    def close(self):
        self.client.close()


# Initialize the saver
saver = MongoDBConversationSaver(
    MONGO_CONNECTION_STRING, MONGO_DB_NAME, MONGO_COLLECTION_NAME
)
print(f"✅ MongoDBConversationSaver initialized")
print(f"   Database: {MONGO_DB_NAME}")
print(f"   Collection: {MONGO_COLLECTION_NAME}")

# COMMAND ----------

# DBTITLE 1,RAG Chat with MongoDB Memory
# ── RAG Chat with MongoDB Memory + Vector Search retrieval ────────────
# This cell connects the MongoDB conversation memory with the Vector Search
# index from the RAG pipeline notebook for retrieval-augmented generation.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()


def rag_chat(saver, user_id, thread_id, user_query, num_results=5, max_history=10):
    """Full RAG pipeline: retrieve context + chat with LLM + store in MongoDB.

    1. Retrieve relevant chunks from Vector Search index
    2. Load conversation history from MongoDB (per user_id + thread_id)
    3. Build prompt with RAG context + conversation history
    4. Call LLM for response
    5. Store both user query and assistant response in MongoDB

    Args:
        saver:      MongoDBConversationSaver instance
        user_id:    User identifier for memory isolation
        thread_id:  Thread identifier for conversation isolation
        user_query: User's question
        num_results: Number of chunks to retrieve from Vector Search
        max_history: Max prior messages to include as context

    Returns:
        dict with 'response', 'retrieved_chunks', 'used_history_count'
    """
    # ── Step 1: Retrieve relevant chunks from Vector Search ──
    vs_results = w.vector_search_indexes.query_index(
        index_name=VS_INDEX_NAME,
        columns=["chunk_id", "doc_id", "page", "section_id", "chunk_type", "chunk_text"],
        query_text=user_query,
        num_results=num_results,
    )

    rag_context = ""
    retrieved_chunks = []
    if vs_results.result and vs_results.result.data_array:
        for row in vs_results.result.data_array:
            score = row[-1]
            chunk_text = row[5] if len(row) > 5 and row[5] else ""
            doc_id = row[1] if len(row) > 1 else "?"
            chunk_type = row[4] if len(row) > 4 else "?"
            retrieved_chunks.append({
                "doc_id": doc_id,
                "chunk_type": chunk_type,
                "score": float(score),
                "preview": chunk_text[:200],
            })
        rag_context = "\n\n".join(
            f"[Source: {c['doc_id']} | Type: {c['chunk_type']} | Score: {c['score']:.3f}]\n{c['preview']}"
            for c in retrieved_chunks
        )

    # ── Step 2: Load conversation history from MongoDB ──
    history_msgs = saver.get_history_as_chat_messages(user_id, thread_id, limit=max_history)
    used_history_count = len(history_msgs)

    # ── Step 3: Build messages for LLM ──
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer questions. "
        "If the context doesn't contain the answer, say you don't know.\n\n"
        f"--- Retrieved Context ---\n{rag_context}\n--- End Context ---"
    )

    chat_messages = [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
    ]
    # Add conversation history
    for msg in history_msgs:
        role = ChatMessageRole.USER if msg["role"] == "user" else ChatMessageRole.ASSISTANT
        chat_messages.append(ChatMessage(role=role, content=msg["content"]))
    # Add current user query
    chat_messages.append(ChatMessage(role=ChatMessageRole.USER, content=user_query))

    # ── Step 4: Call LLM ──
    response = w.serving_endpoints.query(
        name=LLM_MODEL,
        messages=chat_messages,
    )
    assistant_response = response.choices[0].message.content

    # ── Step 5: Store both messages in MongoDB ──
    saver.add_user_message(user_id, thread_id, user_query, metadata={
        "retrieved_chunk_count": len(retrieved_chunks),
    })
    saver.add_assistant_message(user_id, thread_id, assistant_response, metadata={
        "retrieved_chunks": retrieved_chunks,
        "llm_model": LLM_MODEL,
    })

    return {
        "response": assistant_response,
        "retrieved_chunks": retrieved_chunks,
        "used_history_count": used_history_count,
    }


print("✅ rag_chat function ready")
print(f"   Vector Search index : {VS_INDEX_NAME}")
print(f"   LLM model           : {LLM_MODEL}")
print(f"   History source      : MongoDB ({MONGO_DB_NAME}.{MONGO_COLLECTION_NAME})")

# COMMAND ----------

# DBTITLE 1,Test: Per-User Memory Isolation
# ── Test 1: Verify per-user memory isolation ─────────────────────────
# Two different users chat about different topics.
# Then we verify that each user only sees their own conversation history.

# Clear any existing test data
saver.clear_thread("user_alice", "thread_001")
saver.clear_thread("user_bob", "thread_001")

# ── Alice asks about document sections ──
print("=" * 70)
print("Alice's conversation (thread_001)")
print("=" * 70)

alice_result = rag_chat(
    saver, user_id="user_alice", thread_id="thread_001",
    user_query="What sections are in the dummy document?",
)
print(f"\nAlice's response (preview):\n{alice_result['response'][:300]}...")
print(f"  Retrieved chunks: {len(alice_result['retrieved_chunks'])}")
print(f"  History used: {alice_result['used_history_count']} messages")

# ── Bob asks about customer orders ──
print("\n" + "=" * 70)
print("Bob's conversation (thread_001)")
print("=" * 70)

bob_result = rag_chat(
    saver, user_id="user_bob", thread_id="thread_001",
    user_query="Show me customer order data from the table",
)
print(f"\nBob's response (preview):\n{bob_result['response'][:300]}...")
print(f"  Retrieved chunks: {len(bob_result['retrieved_chunks'])}")
print(f"  History used: {bob_result['used_history_count']} messages")

# ── Verify isolation: Alice's history should NOT contain Bob's messages ──
print("\n" + "=" * 70)
print("Memory Isolation Check")
print("=" * 70)

alice_history = saver.get_history("user_alice", "thread_001")
bob_history = saver.get_history("user_bob", "thread_001")

print(f"\nAlice's messages in thread_001: {len(alice_history)}")
for msg in alice_history:
    print(f"  [{msg['role']}] {msg['content'][:80]}...")

print(f"\nBob's messages in thread_001: {len(bob_history)}")
for msg in bob_history:
    print(f"  [{msg['role']}] {msg['content'][:80]}...")

# Assert isolation
alice_contents = [m["content"] for m in alice_history]
bob_contents = [m["content"] for m in bob_history]
assert not any("customer order" in c.lower() for c in alice_contents), \
    "ALICE sees BOB's messages! Isolation broken!"
assert not any("sections are in" in c.lower() for c in bob_contents), \
    "BOB sees ALICE's messages! Isolation broken!"

print("\n✅ Memory isolation verified: Alice and Bob have separate memories!")

# COMMAND ----------

# DBTITLE 1,Test: Multi-Turn Memory
# ── Test 2: Multi-turn conversation with memory ──────────────────────
# Same user, same thread — follow-up questions use prior context.
# Different thread for same user — no prior context (fresh start).

# ── Alice continues in thread_001 (should have memory) ──
print("=" * 70)
print("Alice's follow-up in thread_001 (has memory)")
print("=" * 70)

followup = rag_chat(
    saver, user_id="user_alice", thread_id="thread_001",
    user_query="Can you tell me more about the table section?",
)
print(f"\nResponse (preview):\n{followup['response'][:300]}...")
print(f"  History used: {followup['used_history_count']} messages (should be > 0)")

# ── Alice starts a NEW thread_002 (no memory) ──
print("\n" + "=" * 70)
print("Alice in NEW thread_002 (fresh, no memory)")
print("=" * 70)

fresh = rag_chat(
    saver, user_id="user_alice", thread_id="thread_002",
    user_query="What were we discussing earlier?",
)
print(f"\nResponse (preview):\n{fresh['response'][:300]}...")
print(f"  History used: {fresh['used_history_count']} messages (should be 0)")

# ── Verify: thread_001 has 4 messages, thread_002 has 2 messages ──
thread1_msgs = saver.get_history("user_alice", "thread_001")
thread2_msgs = saver.get_history("user_alice", "thread_002")

print(f"\n--- Alice's threads ---")
print(f"thread_001: {len(thread1_msgs)} messages")
print(f"thread_002: {len(thread2_msgs)} messages")

assert len(thread1_msgs) == 4, f"Expected 4 messages in thread_001, got {len(thread1_msgs)}"
assert len(thread2_msgs) == 2, f"Expected 2 messages in thread_002, got {len(thread2_msgs)}"

print("\n✅ Multi-turn memory verified: thread_001 has context, thread_002 is fresh!")

# ── Show all threads for Alice ──
print("\n--- Alice's all threads ---")
for t in saver.list_threads("user_alice"):
    print(f"  Thread: {t['_id']} | Messages: {t['message_count']} | Last: {t['last_message']}")

# COMMAND ----------

# DBTITLE 1,Utility Functions & Stats
# ── Utility functions for conversation management ────────────────────

def show_conversation(saver, user_id, thread_id, max_msgs=20):
    """Pretty-print a full conversation."""
    history = saver.get_history(user_id, thread_id, limit=max_msgs)
    print(f"\n{'='*70}")
    print(f"Conversation: user={user_id}, thread={thread_id}")
    print(f"Messages: {len(history)}")
    print(f"{'='*70}")
    for i, msg in enumerate(history):
        ts = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if msg["timestamp"] else "?"
        print(f"\n[{i+1}] {msg['role'].upper()} ({ts})")
        print(f"    {msg['content'][:200]}")
        if msg.get("metadata"):
            meta = msg["metadata"]
            if "retrieved_chunks" in meta:
                print(f"    [RAG: {len(meta['retrieved_chunks'])} chunks retrieved]")
            if "retrieved_chunk_count" in meta:
                print(f"    [RAG: {meta['retrieved_chunk_count']} chunks retrieved]")
    return history


def get_stats(saver):
    """Show overall statistics."""
    total = saver.collection.count_documents({})
    users = saver.list_all_users()
    print(f"\n{'='*70}")
    print(f"MongoDB Conversation Stats")
    print(f"{'='*70}")
    print(f"  Total messages : {total}")
    print(f"  Total users    : {len(users)}")
    print(f"  Users          : {users}")
    for user_id in users:
        threads = saver.list_threads(user_id)
        print(f"  User '{user_id}': {len(threads)} thread(s)")
        for t in threads:
            print(f"    - {t['_id']}: {t['message_count']} msgs, last: {t['last_message']}")


def delete_conversation(saver, user_id, thread_id):
    """Delete an entire conversation."""
    count = saver.clear_thread(user_id, thread_id)
    print(f"Deleted {count} messages from user={user_id}, thread={thread_id}")
    return count


# ── Demo: show all conversations ──
print("✅ Utility functions ready: show_conversation(), get_stats(), delete_conversation()")
print("\n--- Current conversation stats ---")
get_stats(saver)

print("\n--- Alice's thread_001 conversation ---")
show_conversation(saver, "user_alice", "thread_001")

print("\n--- Bob's thread_001 conversation ---")
show_conversation(saver, "user_bob", "thread_001")