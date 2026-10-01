def build_prompt(

    query,
    intent,
    emotion,
    chunks

):

    context = "\n".join(chunks)

    return f"""

You are a customer support assistant.

User Intent:
{intent}

User Emotion:
{emotion}

Document Context:
{context}

User Query:
{query}

Rules:
- Answer only from context
- Be empathetic if emotion is angry/frustrated
- If answer unavailable, say:
  "Connecting to support agent"

Generate response:
"""