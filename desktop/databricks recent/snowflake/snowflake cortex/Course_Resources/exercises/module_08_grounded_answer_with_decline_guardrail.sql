-- ==========================================================
-- Exercise 8.2 — Ground the Answer and Decline Outside the Corpus
-- Course: 1018
-- Module 08
-- ==========================================================
--
-- A RAG assistant that confidently guesses when its corpus doesn't cover
-- the question is worse than no assistant — it erodes trust the first
-- time a rep acts on a wrong answer. The module's guardrail pattern wraps
-- AI_COMPLETE with an explicit instruction to answer ONLY from the
-- retrieved context, citing the source, and to decline with an exact
-- string when the context is insufficient. Apply that same guardrail
-- prompt shape to a planogram question instead of the module's product-
-- spec question.
--
-- Your task:
-- 1. Write an AI_COMPLETE call over a `context_text` built with LISTAGG
--    from retrieved_chunks (concatenating chunk_text and a shelf
--    citation), answering a planogram placement question ONLY from that
--    context.
-- 2. Extend the prompt so it declines with the exact string
--    "This is not covered in the indexed planogram corpus." when the
--    context doesn't have enough information — do not let it guess.
-- ==========================================================

-- YOUR CODE:

SELECT AI_COMPLETE(
  '___',
  CONCAT(
    'Answer using ONLY the context below. Cite the source shelf position.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: where should the granola bar SKU be placed in a small-format store planogram?'
  )
) AS grounded_answer
FROM (
  SELECT LISTAGG(chunk_text || ___, '\n---\n') AS context_text
  FROM retrieved_chunks
);

SELECT AI_COMPLETE(
  'claude-sonnet-4-5',
  CONCAT(
    'You are a CPG planogram assistant. Answer ONLY using the context below. ',
    'If the context does not contain enough information to answer confidently, ',
    'respond exactly with: "___" ',
    'Do not guess or use outside knowledge.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: ', :rep_question
  )
) AS grounded_or_decline;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT AI_COMPLETE(
  'claude-sonnet-4-5',
  CONCAT(
    'Answer using ONLY the context below. Cite the source shelf position.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: where should the granola bar SKU be placed in a small-format store planogram?'
  )
) AS grounded_answer
FROM (
  SELECT LISTAGG(chunk_text || ' [shelf ' || store_format || ']', '\n---\n') AS context_text
  FROM retrieved_chunks
);

SELECT AI_COMPLETE(
  'claude-sonnet-4-5',
  CONCAT(
    'You are a CPG planogram assistant. Answer ONLY using the context below. ',
    'If the context does not contain enough information to answer confidently, ',
    'respond exactly with: "This is not covered in the indexed planogram corpus." ',
    'Do not guess or use outside knowledge.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: ', :rep_question
  )
) AS grounded_or_decline;
*/
