# ============================================================
# Course 1018 — Module 08
# On-slide PY reference  (1 snippets)
# Authentic code as shown on the course slides. Copy, run, experiment.
# ============================================================


# ----------------------------------------------------------
# Retrieving Top-k Passages for a Live Question
# ----------------------------------------------------------

# Production path: call the Cortex Search REST/Python API via snowflake.core
from snowflake.core import Root

root = Root(session)
search_service = (
    root.databases["cpg_db"].schemas["public"]
    .cortex_search_services["cpg_product_spec_search"]
)
response = search_service.search(
    query="does the granola bar SKU contain tree nuts",
    columns=["chunk_id", "chunk_text", "sku", "doc_type", "page_number", "effective_date"],
    filter={"@eq": {"doc_type": "product_spec"}},
    limit=3,
)
retrieved_chunks = response.results

