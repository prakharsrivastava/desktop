# Databricks notebook source
# DBTITLE 1,Title & Overview
# MAGIC %md
# MAGIC # RAG Glossary & Taxonomy Explorer — Effort Estimation
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC The RAG Glossary & Taxonomy Explorer system consists of:
# MAGIC
# MAGIC - **Glossary**: A collection of acronyms and terms
# MAGIC - **Taxonomy Explorer**: Per-module training guides (8–10 pages each)
# MAGIC - **RAG with Hybrid Search**: Keyword + semantic search on AWS vector DB
# MAGIC
# MAGIC Two approaches are estimated:
# MAGIC
# MAGIC 1. **Leverage existing infrastructure** — reuse the current AWS vector DB, embedding pipeline, and search stack
# MAGIC 2. **Build new separate DB from scratch** — provision a dedicated vector DB, ingestion pipeline, and search layer

# COMMAND ----------

# DBTITLE 1,Shared Assumptions
# MAGIC %md
# MAGIC ## Shared Assumptions
# MAGIC
# MAGIC | # | Assumption |
# MAGIC |---|------------|
# MAGIC | 1 | 1 developer full-time |
# MAGIC | 2 | Documents are text-based (PDF/Word/HTML), 8–10 pages per module guide |
# MAGIC | 3 | AWS infrastructure (OpenSearch or similar) available |
# MAGIC | 4 | Embedding model accessible (Bedrock/OpenAI) |
# MAGIC | 5 | Python primary language |
# MAGIC | 6 | No UI changes needed |
# MAGIC | 7 | Testing includes unit + integration tests |
# MAGIC | 8 | Code review and documentation included |
# MAGIC | 9 | Hybrid search (keyword + semantic) with configurable ratio required |

# COMMAND ----------

# DBTITLE 1,Approach 1 — Leverage Existing Infrastructure
# MAGIC %md
# MAGIC ## Approach 1 — Leverage Existing Infrastructure
# MAGIC
# MAGIC Reuses the existing AWS vector DB, embedding pipeline, and search infrastructure.
# MAGIC
# MAGIC | Phase | Description | Hours |
# MAGIC |-------|-------------|------|
# MAGIC | Phase 1: Document Analysis & Parsing | Analyze guide structure, implement parser, handle tables/headings | 4–6 hrs |
# MAGIC | Phase 2: Chunking Strategy | Define chunking, implement with metadata tagging, handle overlap | 3–4 hrs |
# MAGIC | Phase 3: Embedding & Storage | Generate embeddings, store in existing AWS vector DB, add metadata filters, verify indexing | 4–6 hrs |
# MAGIC | Phase 4: Search Integration & Testing | Test keyword+semantic search with new data, tune hybrid ratio, test sample queries, verify accuracy/latency | 6–8 hrs |
# MAGIC | Phase 5: Documentation & Handoff | Document ingestion pipeline, chunking strategy, knowledge transfer | 2–3 hrs |
# MAGIC | **Total** | | **19–27 hours (3–4 working days)** |

# COMMAND ----------

# DBTITLE 1,Approach 2 — Build New Separate DB from Scratch
# MAGIC %md
# MAGIC ## Approach 2 — Build New Separate DB from Scratch
# MAGIC
# MAGIC Provisions a dedicated vector DB with full ingestion pipeline and search layer.
# MAGIC
# MAGIC | Phase | Description | Hours |
# MAGIC |-------|-------------|------|
# MAGIC | Phase 1: Schema Design & Architecture | Design vector DB schema, keyword index, data model, architecture doc | 8–12 hrs |
# MAGIC | Phase 2: Vector DB Setup & Configuration | Provision instance, configure indexes, connection layer, embedding storage | 6–8 hrs |
# MAGIC | Phase 3: Document Parsing & Chunking Pipeline | Parser, chunker, ingestion pipeline, edge cases, error handling | 8–10 hrs |
# MAGIC | Phase 4: Embedding Pipeline | Integrate model, batch embedding, rate limits, storage format | 4–6 hrs |
# MAGIC | Phase 5: Search Implementation | Keyword search, semantic search, hybrid with configurable ratio, ranking/fusion, query optimization, latency | 12–16 hrs |
# MAGIC | Phase 6: Testing & Tuning | Unit tests, integration tests, accuracy testing, latency benchmarking, ratio tuning, edge cases | 8–10 hrs |
# MAGIC | Phase 7: Documentation & Deployment | Architecture docs, API docs, deployment guide, runbook, knowledge transfer | 6–8 hrs |
# MAGIC | **Total** | | **52–70 hours (7–9 working days)** |

# COMMAND ----------

# DBTITLE 1,Comparison Table
# MAGIC %md
# MAGIC ## Comparison Table
# MAGIC
# MAGIC | Factor | Approach 1: Existing Infra | Approach 2: New DB |
# MAGIC |--------|---------------------------|-------------------|
# MAGIC | Total Effort | 19–27 hours | 52–70 hours |
# MAGIC | Risk Level | Low — proven infrastructure | Medium — new setup, unknowns |
# MAGIC | Time to Value | 3–4 working days | 7–9 working days |
# MAGIC | Schema Design | Minimal — adapt to existing | Full — design from scratch |
# MAGIC | Search Implementation | Integrate with existing | Build from scratch |
# MAGIC | Hybrid Search | Tune existing ratio | Build keyword + semantic + fusion |
# MAGIC | Latency Optimization | Baseline already known | Must benchmark from zero |
# MAGIC | Scalability | Limited by existing DB capacity | Full control over scaling |
# MAGIC | Maintenance | Shared with existing system | Dedicated maintenance burden |
# MAGIC | New Module Onboarding | Add to existing pipeline | Well-documented, self-service |
# MAGIC | Cost | Low — reuse existing infra | Higher — separate infra costs |
# MAGIC | Team Size | 1 developer | 1 developer |

# COMMAND ----------

# DBTITLE 1,Recommendation
# MAGIC %md
# MAGIC ## Recommendation
# MAGIC
# MAGIC **Recommend Approach 1 — Leverage Existing Infrastructure**
# MAGIC
# MAGIC Key reasons:
# MAGIC - **3–4 days vs 7–9 days** — nearly half the effort
# MAGIC - **Lower risk** — the infrastructure is already proven in production
# MAGIC - **Immediate search** — existing search stack is ready to use
# MAGIC - **Faster feedback loop** — prototype and validate quickly
# MAGIC
# MAGIC ### When to choose Approach 2 instead:
# MAGIC
# MAGIC - Existing DB faces **scale issues** with the new document volume
# MAGIC - You need a **different schema** that doesn't fit the existing model
# MAGIC - You want **full control** over the entire stack
# MAGIC - Existing search quality is **insufficient** and cannot be tuned further
# MAGIC - **Cost/licensing constraints** make the existing infra unsustainable

# COMMAND ----------

# DBTITLE 1,Next Steps
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Review existing AWS vector DB schema** for module guide compatibility
# MAGIC 2. **Validate existing search** handles guide document types (tables, headings, multi-page)
# MAGIC 3. **Build a prototype**: parse one module guide → chunk → store → run test queries
# MAGIC 4. **Measure latency and accuracy** on the prototype
# MAGIC 5. **Finalize approach** based on prototype results — proceed with Approach 1 or pivot to Approach 2

# COMMAND ----------

# DBTITLE 1,Effort Summary Calculator
# Effort Summary Calculator
# RAG Glossary & Taxonomy Explorer — Effort Estimation

HOURS_PER_DAY = 8

approach1 = {
    "name": "Approach 1: Leverage Existing Infrastructure",
    "phases": [
        ("Phase 1: Document Analysis & Parsing", 4, 6),
        ("Phase 2: Chunking Strategy", 3, 4),
        ("Phase 3: Embedding & Storage", 4, 6),
        ("Phase 4: Search Integration & Testing", 6, 8),
        ("Phase 5: Documentation & Handoff", 2, 3),
    ],
}

approach2 = {
    "name": "Approach 2: Build New Separate DB from Scratch",
    "phases": [
        ("Phase 1: Schema Design & Architecture", 8, 12),
        ("Phase 2: Vector DB Setup & Configuration", 6, 8),
        ("Phase 3: Document Parsing & Chunking Pipeline", 8, 10),
        ("Phase 4: Embedding Pipeline", 4, 6),
        ("Phase 5: Search Implementation", 12, 16),
        ("Phase 6: Testing & Tuning", 8, 10),
        ("Phase 7: Documentation & Deployment", 6, 8),
    ],
}

def print_summary(approach):
    print(f"\n{'=' * 70}")
    print(f"  {approach['name']}")
    print(f"{'=' * 70}")
    print(f"  {'Phase':<45} {'Min hrs':>8} {'Max hrs':>8}")
    print(f"  {'-' * 45} {'-' * 8} {'-' * 8}")

    total_min = 0
    total_max = 0
    for phase_name, min_hrs, max_hrs in approach["phases"]:
        print(f"  {phase_name:<45} {min_hrs:>8} {max_hrs:>8}")
        total_min += min_hrs
        total_max += max_hrs

    print(f"  {'-' * 45} {'-' * 8} {'-' * 8}")
    print(f"  {'TOTAL':<45} {total_min:>8} {total_max:>8}")
    print(f"  {'Working Days (8 hrs/day)':<45} {total_min / HOURS_PER_DAY:>7.1f}d {total_max / HOURS_PER_DAY:>7.1f}d")
    print()

print_summary(approach1)
print_summary(approach2)

# Comparison
print(f"{'=' * 70}")
print("  Comparison Summary")
print(f"{'=' * 70}")
print(f"  {'Metric':<30} {'Approach 1':>15} {'Approach 2':>15}")
print(f"  {'-' * 30} {'-' * 15} {'-' * 15}")

a1_min = sum(p[1] for p in approach1["phases"])
a1_max = sum(p[2] for p in approach1["phases"])
a2_min = sum(p[1] for p in approach2["phases"])
a2_max = sum(p[2] for p in approach2["phases"])

print(f"  {'Total Hours (min)':<30} {a1_min:>15} {a2_min:>15}")
print(f"  {'Total Hours (max)':<30} {a1_max:>15} {a2_max:>15}")
print(f"  {'Working Days (min)':<30} {a1_min / HOURS_PER_DAY:>14.1f}d {a2_min / HOURS_PER_DAY:>14.1f}d")
print(f"  {'Working Days (max)':<30} {a1_max / HOURS_PER_DAY:>14.1f}d {a2_max / HOURS_PER_DAY:>14.1f}d")
print(f"  {'Hour Savings (min)':<30} {a2_min - a1_min:>15}")
print(f"  {'Hour Savings (max)':<30} {a2_max - a1_max:>15}")
print(f"  {'Day Savings (min)':<30} {(a2_min - a1_min) / HOURS_PER_DAY:>14.1f}d")
print(f"  {'Day Savings (max)':<30} {(a2_max - a1_max) / HOURS_PER_DAY:>14.1f}d")
print()