# Capstone — The Meridian Intelligence Platform

## The scenario

Module 1 opened with a buyer's 3:14am text: "why is SKU-48213 out again?" — it took 6 days and
$340K to answer. Twenty-one modules later, you've built every individual piece that could have
closed that gap. This capstone wires all of it into **one platform for one company: Meridian
Foods**, a mid-size CPG brand — not a toy demo, the same POS, vendor, catalog, and loyalty data
shapes used all course.

The brief is explicit: **one platform, one role model, one cost story** — not four disconnected
pilots. Every capability reused here already exists in this course; the only new work is the
wiring — how these pieces hand data and control to each other.

Meridian's four data rivers, never joined until now:
- **POS data** — daily sell-through by SKU, store, and region (feeds forecasting + anomaly detection)
- **Vendor documents** — contracts, purchase orders, shipment confirmations, mostly PDFs in a stage
- **Product catalog** — SKU descriptions, formulations, review text
- **Loyalty data** — member purchase history, tying category-manager questions to real customer behavior

## Architecture (three layers, top to bottom)

```
                 ┌─────────────────────────────────────────────┐
                 │   GOVERNANCE  (masking policy + resource     │
                 │   monitor — wraps every layer below)         │
                 └─────────────────────────────────────────────┘
  TOP LAYER            one Streamlit app  ──calls──►  one orchestrating agent
                              (06)                        (05)
                                                             │
                                              ┌──────────────┼──────────────┬──────────────┐
                                              ▼              ▼              ▼              ▼
  MIDDLE LAYER              Cortex Search   Cortex Analyst   ML.FORECAST   ML.ANOMALY_DETECTION
                            (02)            semantic view    (04)          (04)
                                             (03)
                                              ▲              ▲              ▲              ▲
                                              │              │              │              │
                             document pipeline (01)     POS + catalog + loyalty tables
                                              │                       │
  BOTTOM LAYER              raw vendor documents         raw POS, catalog, loyalty tables
                            (stage)                       (Snowflake tables)
```

Every arrow in this diagram maps to a module you already completed:
- Document pipeline → module 11 (AI_PARSE_DOCUMENT / AI_EXTRACT)
- Cortex Search → module 7
- Cortex Analyst semantic model → modules 9-10
- Forecasting → module 12
- Anomaly detection → module 13
- Orchestrating agent → modules 15-16
- Streamlit app → module 19
- Governance (masking + resource monitors) → module 18

## Build order

Assemble the files in this folder in this sequence (matches lesson_84 → lesson_85 → lesson_86):

1. **`01_document_ingestion.sql`** — parse Meridian's 1,400+ vendor contracts with
   `AI_PARSE_DOCUMENT`, extract structured terms with `AI_EXTRACT`.
2. **`02_search_index.sql`** — stand up `CORTEX SEARCH SERVICE meridian_field_search` over the
   unified vendor/catalog/review text so field reps get one search box.
3. **`03_semantic_model.yaml`** — define the Cortex Analyst semantic model over POS, catalog, and
   loyalty so a category manager's "sell-through by region" always means the same query.
4. **`04_forecast_and_anomaly.sql`** — train `SNOWFLAKE.ML.FORECAST` on sell-through history and
   `SNOWFLAKE.ML.ANOMALY_DETECTION` on vendor lead times, so the platform flags risk before
   anyone asks.
5. **`05_orchestrating_agent.sql`** — register one agent with all four tools (search, analyst,
   forecast, anomaly) and routing instructions, replacing the five disconnected systems from
   lesson_01 with one front door.
6. **`06_streamlit_app.py`** — the chat-style front end that calls the agent and renders its
   response, inside the same governed Snowflake account (no external hosting, no separate API key).

Governance (masking policy on `loyalty_members.email` + a resource monitor on `meridian_wh`) is
layered across every step above, not built as a separate numbered file — apply module 18's
checklist to each object as you create it.

## What "done" looks like

- A category manager can ask a natural-language question and get an answer grounded in real tables.
- A field rep can search vendor contracts and reviews without touching raw PDFs directly.
- A stockout-style incident surfaces same-day, with forecast and anomaly signals attached automatically.
- Every query respects role-based access, and every AI call stays inside a tracked cost budget.
- Ask the agent the exact question from lesson_01 — "why is SKU-48213 out again?" — and it calls
  Anomaly Detection, then Forecast, then Search, and folds all three into one plain-English answer
  in a single conversational turn.

## Grading rubric (6 checkpoints)

| # | Checkpoint | Pass criteria |
|---|------------|----------------|
| 1 | Document ingestion | `vendor_contracts_parsed` and `vendor_terms` populate from a real stage via one `DIRECTORY()` scan — no per-file script. |
| 2 | Search index | `meridian_field_search` is created `ON search_text` with `ATTRIBUTES doc_type, sku_id`, a named `WAREHOUSE`, and `TARGET_LAG` set (not left at default). |
| 3 | Semantic model | Tables, relationships, and metrics are declared with the facts/dimensions/synonyms correctly nested **inside** each table's block (not flattened) — `sell_through_units` and `shelf_revenue` resolve to one definition each. |
| 4 | Forecast + anomaly | Both `SNOWFLAKE.ML.FORECAST` and `SNOWFLAKE.ML.ANOMALY_DETECTION` are created with a `SERIES_COLNAME`, and `DETECT_ANOMALIES` is called against a *new* events view, not the training view. |
| 5 | Agent + app | The agent registers all four tools with `INSTRUCTIONS` that route by question type, and the Streamlit app's only logic is passing text to the agent and rendering `RESPONSE` — no duplicated business logic in the UI layer. |
| 6 | Governance | A masking policy hides `loyalty_members.email` from every role except the compliance role, and a resource monitor caps warehouse spend with a `NOTIFY`/`SUSPEND` threshold pair. |

A passing capstone closes all five silos named in lesson_01 (vendor documents, structured
sell-through, inventory/ERP forward risk, cross-signal support, and one governed front door) —
not just some of them.
