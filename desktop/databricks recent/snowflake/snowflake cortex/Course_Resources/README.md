# Snowflake Cortex AI Masterclass — Course Resources

Everything you need to follow along hands-on, in the order you'll use it.

## What's in here

- **`setup/`** — start here. `environment_setup.md` gets you a working Snowflake trial
  environment (Cortex access confirmed, database, warehouse, sample data) matching every
  exercise in this course.
- **`slide_queries/`** — every piece of runnable SQL shown on-screen, one file per module
  plus `1018_all_queries.sql` with everything combined. Copy-paste-friendly; this is the
  authentic code from the slides, not a paraphrase.
- **`slide_code/`** — non-SQL on-slide content (semantic-model YAML, the Streamlit app's
  Python, a couple of shell snippets), grouped the same way as `slide_queries/`.
- **`exercises/`** — hands-on practice per module. Each file has 1-2 scenario-based
  exercises grounded in the course's running CPG story (product reviews, POS/shipment data,
  product documents): a SCENARIO, a TASK, space for YOUR CODE, and a SOLUTION to check
  yourself against afterward.
- **`capstone/`** — the end-to-end capstone project skeleton (module 22, the CPG
  intelligence platform tying every prior module together). Start with `capstone/README.md`
  for the brief, architecture, build order, and grading rubric, then work through
  `01_document_ingestion.sql` → `02_search_index.sql` → `03_semantic_model.yaml` →
  `04_forecast_and_anomaly.sql` → `05_orchestrating_agent` → `06_streamlit_app.py` in order.

## How to use this

1. Read `setup/environment_setup.md` and provision your environment (~20-30 min, one time) —
   including confirming Cortex access is actually enabled on your account.
2. Work through `exercises/module_01_*` onward, in module order — later modules assume
   objects created in earlier ones exist (a semantic model from module 9, a search service
   from module 8, etc.). Try each exercise's TASK yourself before reading its SOLUTION.
3. Use `slide_queries/module_NN_*.sql` as a reference if you want to revisit exactly what a
   lesson showed on screen, without re-watching the video.
4. When you reach module 22, do the capstone — it's the intended culmination of everything
   before it: document parsing, search, a semantic model, forecasting and anomaly
   detection, an orchestrating agent, and a governed Streamlit front end, all on one
   consistent CPG dataset.

## A note on accuracy

Every snippet in this bundle is grounded in the actual on-slide code or the course's
recorded teaching content — nothing here is invented. This course went through an
unusually thorough fact-audit pass before publish (two full passes, both before and after
slide/narration authoring), which caught and fixed several real errors worth knowing about
if you see an OLDER tutorial elsewhere that disagrees with this bundle:

- **Cortex usage cost/monitoring** uses `SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY`
  (column `TOKEN_CREDITS`) — the OLDER `CORTEX_FUNCTIONS_USAGE_HISTORY` view is deprecated
  and no longer updated. If an older reference uses that view name, it's stale, not this
  bundle.
- **`SNOWFLAKE.ML.TOP_INSIGHTS`** is the real contribution-analysis class — not
  `CONTRIBUTION_EXPLORER`, which doesn't exist as a SQL identifier (the marketing name
  "Contribution Explorer" refers to the same underlying feature, but the class you actually
  call is `TOP_INSIGHTS`, with a `LABEL_COLNAME` parameter, not `BASELINE_PERIOD`).
  Module 13 covers this.
- **Semantic model YAML nesting**: `facts:`/`dimensions:`/`synonyms:` nest INSIDE each
  table's own block, not flat at the top level of the file. Module 9 and the capstone's
  `03_semantic_model.yaml` both use the correct nested structure.
- **`AI_SETTINGS` guardrails** use a nested YAML config block (`$$ guardrails:
  advanced_prompt_injection: - enabled: true $$`), not a flat key-value pair. Module 18.
- **`AI_TRANSLATE`** requires all 3 positional arguments — `(text, source_language,
  target_language)` — pass `''` for `source_language` to auto-detect; it's never a 2-argument
  call. Modules 1 and 4.

Questions or found something that doesn't work as written? That's useful signal — Cortex AI
is a fast-moving product area; check `docs.snowflake.com` for the current syntax if
something here ever falls out of date.
