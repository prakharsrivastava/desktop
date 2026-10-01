# ============================================================
# Course 1018 — Module 15
# On-slide PY reference  (1 snippets)
# Authentic code as shown on the course slides. Copy, run, experiment.
# ============================================================


# ----------------------------------------------------------
# Anti-Pattern First — A Hand-Rolled If/Else Tool Router
# ----------------------------------------------------------

# WRONG — Priya's first instinct: bypass the agent's own routing, hand-code it
def pick_tool(question: str):
    q = question.lower()
    if "how many" in q or "total" in q or "$" in q:
        return "cortex_analyst_sales_model"
    elif "why" in q or "policy" in q:
        return "cortex_search_returns_docs"
    else:
        return "custom_pricing_proc"  # default guess — zero evidence

# RIGHT — agent-level routing instructions (plain-language tie-breakers)
When the user asks about NUMBERS, TOTALS, or TRENDS in sales/returns,
route to cortex_analyst_sales_model.
When the user asks WHY, POLICY, or references a DOCUMENT/NOTICE,
route to cortex_search_returns_docs.

