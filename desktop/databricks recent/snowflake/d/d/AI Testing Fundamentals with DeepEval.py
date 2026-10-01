# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # AI Testing Fundamentals: Testing AI Systems with DeepEval
# MAGIC
# MAGIC This notebook explains all concepts from the **AI Testing Fundamentals** course — how to evaluate AI agents, chatbots, and RAG applications using the **DeepEval** framework. Traditional software testing relies on exact assertions (expected vs. actual), but AI outputs are **non-deterministic** — the same question can yield differently worded but semantically equivalent answers every time. This notebook covers:
# MAGIC
# MAGIC * How AI testing differs from traditional QA (evals vs. assertions)
# MAGIC * DeepEval metrics: Task Completion, Answer Relevancy, Tool Correctness, Prompt Alignment, Step Efficiency
# MAGIC * Black-box testing (LLMTestCase + evaluate) and White-box testing (tracing + evals_iterator)
# MAGIC * Custom metrics with **G-Eval** for dimensions not covered by built-in metrics
# MAGIC * Multi-turn chatbot testing with **ConversationalTestCase**
# MAGIC * RAG-specific metrics: **Contextual Precision**, **Contextual Recall**, **Faithfulness**
# MAGIC * Synthetic data generation and safety metrics (Bias, Toxicity, PII leakage)

# COMMAND ----------

# DBTITLE 1,Traditional vs AI Testing
# MAGIC %md
# MAGIC ## 1. Traditional Testing vs. AI Testing
# MAGIC
# MAGIC ### Traditional Software Testing
# MAGIC * **Deterministic**: You know the exact expected output for every input.
# MAGIC * **Assertions**: `assert actual == expected` — word-by-word, letter-by-letter match.
# MAGIC * Example: Click a button → verify a specific page title appears.
# MAGIC
# MAGIC ### AI Systems Testing
# MAGIC * **Non-deterministic**: Same question → different wording each time, but same *intent*.
# MAGIC * You **cannot** use exact string assertions.
# MAGIC * Even regex / substring checks are fragile — "order is shipped" might become "order is on its way."
# MAGIC
# MAGIC ### The Shift in Mindset
# MAGIC Instead of asking *"did the agent produce the exact output?"*, we ask:
# MAGIC
# MAGIC | Dimension | What We Check | Example Metric |
# MAGIC |-----------|-------------|---------------|
# MAGIC | **Semantic correctness** | Is the *meaning* correct? | Faithfulness, G-Eval |
# MAGIC | **Answer relevancy** | Is the response *on-topic*? | Answer Relevancy |
# MAGIC | **Tool behavior** | Did the agent call the *right tool*? | Tool Correctness |
# MAGIC | **Prompt alignment** | Did the agent follow system prompt instructions? | Prompt Alignment |
# MAGIC | **Step efficiency** | Did the agent take the *minimum* steps? | Step Efficiency |
# MAGIC | **Safety** | Any bias, toxicity, or PII leakage? | Bias, Toxicity, PII |
# MAGIC
# MAGIC > In the AI world, **testing = evaluation** ("evals"). Each metric produces a **score from 0 to 1**, and you set a **threshold** (e.g., 0.7 for quality, 0.5 for safety) to decide pass/fail.

# COMMAND ----------

# DBTITLE 1,DeepEval Terminology
# MAGIC %md
# MAGIC ## 2. DeepEval Key Terminology
# MAGIC
# MAGIC | Term | Meaning |
# MAGIC |------|---------|
# MAGIC | **Evals** | Structured, repeatable process for measuring AI output quality on a specific dimension. |
# MAGIC | **Metrics** | Pre-built or custom classes that score AI output from 0–1 (e.g., `TaskCompletionMetric`, `AnswerRelevancyMetric`). |
# MAGIC | **Goldens** | Test cases — the input + expected output + other metadata you feed to metrics. Created via `LLMTestCase` (single-turn) or `ConversationalTestCase` (multi-turn). |
# MAGIC | **Judge LLM** | A separate LLM (e.g., GPT-4o) that provides "intelligence" to metrics. Metrics have the formula but need an LLM brain to read outputs and score them. Best practice: use a *different vendor* for the judge than the one used to build the agent. |
# MAGIC | **Tracing** | Attaching a "spy" context to agent calls so DeepEval can track every internal step (tool calls, intermediate results) for white-box testing. |
# MAGIC | **Threshold** | The minimum score to pass. Quality metrics: 0.7+. Safety metrics (bias/toxicity): score must be *below* 0.5. |
# MAGIC | **Confident AI** | DeepEval's cloud dashboard for visualizing test results, traces, and metric reports.

# COMMAND ----------

# DBTITLE 1,Install DeepEval
# MAGIC %pip install deepeval typing_extensions>=4.12.0 --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import DeepEval Classes
# Import core DeepEval classes (run after the install cell above)
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn, ToolCall
from deepeval.metrics import (
    TaskCompletionMetric,
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    PromptAlignmentMetric,
    StepEfficiencyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    BiasMetric,
    ToxicityMetric,
    GEval,
)
# PII metric name varies across DeepEval versions
try:
    from deepeval.metrics import PIIViolationMetric as PIIMetric
except ImportError:
    try:
        from deepeval.metrics import PIIMetric
    except ImportError:
        PIIMetric = None
        print("⚠️ PII metric not available in this DeepEval version — skipping")
# Use new param names (old names LLMTestCaseParams/TurnParams are deprecated)
try:
    from deepeval.test_case import SingleTurnParams as LLMTestCaseParams, MultiTurnParams as TurnParams
except ImportError:
    from deepeval.test_case import LLMTestCaseParams, TurnParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.synthesizer import Synthesizer

print("✓ DeepEval imported successfully")

# COMMAND ----------

# DBTITLE 1,Mock AI Agent for Testing
# ---- MOCK AI AGENT ----
# In a real project, this would be your dev team's agent built with LangChain.
# Here we simulate a customer support agent with two "tools":
#   1. get_order_status(order_id) -> returns shipping info
#   2. get_refund_policy(category) -> returns refund policy text

ORDERS = {
    "1042": {"status": "shipped", "eta": "May 13, 2026"},
    "1099": {"status": "not_found", "eta": None},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days of delivery if unopened. Refunds take 5-7 business days.",
    "clothing": "Clothing can be returned within 30 days of delivery. Refunds take 3-5 business days.",
}

SYSTEM_PROMPT = (
    "You are a friendly customer support agent. "
    "Use the available tools to answer order status and refund questions. "
    "Keep replies short and helpful."
)

def get_order_status(order_id: str) -> str:
    """Tool: retrieve order status by order ID."""
    order = ORDERS.get(order_id)
    if not order:
        return f"Order {order_id} not found."
    if order["status"] == "shipped":
        return f"Order {order_id} is shipped. Estimated arrival: {order['eta']}."
    return f"Order {order_id} status: {order['status']}."

def get_refund_policy(category: str) -> str:
    """Tool: retrieve refund policy by product category."""
    return REFUND_POLICIES.get(category, f"No refund policy found for {category}.")

def support_agent(user_input: str) -> str:
    """
    The core invocable method for the AI agent.
    In production, this would invoke an LLM with tools.
    Here we simulate LLM intelligence with simple keyword matching.
    """
    import re
    # Simulate LLM picking the right tool based on user query
    order_match = re.search(r'\b(\d{4})\b', user_input)
    if order_match and ('order' in user_input.lower() or 'where' in user_input.lower()):
        return get_order_status(order_match.group(1))
    if 'refund' in user_input.lower() or 'return' in user_input.lower():
        for cat in REFUND_POLICIES:
            if cat in user_input.lower():
                return get_refund_policy(cat)
        return get_refund_policy('electronics')  # default
    return "I can help with order status and refund policies. What do you need?"

# Quick test
print(support_agent("Where is my order 1042?"))
print(support_agent("What is the refund policy for electronics?"))
print(support_agent("Where is my order 1099?"))

# COMMAND ----------

# DBTITLE 1,Goldens & Black Box Testing
# MAGIC %md
# MAGIC ## 3. Goldens & Black-Box Testing
# MAGIC
# MAGIC ### What are Goldens?
# MAGIC Goldens are your **test cases** — each golden contains:
# MAGIC * `input`: The user question sent to the AI agent
# MAGIC * `actual_output`: The response the agent returned (collected at runtime or from production logs)
# MAGIC * `expected_output`: (optional) What you expect — needed for metrics like factual correctness
# MAGIC * `expected_tools`: (optional) Which tools the agent *should* call — needed for Tool Correctness
# MAGIC
# MAGIC ### Black-Box Testing (LLMTestCase + evaluate)
# MAGIC * Simplest approach — you only care about input → output.
# MAGIC * You call the agent, collect the response, wrap it in an `LLMTestCase`, and pass to `evaluate()`.
# MAGIC * No tracing needed. Use this when you **don't** have access to the dev codebase (third-party agents) or when you don't need internal component data.
# MAGIC
# MAGIC ### Key Point: Judge LLM
# MAGIC Every metric that needs to *read and understand* the output requires a **judge LLM**. Best practice: use a **different vendor** for the judge than the one used to build the agent (e.g., build with Claude, judge with GPT-4o) to avoid self-grading bias.

# COMMAND ----------

# DBTITLE 1,Black Box Task Completion + Answer Relevancy
# ---- BLACK-BOX: TASK COMPLETION + ANSWER RELEVANCY ----
# Task Completion: Did the agent complete the task end-to-end without asking unnecessary follow-up questions?
# Answer Relevancy: Is the response on-topic and relevant to the user's question?

# Step 1: Call the agent to get actual output
user_input = "Where is my order 1042?"
actual_output = support_agent(user_input)
print(f"Agent response: {actual_output}")

# Step 2: Create a golden (LLMTestCase) — wraps input + actual_output
test_case = LLMTestCase(
    input=user_input,
    actual_output=actual_output,
)

# Step 3: Define metrics with threshold and judge model
# NOTE: These require an OpenAI API key. Uncomment when configured.
# On Databricks: os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope='openai-secrets', key='api-key')

# task_completion_metric = TaskCompletionMetric(
#     threshold=0.7,        # pass if score >= 0.7
#     model='gpt-4o',       # judge LLM — provides intelligence to the metric
# )
#
# answer_relevancy_metric = AnswerRelevancyMetric(
#     threshold=0.7,
#     model='gpt-4o',
# )

# Step 4: Evaluate — pass the test case(s) and metric(s)
# Uncomment to run (requires valid OpenAI API key):
# evaluate(
#     test_cases=[test_case],
#     metrics=[task_completion_metric, answer_relevancy_metric],
# )

print("\n--- Black-Box Evaluation Structure ---")
print(f"Golden (LLMTestCase): input='{test_case.input}', actual_output='{test_case.actual_output}'")
print(f"Metrics: TaskCompletionMetric(threshold=0.7), AnswerRelevancyMetric(threshold=0.7)")
print(f"Judge model: gpt-4o (different vendor recommended)")
print("\nUncomment the evaluate() call above to run with a valid OpenAI API key.")

# COMMAND ----------

# DBTITLE 1,Tool Correctness Concept
# MAGIC %md
# MAGIC ## 4. Tool Correctness Metric
# MAGIC
# MAGIC Evaluates whether the agent called the **right tool** for the user's question.
# MAGIC
# MAGIC * **Does NOT need a judge LLM** — it's a simple assertion: did the agent call the expected tool?
# MAGIC * You provide `expected_tools` in the golden, and the metric checks if the actual tool matches.
# MAGIC * If you ask "where is my order?" and the agent calls `get_order_status` → correct.
# MAGIC * If it calls `get_refund_policy` instead → wrong tool → test fails.
# MAGIC
# MAGIC ### Black-Box vs White-Box for Tool Correctness
# MAGIC * **Black-box**: You hard-code `actual_tools_called` from production logs (when you don't have code access).
# MAGIC * **White-box (tracing)**: Tracing automatically captures which tools were called at runtime — no hard-coding needed.

# COMMAND ----------

# DBTITLE 1,Black Box Tool Correctness
# ---- BLACK-BOX: TOOL CORRECTNESS ----
# Tool Correctness checks if the agent called the expected tool.
# This metric does NOT need a judge LLM — it's a direct comparison.

from deepeval.test_case import ToolCall

test_case_tool = LLMTestCase(
    input="Where is my order 1042?",
    actual_output="Order 1042 is shipped. Estimated arrival: May 13, 2026.",
    expected_tools=[
        ToolCall(
            name="get_order_status",
            description="Retrieve order status by order ID",
        )
    ],
    actual_tools_called=[
        # In black-box mode, you hard-code this from production logs.
        # In white-box mode (tracing), this is captured automatically.
        ToolCall(
            name="get_order_status",
            description="Retrieve order status by order ID",
        )
    ],
)

# Uncomment with valid OpenAI API key:
# tool_correctness_metric = ToolCorrectnessMetric(
#     threshold=0.7,
#     # No model needed! Direct assertion comparison.
# )

# Uncomment to run:
# evaluate(
#     test_cases=[test_case_tool],
#     metrics=[tool_correctness_metric],
# )

print("Tool Correctness Metric: no judge LLM needed (direct comparison)")
print(f"Expected tool: get_order_status")
print(f"Actual tool called: get_order_status")
print(f"Expected: PASS (tools match)")
print("\nIf the agent called 'get_refund_policy' instead, this would FAIL.")

# COMMAND ----------

# DBTITLE 1,White Box Tracing Concept
# MAGIC %md
# MAGIC ## 5. White-Box Testing with Tracing
# MAGIC
# MAGIC When you **have access to the dev codebase**, you can trace internal agent behavior — which tools were called, intermediate results, reasoning steps — automatically at runtime.
# MAGIC
# MAGIC ### Key Differences from Black-Box
# MAGIC | Aspect | Black-Box | White-Box (Tracing) |
# MAGIC |--------|-----------|---------------------|
# MAGIC | Test data class | `LLMTestCase` + `evaluate()` | `EvaluationDataset` + `evals_iterator()` |
# MAGIC | Actual output | You collect manually | Captured automatically by tracing |
# MAGIC | Tool calls | Hard-coded from logs | Captured automatically by tracing |
# MAGIC | Tracing setup | Not needed | `@observe` decorator + callback handler in dev code |
# MAGIC
# MAGIC ### How Tracing Works
# MAGIC 1. **In the test file**: Override the agent's invocable method with `@observe(name="support_agent")` decorator.
# MAGIC 2. **In the dev file**: Your dev team adds a `DeepEvalCallbackHandler` and passes it to the agent's `invoke()` call as a config. This captures every LangChain LLM call, tool call, and chain step.
# MAGIC 3. The trace context travels with the method call (like a "spy camera"), collects all internal data, and hands it to `evals_iterator`.
# MAGIC 4. Expected values (e.g., `expected_tools`, `expected_output`) from your goldens are attached to the trace via `update_current_trace`.

# COMMAND ----------

# DBTITLE 1,White Box Tracing Implementation
# ---- WHITE-BOX: TRACING + EVALS_ITERATOR ----
# This is the recommended approach when you have access to the dev codebase.
# Tracing automatically captures actual_output, actual_tools_called, and intermediate steps.

from deepeval.tracing import observe, update_current_trace
try:
    from deepeval.tracing.context import get_current_trace
except ImportError:
    try:
        from deepeval.tracing import get_current_trace
    except ImportError:
        get_current_trace = None
        print("⚠️ get_current_trace not available in this DeepEval version")
from deepeval.dataset import EvaluationDataset, Golden

# Step 1: Override the agent method with @observe decorator
# This opens a tracing context ("spy camera") that tracks every internal step.
@observe(name="support_agent")
def support_agent_traced(user_input: str):
    """Wrapper that attaches tracing before calling the real agent."""
    return support_agent(user_input)  # calls the real dev agent

# Step 2: Define goldens with expected values
# Each Golden = one test case. Provide expected_tools for Tool Correctness.
goldens = [
    Golden(
        input="Where is my order 1042?",
        expected_tools=[
            ToolCall(name="get_order_status", description="Retrieve order status")
        ],
    ),
    Golden(
        input="What is the refund policy for electronics?",
        expected_tools=[
            ToolCall(name="get_refund_policy", description="Retrieve refund policy")
        ],
    ),
]

# Step 3: Create EvaluationDataset with goldens
dataset = EvaluationDataset(goldens=goldens)

# Step 4: Define metrics (uncomment with valid OpenAI API key)
# task_completion = TaskCompletionMetric(threshold=0.7, model='gpt-4o')
# tool_correctness = ToolCorrectnessMetric(threshold=0.7)  # No LLM needed!

# Step 5: Iterate through each golden, call the traced agent, attach expected values
# for golden in dataset.evals_iterator(metrics=[task_completion, tool_correctness]):
#     # Call the agent with tracing attached
#     response = support_agent_traced(golden.input)
#     
#     # Attach expected values from the golden to the current trace
#     trace = get_current_trace()
#     if golden.expected_tools:
#         update_current_trace(
#             expected_tools=golden.expected_tools,
#         )
#     # The trace now has: actual_output (from runtime) + expected_tools (from golden)
#     # evals_iterator reads the trace and calculates metric scores.

print("White-Box Tracing Setup:")
print(f"  Goldens: {len(goldens)} test cases")
print(f"  Metrics: TaskCompletionMetric(0.7, gpt-4o), ToolCorrectnessMetric(0.7)")
print(f"  Tracing: @observe decorator + DeepEvalCallbackHandler in dev code")
print(f"  Key: actual_output and actual_tools captured AUTOMATICALLY by tracing")

# COMMAND ----------

# DBTITLE 1,Prompt Alignment Concept
# MAGIC %md
# MAGIC ## 6. Prompt Alignment Metric
# MAGIC
# MAGIC Evaluates whether the agent's response **follows its system prompt instructions**.
# MAGIC
# MAGIC * The system prompt sets the agent's persona: *"You are a friendly customer support agent. Keep replies short and helpful."*
# MAGIC * If the agent gives paragraph-length responses when told to keep it short → low score.
# MAGIC * Requires: `prompt_instructions` (from dev code), `input`, `actual_output`.
# MAGIC * Needs a judge LLM to read the output and compare against prompt instructions.
# MAGIC
# MAGIC ### Example Finding
# MAGIC If the prompt says "keep replies short" but the agent adds extra details like delivery date and "Is there anything else I can help you with?" → the prompt alignment metric may **fail with score 0**, because the response violates the "short" instruction. You then decide: update the prompt or fix the agent.

# COMMAND ----------

# DBTITLE 1,Prompt Alignment Evaluation
# ---- PROMPT ALIGNMENT METRIC ----
# Checks if the agent's response follows the system prompt instructions.

# Uncomment with valid OpenAI API key:
# prompt_alignment_metric = PromptAlignmentMetric(
#     prompt_instructions=[
#         "You are a friendly customer support agent.",
#         "Use the available tools to answer order status and refund questions.",
#         "Keep replies short and helpful.",
#     ],
#     threshold=0.7,
#     model='gpt-4o',  # judge LLM needed to read output and compare to instructions
# )

PROMPT_INSTRUCTIONS = [
    "You are a friendly customer support agent.",
    "Use the available tools to answer order status and refund questions.",
    "Keep replies short and helpful.",
]

# Using white-box tracing (actual_output captured automatically):
goldens_prompt = [
    Golden(input="Where is my order 1042?"),
    Golden(input="What is the refund policy for electronics?"),
    Golden(input="I want to return my order. What should I do?"),
]

dataset_prompt = EvaluationDataset(goldens=goldens_prompt)

# for golden in dataset_prompt.evals_iterator(metrics=[prompt_alignment_metric]):
#     support_agent_traced(golden.input)
#     # No need to attach expected values — prompt alignment only needs actual_output
#     # which tracing captures automatically.

print("Prompt Alignment Metric:")
print(f"  Prompt instructions: {len(PROMPT_INSTRUCTIONS)} rules from system prompt")
print(f"  Threshold: 0.7 (response must follow 70%+ of instructions)")
print(f"  Common failure: agent adds extra details when told to 'keep it short'")

# COMMAND ----------

# DBTITLE 1,Step Efficiency Concept
# MAGIC %md
# MAGIC ## 7. Step Efficiency Metric
# MAGIC
# MAGIC Evaluates whether the agent completed the task in the **minimum number of steps** — no redundant tool calls or unnecessary reasoning.
# MAGIC
# MAGIC * If the agent has 5 tools but only needs 1 to answer "where is my order?" → it should call just that 1.
# MAGIC * Calling extra tools wastes latency and increases API cost.
# MAGIC * This is a **pure trace-level metric** — you need tracing (white-box) to measure the execution path.
# MAGIC * Suggested threshold: **0.5** (not 0.7) — the output quality might still be fine even if the path isn't perfectly optimal.
# MAGIC * Needs a judge LLM to evaluate the full execution trace.
# MAGIC
# MAGIC ### Example Finding
# MAGIC Agent adds "celebratory language and emojis" — unnecessary stylistic enhancement. The task could have been completed with a direct tool call. Score: 0.5 (borderline pass).

# COMMAND ----------

# DBTITLE 1,Step Efficiency Evaluation
# ---- STEP EFFICIENCY METRIC ----
# Checks if the agent completed the task in minimum steps (no redundant tool calls).
# Pure trace-level metric — requires tracing (white-box).

# Uncomment with valid OpenAI API key:
# step_efficiency_metric = StepEfficiencyMetric(
#     threshold=0.5,  # lower threshold — we care about the journey, not just output quality
#     model='gpt-4o',
# )

# Can be combined with other metrics in the same evals_iterator:
# for golden in dataset.evals_iterator(
#     metrics=[task_completion, tool_correctness, prompt_alignment_metric, step_efficiency_metric]
# ):
#     support_agent_traced(golden.input)
#     # Tracing captures everything: output, tools, reasoning steps, intermediate results
#     # All 4 metrics evaluated on each golden in one pass!

print("Step Efficiency Metric:")
print(f"  Threshold: 0.5 (journey quality, not output quality)")
print(f"  Checks: redundant tool calls, unnecessary reasoning steps")
print(f"  Why 0.5 not 0.7: output may be correct even if path isn't perfectly optimal")
print(f"  Cost impact: unnecessary tool calls = wasted API cost + latency")

# COMMAND ----------

# DBTITLE 1,Custom Metrics with G-Eval
# MAGIC %md
# MAGIC ## 8. Custom Metrics with G-Eval
# MAGIC
# MAGIC When DeepEval doesn't have a built-in metric for your quality dimension, you can build a **custom metric** using the `GEval` class.
# MAGIC
# MAGIC ### Example: Factual Correctness
# MAGIC DeepEval doesn't ship a ready-made "factual correctness" metric. We need one to verify the agent's output contains the *right facts* (not just relevant or complete).
# MAGIC
# MAGIC ### How G-Eval Works
# MAGIC 1. **`name`**: A label for your custom metric (e.g., "correctness")
# MAGIC 2. **`criteria`**: A natural-language description of what to validate. The judge LLM reads this to understand the check.
# MAGIC 3. **`model`**: The judge LLM that provides intelligence.
# MAGIC 4. **`threshold`**: Pass/fail cutoff.
# MAGIC 5. **`evaluation_params`**: Which fields from the golden to evaluate (`input`, `expected_output`, `actual_output`).
# MAGIC
# MAGIC ### Singleton vs Multi-turn Params
# MAGIC * **`LLMTestCaseParams`** (singleton): For single-turn agents — you provide `input`, `expected_output`, `actual_output`.
# MAGIC * **`TurnParams`** (multi-turn): For chatbots — you provide `role`, `content` from each turn.
# MAGIC
# MAGIC > The criteria is where the magic happens. You can write ANY requirement: "response should be minimum 600 words", "output must not contain promotional language", etc.

# COMMAND ----------

# DBTITLE 1,G-Eval Factual Correctness
# ---- CUSTOM METRIC: FACTUAL CORRECTNESS WITH G-EVAL ----
# DeepEval has no built-in "factual correctness" metric.
# We build one using GEval to check if actual_output conveys the same facts as expected_output.

# Uncomment with valid OpenAI API key:
# correctness_metric = GEval(
#     name="factual_correctness",
#     criteria=(
#         "Determine whether the actual output conveys the same factual information "
#         "as the expected output. Minor wording differences are acceptable, "
#         "but wrong facts are not acceptable."
#     ),
#     model='gpt-4o',
#     threshold=0.8,  # stricter for factuality — 80% required
#     evaluation_params=[
#         LLMTestCaseParams.INPUT,         # the user question
#         LLMTestCaseParams.EXPECTED_OUTPUT,  # ground truth
#         LLMTestCaseParams.ACTUAL_OUTPUT,    # what the agent returned
#     ],
# )

# This golden needs expected_output because we're checking factuality
golden_correctness = Golden(
    input="Where is my order 1042?",
    expected_output="Order 1042 is shipped and will arrive by May 13, 2026.",
)

dataset_correctness = EvaluationDataset(goldens=[golden_correctness])

# for golden in dataset_correctness.evals_iterator(metrics=[correctness_metric]):
#     response = support_agent_traced(golden.input)
#     trace = get_current_trace()
#     if golden.expected_output:
#         update_current_trace(expected_output=golden.expected_output)
#     # Trace now has: actual_output (runtime) + expected_output (from golden)
#     # G-Eval compares them for factual correctness.

print("G-Eval Custom Metric: factual_correctness")
print(f"  Criteria: compare actual vs expected factual information")
print(f"  Threshold: 0.8 (stricter for factuality)")
print(f"  Evaluation params: INPUT, EXPECTED_OUTPUT, ACTUAL_OUTPUT (singleton)")
print(f"  Expected output required in golden: YES")
print("\nYou can create ANY custom metric by changing the 'criteria' text.")
print("Example: 'The response should be minimum 600 words.' → custom word-count metric")

# COMMAND ----------

# DBTITLE 1,Multi-Turn Chatbot Testing
# MAGIC %md
# MAGIC ## 9. Multi-Turn Chatbot Testing
# MAGIC
# MAGIC Chatbots involve **conversations** — the user asks a question, the bot responds, the user asks a follow-up, etc. We need to test:
# MAGIC
# MAGIC * **Context retention**: Does the bot remember earlier turns?
# MAGIC * **Turn relevancy**: Is each response relevant to the *immediate* question?
# MAGIC * **Knowledge retention**: Does the bot retain factual information mentioned earlier?
# MAGIC * **Conversation completeness**: Did the conversation satisfy the user's needs end-to-end?
# MAGIC
# MAGIC ### Key Differences from Single-Turn Agents
# MAGIC | Aspect | Single-Turn Agent | Multi-Turn Chatbot |
# MAGIC |--------|-----------------|-------------------|
# MAGIC | Test case class | `LLMTestCase` | `ConversationalTestCase` |
# MAGIC | Data structure | `input` + `actual_output` | `turns` list (alternating user/assistant `Turn` objects) |
# MAGIC | Key method | `evaluate()` | `evaluate()` with `ConversationalTestCase` |
# MAGIC | Tracing complexity | Simple | Complex (context per turn) → **black-box recommended** |
# MAGIC
# MAGIC ### Turns
# MAGIC Each `Turn` has a `role` (`"user"` or `"assistant"`) and `content`. A 4-question conversation = 8 turns (4 user + 4 assistant).

# COMMAND ----------

# DBTITLE 1,Building Chatbot Turns
# ---- MULTI-TURN CHATBOT: BUILDING TURNS ----
# Step 1: Define the questions the user will ask (simulating a real conversation)
chatbot_questions = [
    "Hi, I placed an order last week. The order ID is 1042.",
    "Is it going to arrive on time?",
    "What was the ETA you just mentioned?",
    "Can I upgrade to express shipping?",
]

# Step 2: Simulate chatbot responses (in production, you'd call the real chatbot method)
chatbot_responses = [
    "Your order 1042 has been shipped and is expected to arrive by May 13, 2026.",
    "Yes, your order is expected to arrive as scheduled. You'll be notified of any changes.",
    "The ETA I mentioned is May 13, 2026.",
    "Unfortunately, once an order has been shipped, the shipping method cannot be changed.",
]

# Step 3: Build turns list — alternating user and assistant turns
turns = []
for question, response in zip(chatbot_questions, chatbot_responses):
    # User turn
    turns.append(Turn(role="user", content=question))
    # Assistant turn
    turns.append(Turn(role="assistant", content=response))

print(f"Total turns: {len(turns)} ({len(turns)//2} user + {len(turns)//2} assistant)")
for i, turn in enumerate(turns):
    role_label = "👤 USER" if turn.role == "user" else "🤖 BOT"
    print(f"  Turn {i+1} [{role_label}]: {turn.content[:60]}...")

# Step 4: Create ConversationalTestCase
conversational_test_case = ConversationalTestCase(turns=turns)
print(f"\nConversationalTestCase created with {len(conversational_test_case.turns)} turns")

# COMMAND ----------

# DBTITLE 1,Chatbot Metrics Evaluation
# ---- CHATBOT METRICS: TURN RELEVANCY + KNOWLEDGE RETENTION + COMPLETENESS ----
# All three are conversational metrics that work with ConversationalTestCase.

from deepeval.metrics import (
    TurnRelevancyMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
)

# Uncomment with valid OpenAI API key:
# turn_relevancy = TurnRelevancyMetric(threshold=0.7)
# knowledge_retention = KnowledgeRetentionMetric(threshold=0.7)
# conversation_completeness = ConversationCompletenessMetric(threshold=0.7)

# Evaluate all three metrics on the conversational test case:
# evaluate(
#     test_cases=[conversational_test_case],
#     metrics=[turn_relevancy, knowledge_retention, conversation_completeness],
# )

print("Chatbot Metrics:")
print(f"  1. TurnRelevancyMetric(0.7) — each response relevant to immediate question")
print(f"  2. KnowledgeRetentionMetric(0.7) — bot remembers facts from earlier turns")
print(f"  3. ConversationCompletenessMetric(0.7) — user needs satisfied end-to-end")
print(f"\nNote: Some conversational metrics use DeepEval's built-in LLM by default.")
print(f"For enterprise data privacy, pass model='gpt-4o' to keep data in your own API.")

# COMMAND ----------

# DBTITLE 1,Conversational G-Eval Concept
# MAGIC %md
# MAGIC ## 10. Conversational G-Eval: Custom Chatbot Metric
# MAGIC
# MAGIC Same G-Eval concept as single-turn, but for multi-turn conversations.
# MAGIC
# MAGIC ### Key Differences
# MAGIC * Use `TurnParams` instead of `LLMTestCaseParams` — the evaluation parameters are `role` and `content` from turns.
# MAGIC * Use `multi_turn_params` instead of `evaluation_params`.
# MAGIC * The criteria reads the entire conversation history and evaluates across all turns.
# MAGIC
# MAGIC ### Example Custom Metric
# MAGIC "Did the chatbot fully resolve the customer issue? It should use tools when needed and provide accurate answers based on the tools' response only (no hallucination)."

# COMMAND ----------

# DBTITLE 1,Conversational G-Eval Implementation
# ---- CONVERSATIONAL G-EVAL: CUSTOM CHATBOT METRIC ----
# Same GEval class, but configured for multi-turn conversations.

# Uncomment with valid OpenAI API key:
# conversational_g_eval = GEval(
#     name="issue_resolution_correctness",
#     criteria=(
#         "Did the chatbot fully resolve the customer issue? "
#         "It should use tools when needed and provide accurate answers "
#         "based on the tools' response only (no hallucination)."
#     ),
#     model='gpt-4o',
#     threshold=0.8,
#     # Multi-turn params: evaluate role + content from each turn
#     multi_turn_params=[
#         TurnParams.ROLE,
#         TurnParams.CONTENT,
#     ],
# )

# Evaluate with the conversational test case:
# evaluate(
#     test_cases=[conversational_test_case],
#     metrics=[turn_relevancy, knowledge_retention, conversation_completeness, conversational_g_eval],
# )

print("Conversational G-Eval: issue_resolution_correctness")
print(f"  Criteria: did chatbot fully resolve issue + use tools + no hallucination")
print(f"  Multi-turn params: TurnParams.ROLE, TurnParams.CONTENT")
print(f"  Threshold: 0.8")
print(f"  Reads entire conversation history across all turns")

# COMMAND ----------

# DBTITLE 1,RAG Architecture Overview
# MAGIC %md
# MAGIC ## 11. RAG Architecture Overview
# MAGIC
# MAGIC **RAG = Retrieval + Augmentation + Generation**
# MAGIC
# MAGIC RAG shines when you need to search across **large unstructured document sets** (thousands of PDFs, policy documents, etc.) and the answer requires **semantic search** (matching by meaning, not just keywords).
# MAGIC
# MAGIC ### RAG vs. Traditional Agent with Tools
# MAGIC | Aspect | Traditional Agent (Tools) | RAG Agent |
# MAGIC |--------|--------------------------|----------|
# MAGIC | Data retrieval | Direct SQL query, known table/schema | Semantic search across vector DB |
# MAGIC | When to use | You know exactly where data sits (1-2 fixed tables) | Large unstructured document corpus |
# MAGIC | Search type | **Syntactic** (keyword matching) | **Semantic** (meaning-based matching) |
# MAGIC | Example | `SELECT * FROM orders WHERE id = 1042` | Search 10,000 policy docs for "flood damage coverage" |
# MAGIC | Context window | Minimal — only needed rows | Minimal — only top-k relevant doc chunks |
# MAGIC
# MAGIC ### RAG Pipeline
# MAGIC 1. **Retrieval**: User query → converted to vector → matched against vector DB → top-k relevant documents returned
# MAGIC 2. **Augmentation**: Original question + retrieved documents (context) + system prompt sent to LLM
# MAGIC 3. **Generation**: LLM reads the context, understands the question, and generates an answer from the retrieved documents
# MAGIC
# MAGIC ### Why RAG Instead of Loading All Docs into LLM?
# MAGIC * **Context window pressure**: Loading 10,000 documents into the LLM would consume massive tokens.
# MAGIC * **Cost**: Each token costs money — RAG sends only the 3-5 most relevant chunks.
# MAGIC * **Semantic gap**: User asks about "flood damage" but documents use "water damage" / "natural disaster" / "act of God" — RAG bridges this gap via semantic similarity.

# COMMAND ----------

# DBTITLE 1,RAG Metrics Concept
# MAGIC %md
# MAGIC ## 12. RAG-Specific Metrics
# MAGIC
# MAGIC Three metrics are **exclusive to RAG** — they evaluate the retrieval architecture itself, not just the final response.
# MAGIC
# MAGIC ### Contextual Precision
# MAGIC > Are the relevant documents ranked **higher** than irrelevant ones?
# MAGIC * Out of 5 retrieved documents, is the answer in the **top 1-2**? If the answer is buried in document #4, precision is low.
# MAGIC * Why it matters: if the answer is in the top document, the LLM reads less → faster + cheaper.
# MAGIC * Needs: `retrieval_context` (from tracing), `expected_output` (ground truth).
# MAGIC
# MAGIC ### Contextual Recall
# MAGIC > How much **noise** (irrelevant documents) is in the retrieval context?
# MAGIC * If 10 documents are retrieved but only 2 are relevant → 80% noise.
# MAGIC * High noise = wasteful retrieval. All retrieved docs should contribute to the answer.
# MAGIC * Needs: same as precision — `retrieval_context` + `expected_output`.
# MAGIC
# MAGIC ### Faithfulness
# MAGIC > Is the response **grounded in** the retrieved context, or did the LLM hallucinate?
# MAGIC * The LLM should generate answers **only** from the retrieved documents.
# MAGIC * If the LLM ignores the company's policy docs and generates a generic refund policy → faithfulness FAIL.
# MAGIC * Needs: `actual_output` + `retrieval_context` (both captured by tracing automatically).
# MAGIC
# MAGIC ### General Metrics Also Apply to RAG
# MAGIC Since a RAG agent is still an agent, you can also apply: `AnswerRelevancyMetric`, `TaskCompletionMetric`, `PromptAlignmentMetric`, and custom `GEval` metrics. The three above are the ones **unique** to RAG.

# COMMAND ----------

# DBTITLE 1,RAG Metrics Implementation
# ---- RAG AGENT MOCK + METRICS ----
# Simulate a RAG agent that searches policy documents in a vector DB.

POLICY_DOCS = [
    "Electronics refund policy: Items can be returned within 15 days of delivery if unopened. Refunds take 5-7 business days.",
    "Clothing refund policy: Items can be returned within 30 days of delivery. Refunds take 3-5 business days.",
    "Standard shipping: 5-7 business days, free for orders over $50.",
    "Express shipping: 2-3 business days, costs $15.",
    "Overnight shipping: Next business day delivery, costs $35.",
    "Furniture refund policy: Items can be returned within 60 days. Refunds take 10-14 business days.",
    "Food items: Non-returnable due to health regulations.",
]

def rag_support_agent(query: str) -> str:
    """Simulates RAG: semantic search → retrieve top docs → LLM generates answer."""
    query_lower = query.lower()
    # Simulate semantic search: match by meaning (simplified)
    retrieved = []
    for doc in POLICY_DOCS:
        if any(word in doc.lower() for word in query_lower.split()):
            retrieved.append(doc)
    if not retrieved:
        retrieved = POLICY_DOCS[:2]  # fallback
    # Simulate LLM generating answer from retrieved context
    return retrieved[0] if retrieved else "No information found."

# Step 1: Define RAG-specific metrics (uncomment with valid OpenAI API key)
# context_precision = ContextualPrecisionMetric(threshold=0.7, model='gpt-4o')
# context_recall = ContextualRecallMetric(threshold=0.7, model='gpt-4o')
# faithfulness = FaithfulnessMetric(threshold=0.7, model='gpt-4o')

# Step 2: Create goldens with expected_output (ground truth needed for precision/recall)
rag_goldens = [
    Golden(
        input="What is the return policy for electronics?",
        expected_output="Electronics can be returned within 15 days of delivery if unopened. Refunds take 5-7 business days.",
    ),
    Golden(
        input="How long does express shipping take and what does it cost?",
        expected_output="Express shipping takes 2-3 business days and costs $15.",
    ),
]

rag_dataset = EvaluationDataset(goldens=rag_goldens)

# Step 3: Evaluate with tracing (white-box) — tracing captures retrieval_context automatically
# for golden in rag_dataset.evals_iterator(
#     metrics=[context_precision, context_recall, faithfulness, answer_relevancy_metric]
# ):
#     response = rag_support_agent(golden.input)  # call with @observe tracing
#     trace = get_current_trace()
#     if golden.expected_output:
#         update_current_trace(expected_output=golden.expected_output)
#     # Trace now has:
#     #   - retrieval_context (captured by tracing from the vector search)
#     #   - expected_output (from golden → for precision/recall)
#     #   - actual_output (captured by tracing → for faithfulness + answer relevancy)

print("RAG Metrics Evaluation:")
print(f"  1. ContextualPrecisionMetric(0.7) — relevant docs ranked higher than irrelevant")
print(f"  2. ContextualRecallMetric(0.7) — low noise (all retrieved docs are relevant)")
print(f"  3. FaithfulnessMetric(0.7) — response grounded in retrieved context (no hallucination)")
print(f"  + AnswerRelevancyMetric(0.7) — also applies (RAG agent is still an agent)")
print(f"\nAll 3 RAG metrics need retrieval_context — captured automatically by tracing.")
print(f"Precision & Recall also need expected_output — provided in goldens.")

# COMMAND ----------

# DBTITLE 1,Synthetic Data Generation Concept
# MAGIC %md
# MAGIC ## 13. Synthetic Data Generation
# MAGIC
# MAGIC When you don't have enough representative test cases (goldens), DeepEval's `Synthesizer` can **generate goldens from your domain documents**.
# MAGIC
# MAGIC ### How It Works
# MAGIC 1. Provide document paths (PDFs, text files, Word docs, Markdown).
# MAGIC 2. The synthesizer reads the documents using an LLM.
# MAGIC 3. It generates inputs (questions) and optionally expected outputs (answers from the docs).
# MAGIC 4. The generated goldens can be fed directly into `EvaluationDataset`.
# MAGIC
# MAGIC ### Important Caveat
# MAGIC > Synthetic data should **complement, not replace** real test data. Always review generated goldens — the LLM may produce unrealistic questions. Get real user questions from **production logs** whenever possible.
# MAGIC
# MAGIC ### Use Cases
# MAGIC * Bootstrapping when you have no test data yet
# MAGIC * Generating additional scenarios to complement hand-written goldens
# MAGIC * Exploring edge cases you might not think of

# COMMAND ----------

# DBTITLE 1,Synthetic Data Generation
# ---- SYNTHETIC DATA GENERATION ----
# Generate goldens from domain documents using DeepEval's Synthesizer.

# Step 1: Define your domain documents as text (in production, these would be PDF/Word paths)
sample_policies = """ShopEasy Customer Support Policies:

1. Order Status: Customers can check order status using their order ID. Orders are typically shipped within 2 business days.

2. Refund Policy - Electronics: Can be returned within 15 days of delivery if unopened. Refunds take 5-7 business days.

3. Refund Policy - Clothing: Can be returned within 30 days of delivery. Refunds take 3-5 business days.

4. Shipping Options: Standard (5-7 days, free over $50), Express (2-3 days, $15), Overnight (next day, $35).

5. Damaged Items: If your item arrives damaged, contact support within 48 hours for a replacement.
"""
print("✓ Sample policy document defined (inline text)")

# Step 2: Create synthesizer (requires OpenAI API key to instantiate)
# Uncomment when you have a valid API key configured:
# synthesizer = Synthesizer(
#     model='gpt-4o',  # LLM used to read documents and generate questions/answers
# )
print("✓ Synthesizer class available (requires OpenAI API key to instantiate)")

# In production, use document_paths=[...] pointing to real PDF/txt files:
# synthesizer.generate_goldens_from_docs(
#     document_paths=['/path/to/policies.pdf', '/path/to/refund_policy.docx'],
#     include_expected_output=True,  # also generate expected answers from the docs
#     max_goldens_per_context=2,    # generate 2 test cases per document chunk
# )
#
# # The generated goldens can be used directly as an EvaluationDataset
# synthetic_dataset = EvaluationDataset(goldens=synthesizer.generated_goldens)
# print(f"Generated {len(synthetic_dataset)} synthetic goldens")
# for g in synthetic_dataset:
#     print(f"  Input: {g.input}")
#     if g.expected_output:
#         print(f"  Expected: {g.expected_output}")

print("\nSynthesizer Configuration:")
print(f"  Model: gpt-4o (reads docs and generates questions/answers)")
print(f"  Document: inline text (production: use document_paths=['/path/to/policies.pdf']")
print(f"  include_expected_output: True (generates both questions AND answers)")
print(f"  max_goldens_per_context: 2")
print(f"\n⚠️  Always review synthetic goldens — they complement, not replace, real data.")
print(f"⚠️  Get real questions from production logs for the most representative test coverage.")

# COMMAND ----------

# DBTITLE 1,Safety Metrics Concept
# MAGIC %md
# MAGIC ## 14. Safety Metrics
# MAGIC
# MAGIC Safety metrics ensure your AI system's responses are **safe, unbiased, and privacy-preserving**.
# MAGIC
# MAGIC ### Three Key Safety Metrics
# MAGIC
# MAGIC | Metric | What It Checks | Direction | Threshold |
# MAGIC |--------|--------------|----------|----------|
# MAGIC | **BiasMetric** | Gender, racial, or political bias in output | **Lower = better** (reverse scoring) | Score must be **< 0.5** to pass |
# MAGIC | **ToxicityMetric** | Toxic, harmful, or abusive language | **Lower = better** (reverse scoring) | Score must be **< 0.5** to pass |
# MAGIC | **PIIViolationMetric** | Personally identifiable information leakage (SSN, phone, address) | **Higher = better** | Score must be **> 0.5** to pass |
# MAGIC
# MAGIC ### Important: Reverse Scoring for Bias & Toxicity
# MAGIC For bias and toxicity, a **higher score = worse**. A score of 0.9 means highly biased/toxic. Your test **passes** when the score is **below** the threshold. This is the opposite of quality metrics where higher = better.
# MAGIC
# MAGIC ### When to Use
# MAGIC Include safety metrics in at least 10% of your overall test suite to catch:
# MAGIC * Responses that leak customer PII (SSN, addresses, phone numbers)
# MAGIC * Biased or toxic responses that could harm your brand
# MAGIC * Guardrail violations where the agent should refuse to share sensitive info

# COMMAND ----------

# DBTITLE 1,Safety Metrics Implementation
# ---- SAFETY METRICS: BIAS + TOXICITY + PII ----
# These metrics check if the AI output is safe, unbiased, and privacy-preserving.
# NOTE: All metrics require an OpenAI API key to instantiate. Uncomment when configured.

# 1. Bias Metric — checks for gender, racial, political bias
#    REVERSE scoring: lower score = less bias = PASS
#    Test fails if score > threshold (0.5)
bias_metric = None  # BiasMetric(threshold=0.5)  # pass if score < 0.5 (low bias)

# 2. Toxicity Metric — checks for toxic, harmful, abusive language
#    REVERSE scoring: lower score = less toxic = PASS
toxicity_metric = None  # ToxicityMetric(threshold=0.5)  # pass if score < 0.5 (low toxicity)

# 3. PII Violation Metric — checks for leaked personally identifiable information
#    NORMAL scoring: higher score = less leakage = PASS
#    Note: PIIMetric may not be available in all DeepEval versions
pii_metric = None
if PIIMetric is not None:
    pii_metric = PIIMetric(
        threshold=0.5,  # pass if score > 0.5 (minimal PII leakage)
    )
else:
    print("⚠️ PII metric not available in this DeepEval version — skipping")

# Using with synthesized goldens + tracing:
# for golden in synthetic_dataset.evals_iterator(
#     metrics=[bias_metric, toxicity_metric, pii_metric]
# ):
#     support_agent_traced(golden.input)
#     # Tracing captures actual_output automatically
#     # No expected values needed — these metrics only check actual_output

print("Safety Metrics:")
print(f"  1. BiasMetric(0.5) — REVERSE: pass if score < 0.5 (low bias)")
print(f"  2. ToxicityMetric(0.5) — REVERSE: pass if score < 0.5 (low toxicity)")
print(f"  3. PIIViolationMetric(0.5) — NORMAL: pass if score > 0.5 (no PII leak)")
print(f"\nAll three only need actual_output (captured by tracing).")
print(f"No expected output or expected tools required.")
print(f"\n⚠️  Model is optional for safety metrics — defaults to DeepEval's built-in LLM.")
print(f"    For enterprise data privacy, pass model='gpt-4o' with your own API key.")

# COMMAND ----------

# DBTITLE 1,Summary & Best Practices
# MAGIC %md
# MAGIC ## 15. Summary & Best Practices
# MAGIC
# MAGIC ### Testing Strategy Decision Framework
# MAGIC
# MAGIC ```
# MAGIC Do you have access to the dev codebase?
# MAGIC ├── YES → White-box testing (tracing + evals_iterator)
# MAGIC │   └── Need internal component data (tools, steps, retrieval context)?
# MAGIC │       ├── YES → Use @observe + DeepEvalCallbackHandler + update_current_trace
# MAGIC │       └── NO  → Black-box also works (LLMTestCase + evaluate)
# MAGIC └── NO (third-party agent) → Black-box only
# MAGIC     └── Collect input/output/tools from production logs → hard-code in goldens
# MAGIC ```
# MAGIC
# MAGIC ### AI System Type → Recommended Metrics
# MAGIC
# MAGIC | System Type | Key Metrics |
# MAGIC |-------------|-------------|
# MAGIC | **AI Agent** (single-turn) | Task Completion, Answer Relevancy, Tool Correctness, Prompt Alignment, Step Efficiency |
# MAGIC | **Chatbot** (multi-turn) | Turn Relevancy, Knowledge Retention, Conversation Completeness (+ conversational G-Eval) |
# MAGIC | **RAG Agent** | Contextual Precision, Contextual Recall, Faithfulness (+ all general agent metrics) |
# MAGIC | **Any system** | Bias, Toxicity, PII Violation (safety — at least 10% of test suite) |
# MAGIC
# MAGIC ### Key Best Practices
# MAGIC 1. **Use a different LLM vendor for judging** than the one used to build the agent (avoid self-grading bias).
# MAGIC 2. **Thresholds are project-specific**: Medical domain → 0.9. Real estate → 0.5. Adjust per domain risk.
# MAGIC 3. **Quality metrics**: threshold 0.7+. **Safety metrics (bias/toxicity)**: score must be *below* 0.5.
# MAGIC 4. **G-Eval for any missing dimension**: If DeepEval doesn't have a built-in metric, write a `criteria` and create a custom one.
# MAGIC 5. **Synthetic data complements, not replaces** real test data — always review generated goldens.
# MAGIC 6. **Get real questions from production logs** for the most representative test coverage.
# MAGIC 7. **Integrate with Confident AI** (app.confident.ai) for enterprise dashboards and reporting.
# MAGIC 8. **Coordinate with your dev team** to add `DeepEvalCallbackHandler` in the agent's `invoke()` call for tracing support.
# MAGIC
# MAGIC ### Course Concepts Covered
# MAGIC ✅ Non-deterministic nature of AI → evals instead of assertions  
# MAGIC ✅ DeepEval metrics, goldens, judge LLM, thresholds  
# MAGIC ✅ Black-box (LLMTestCase + evaluate) and White-box (tracing + evals_iterator)  
# MAGIC ✅ Task Completion, Answer Relevancy, Tool Correctness, Prompt Alignment, Step Efficiency  
# MAGIC ✅ Custom metrics with G-Eval (single-turn and conversational)  
# MAGIC ✅ Multi-turn chatbot testing with ConversationalTestCase  
# MAGIC ✅ RAG-specific metrics: Contextual Precision, Contextual Recall, Faithfulness  
# MAGIC ✅ Synthetic data generation with Synthesizer  
# MAGIC ✅ Safety metrics: Bias, Toxicity, PII Violation