# Databricks notebook source
# DBTITLE 1,Setup
# MAGIC %md
# MAGIC # Understanding Agent.py Code - Simple Examples
# MAGIC
# MAGIC Yeh notebook agent.py ki complex code ko simple examples ke saath explain karta hai.
# MAGIC
# MAGIC ## Topics:
# MAGIC 1. **StateGraph** - Agent ka basic structure
# MAGIC 2. **Nodes** - Processing steps
# MAGIC 3. **Edges** - Flow control
# MAGIC 4. **Conditional Edges** - Decision making
# MAGIC 5. **Tools** - External actions
# MAGIC 6. **Stream vs Invoke** - Different execution modes

# COMMAND ----------

# DBTITLE 1,Install Packages
# MAGIC %pip install -q langgraph langchain-core
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Example 1: Basic StateGraph
# MAGIC %md
# MAGIC ## Example 1: Sabse Simple StateGraph
# MAGIC
# MAGIC Pehle dekhte hain ki **StateGraph** kya hai - yeh ek workflow/pipeline hai jisme data ek step se dusre step mein flow hota hai.

# COMMAND ----------

# DBTITLE 1,Simple State Example
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Step 1: State define karo - yeh data structure hai jo har node ke beech pass hota hai
class SimpleState(TypedDict):
    message: str
    count: int

# Step 2: Node functions - yeh actual processing steps hain
def step1(state: SimpleState):
    print("Step 1 running...")
    return {
        "message": "Hello from step 1",
        "count": state["count"] + 1
    }

def step2(state: SimpleState):
    print("Step 2 running...")
    return {
        "message": f"Step 2 received: {state['message']}",
        "count": state["count"] + 1
    }

# Step 3: Graph banao
workflow = StateGraph(SimpleState)

# Step 4: Nodes add karo
workflow.add_node("step1", step1)
workflow.add_node("step2", step2)

# Step 5: Flow define karo (edges)
workflow.set_entry_point("step1")  # Yahan se shuru hoga
workflow.add_edge("step1", "step2")  # step1 -> step2
workflow.add_edge("step2", END)       # step2 -> END

# Step 6: Compile karo
app = workflow.compile()

print("✓ Simple workflow ready!")
print("\nFlow: START → step1 → step2 → END")

# COMMAND ----------

# DBTITLE 1,Run Simple Workflow
# Invoke vs Stream ka difference

print("=" * 50)
print("METHOD 1: invoke() - Sirf final result")
print("=" * 50)

result = app.invoke({"message": "Start", "count": 0})
print(f"\nFinal Result: {result}")

print("\n" + "=" * 50)
print("METHOD 2: stream() - Har step ka output")
print("=" * 50)

for step_output in app.stream({"message": "Start", "count": 0}):
    for node_name, node_result in step_output.items():
        print(f"\nNode: {node_name}")
        print(f"Output: {node_result}")

# COMMAND ----------

# DBTITLE 1,Example 2: Conditional Routing
# MAGIC %md
# MAGIC ## Example 2: Conditional Edges - Decision Making
# MAGIC
# MAGIC Agent.py mein **should_continue** function yeh decide karta hai ki:
# MAGIC * Tool call karna hai? → "continue" (tools node pe jao)
# MAGIC * Response ready hai? → "end" (END pe jao)
# MAGIC
# MAGIC Yeh simple example mein dekhte hain:

# COMMAND ----------

# DBTITLE 1,Conditional Routing Example
from langgraph.graph import StateGraph, END
from typing import TypedDict

class CounterState(TypedDict):
    count: int
    message: str

def increment(state: CounterState):
    """Count badhao"""
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "message": f"Count is now {new_count}"
    }

def check_limit(state: CounterState):
    """Decision function - kahan jaana hai?"""
    if state["count"] < 3:
        return "continue"  # Abhi aur increment karo
    else:
        return "stop"      # Bas, ruk jao

def finish(state: CounterState):
    """Final step"""
    return {
        "count": state["count"],
        "message": f"Done! Final count: {state['count']}"
    }

# Graph banao
workflow = StateGraph(CounterState)
workflow.add_node("increment", increment)
workflow.add_node("finish", finish)

# Entry point
workflow.set_entry_point("increment")

# CONDITIONAL EDGE - yeh agent.py ki should_continue jaisa hai
workflow.add_conditional_edges(
    "increment",           # Yeh node se
    check_limit,          # Yeh function decide karega
    {
        "continue": "increment",  # Agar continue, to wapas increment pe jao (loop!)
        "stop": "finish"         # Agar stop, to finish pe jao
    }
)

workflow.add_edge("finish", END)

app = workflow.compile()
print("✓ Conditional workflow ready!")
print("\nFlow: increment → [check] → increment (loop) → finish → END")

# COMMAND ----------

# DBTITLE 1,Run Conditional Workflow
print("Conditional workflow running...\n")
print("Yeh loop chalega jab tak count < 3\n")

for output in app.stream({"count": 0, "message": "Start"}):
    for node, result in output.items():
        print(f"Node '{node}': {result['message']}")

# COMMAND ----------

# DBTITLE 1,Example 3: Agent with Tools
# MAGIC %md
# MAGIC ## Example 3: Tool-Calling Agent Pattern
# MAGIC
# MAGIC Agent.py mein yeh pattern hai:
# MAGIC 1. **agent node**: LLM se baat karo, decide karo tool chahiye ya nahi
# MAGIC 2. **tools node**: Tool execute karo
# MAGIC 3. **Conditional edge**: Wapas agent pe jao ya END pe?
# MAGIC
# MAGIC Yeh simplified version hai:

# COMMAND ----------

# DBTITLE 1,Simple Tool Agent
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: list
    need_tool: bool

# Fake tool - calculator
def calculator_tool(expression: str):
    """Simple calculator tool"""
    try:
        result = eval(expression)  # Real agent mein safe tool use karo!
        return f"Result: {result}"
    except:
        return "Error in calculation"

def agent_node(state: AgentState):
    """Yeh LLM jaisa decide karta hai - tool chahiye ya nahi"""
    last_message = state["messages"][-1]
    
    # Simple logic: agar math problem hai, tool chahiye
    if any(op in last_message for op in ['+', '-', '*', '/', '**']):
        return {
            "messages": state["messages"] + ["Agent: I need calculator tool"],
            "need_tool": True
        }
    else:
        return {
            "messages": state["messages"] + ["Agent: I can answer directly"],
            "need_tool": False
        }

def tools_node(state: AgentState):
    """Tools execute karo"""
    last_message = state["messages"][-2]  # Original user message
    
    # Extract expression aur tool call karo
    for word in last_message.split():
        if any(op in word for op in ['+', '-', '*', '/']):
            result = calculator_tool(word)
            return {
                "messages": state["messages"] + [f"Tool result: {result}"],
                "need_tool": False
            }
    
    return {
        "messages": state["messages"] + ["Tool: No calculation found"],
        "need_tool": False
    }

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Yeh agent.py ka should_continue jaisa hai"""
    if state["need_tool"]:
        return "continue"  # Tools pe jao
    else:
        return "end"       # Bas ho gaya

# Graph banao
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.set_entry_point("agent")

# YAHI AGENT.PY KA MAIN PATTERN HAI!
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # Tool chahiye
        "end": END           # Direct answer
    }
)

workflow.add_edge("tools", "agent")  # Tool ke baad wapas agent pe jao

app = workflow.compile()
print("✓ Tool-calling agent ready!")
print("\nFlow: agent → [decision] → tools → agent → END")

# COMMAND ----------

# DBTITLE 1,Test Tool Agent
print("Test 1: Simple question (no tool needed)")
print("=" * 50)
for output in app.stream({"messages": ["Hello"], "need_tool": False}):
    for node, result in output.items():
        print(f"\n{node}: {result['messages'][-1]}")

print("\n\nTest 2: Math question (tool needed)")
print("=" * 50)
for output in app.stream({"messages": ["Calculate 5+3"], "need_tool": False}):
    for node, result in output.items():
        print(f"\n{node}: {result['messages'][-1]}")

# COMMAND ----------

# DBTITLE 1,Understanding agent.py
# MAGIC %md
# MAGIC ## Agent.py Code Breakdown
# MAGIC
# MAGIC Ab original agent.py ko samajhte hain:
# MAGIC
# MAGIC ### Main Components:
# MAGIC
# MAGIC **1. AgentState (Line 98-101)**
# MAGIC ```python
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC ```
# MAGIC * Yeh conversation history store karta hai
# MAGIC * `add_messages` automatically messages ko merge karta hai
# MAGIC
# MAGIC **2. create_tool_calling_agent (Line 104-159)**
# MAGIC * **model.bind_tools(tools)** - LLM ko batao ki kaunse tools available hain
# MAGIC * **should_continue** - Tool call hai ya nahi, yeh check karo
# MAGIC * **workflow.add_conditional_edges** - Decision making logic
# MAGIC
# MAGIC **3. LangGraphResponsesAgent (Line 162-262)**
# MAGIC * Yeh LangGraph agent ko Databricks ke ResponsesAgent format mein wrap karta hai
# MAGIC * **predict()** - Ek baar mein complete response
# MAGIC * **predict_stream()** - Token-by-token streaming response
# MAGIC
# MAGIC ### Key Pattern:
# MAGIC ```
# MAGIC User message → agent (LLM thinks) → [tool needed?]
# MAGIC                                          ↓ YES          ↓ NO
# MAGIC                                     tools (execute) → END
# MAGIC                                          ↓
# MAGIC                                     agent (process result) → END
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Visualize agent.py Flow
print("AGENT.PY FLOW DIAGRAM")
print("=" * 60)
print("""
    START
      ↓
   [agent node]
   - LLM thinks
   - Checks if tool call needed
      ↓
   [should_continue?]
      ↓                    ↓
   tool_calls exist    no tool_calls
      ↓                    ↓
   [tools node]         [END]
   - Execute tool
   - Get result
      ↓
   [agent node] (loop back)
   - Process tool result
   - Generate final answer
      ↓
   [END]

""")

print("\nSTREAM MODE:")
print("* stream_mode=['updates'] - Har node complete hone pe output")
print("* stream_mode=['messages'] - Har token generate hone pe output")
print("\nAgent.py dono modes use karta hai streaming response ke liye!")