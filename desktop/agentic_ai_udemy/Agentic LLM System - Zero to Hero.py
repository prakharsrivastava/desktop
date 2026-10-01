# Databricks notebook source
from dataclasses import dataclass


class Student:
    name: str
    age: int

    def __init__(self, name, age):
        self.name = name
        self.age = age

s1 = Student("Rahul", 20)
s2 = Student("Rahul", 20)

print(s1 == s2)

# COMMAND ----------

class Student:
    name: str
    age: int

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        if not isinstance(other, Student):
            return False
        return self.name == other.name and self.age == other.age

s1 = Student("Rahul", 20)
s2 = Student("Rahul", 20)

print(s1 == s2)

# COMMAND ----------

class User:
    def __init__(self, name: str):
        self.name = name.title()

u = User("rahul")
print(u.name)  # Rahul

# COMMAND ----------

from dataclasses import dataclass

@dataclass(slots=True)
class User:
    name: str
    age: int

# COMMAND ----------

class User:
    __slots__ = ['name', 'age']

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# COMMAND ----------

from dataclasses import asdict

user = User("rahul", 20)
user_dict = asdict(user)
user_dict

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class Rectangle:
    width: int
    height: int

    @property
    def area(self):
        return self.width * self.height
    
r = Rectangle(10, 20)
print(r.area)  # 200    

# COMMAND ----------

from dataclasses import dataclass

@dataclass(order=True)
class Student:
    marks: int
    name: str

students = [
    Student(90, "A"),
    Student(80, "B")
]

print(sorted(students))

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class Rectangle:
    width: int
    height: int

    def area(self):
        return self.width * self.height

r = Rectangle(10, 20)
print(r.area())  # 200

# COMMAND ----------



# COMMAND ----------

class Student:
    name: str
    age: int

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.__post_init__()  # Manually call post-init
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.name = self.name.title()

    def __eq__(self, other):
        if not isinstance(other, Student):
            return False
        return self.name == other.name and self.age == other.age

    def __repr__(self):
        return f"Student(name='{self.name}', age={self.age})"

s1 = Student("rahul", 20)
s2 = Student("rahul", 20)

print(s1 == s2)
print(repr(s1))
print(repr(s2))

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class User:
    name: str

    def __post_init__(self):
        self.name = self.name.title()

u = User("rahul")
print(u.name)  # Rahul

# COMMAND ----------

from dataclasses import dataclass, field

@dataclass
class Course:
    students: list = field(default_factory=list) # mutable list

c = Course()
print(c.students)
c.students.append("Rahul")
c.students.append("Rdahul")
print(c.students)

# COMMAND ----------

from dataclasses import dataclass, field

@dataclass(frozen=True)
class Course:
    x: int
    y: int
    students: list = field(default_factory=list) # mutable list

c = Course(1,2)
c.x=0
print(c.students)
c.students.append("Rahul")
c.students.append("Rdahul")
print(c.students)

# COMMAND ----------

# DBTITLE 1,📚 Table of Contents - Agentic Email Assistant
# MAGIC %md
# MAGIC # 🚀 Agentic Email Assistant - Zero to Hero
# MAGIC
# MAGIC ## Complete Interactive Tutorial: Email Triage System Samajhne Ka Roadmap
# MAGIC
# MAGIC **Project Context:** Ek health insurance company ke liye automated email assistant jo emails ko read kare, intent samjhe, aur appropriate queue mein route kare (ID Card request, Address Change, General Inquiry, etc.)
# MAGIC
# MAGIC **Tech Stack:** Python + FastAPI + LangChain + GPT-4o (via Horizon/Elevance) + Pydantic + SQLAlchemy + OAuth
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📖 Learning Path (10 Topics)
# MAGIC
# MAGIC ### 🏗️ **MUST-HAVE (BASE)** — Foundation
# MAGIC
# MAGIC 1. **Python Advanced** 🐍
# MAGIC    - async/await (concurrent email processing)
# MAGIC    - dataclasses (config objects)
# MAGIC    - type hints (code clarity)
# MAGIC    - decorators (logging, retry logic)
# MAGIC    - exceptions (error handling)
# MAGIC
# MAGIC 2. **Pydantic** ✅
# MAGIC    - BaseModel (all schemas inherit this)
# MAGIC    - Field validation (email format, HCID format)
# MAGIC    - Validators (custom business rules)
# MAGIC    - Response models
# MAGIC
# MAGIC 3. **FastAPI** ⚡
# MAGIC    - REST endpoints (/process-email, /get-status)
# MAGIC    - Request/Response models
# MAGIC    - JWT authentication
# MAGIC    - Dependency injection
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🤖 **GenAI/LLM (CORE)** — The Brain
# MAGIC
# MAGIC 4. **LLM Basics** 🧠
# MAGIC    - System vs User prompts
# MAGIC    - Temperature (creativity control)
# MAGIC    - Tokens (cost + context window)
# MAGIC    - **Structured JSON Output** (dil of the project!)
# MAGIC
# MAGIC 5. **LangChain Basics** 🔗
# MAGIC    - invoke() method
# MAGIC    - AIMessage format
# MAGIC    - Chat models
# MAGIC    - Horizon wrapper pattern
# MAGIC
# MAGIC 6. **OAuth Client-Credentials** 🔐
# MAGIC    - TokenManager class
# MAGIC    - Token lifecycle (get, refresh, cache)
# MAGIC    - API authentication flow
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🏛️ **AGENTIC/ARCHITECTURE** — The Design
# MAGIC
# MAGIC 7. **Agentic Concepts** 🤝
# MAGIC    - **Orchestrator** (main controller)
# MAGIC    - **Planner Agent** (decides what to do)
# MAGIC    - **Executor Agents** (ID Card, Address, Inquiry agents)
# MAGIC    - **Intent Classification** (email → intent)
# MAGIC    - **Entity Extraction** (HCID, address, etc.)
# MAGIC    - **Routing** (intent → PEGA queue)
# MAGIC    - **Human-in-the-Loop** (confidence threshold)
# MAGIC
# MAGIC 8. **Planner Pattern** 📋
# MAGIC    - PlanStep (data structure)
# MAGIC    - ExecutionPlan (list of steps)
# MAGIC    - Shared context (ctx object)
# MAGIC    - Routing decisions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔧 **SUPPORTING** — The Infrastructure
# MAGIC
# MAGIC 9. **State Management** 💾
# MAGIC    - SQLAlchemy basics
# MAGIC    - OracleDBConnector
# MAGIC    - Task state tracking
# MAGIC    - log_step pattern (observability)
# MAGIC
# MAGIC 10. **Domain Knowledge** 🏥
# MAGIC     - Health insurance terminology
# MAGIC     - HCID (Health Care ID)
# MAGIC     - PEGA queues (workflow system)
# MAGIC     - SOA mainframe integration
# MAGIC     - Intent types (ID Card / Address Change / Enquiry)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Approach
# MAGIC
# MAGIC Har topic mein:
# MAGIC - 📚 **Theory** (concepts in Hinglish)
# MAGIC - 💻 **Simple Example** (code with comments)
# MAGIC - 🎮 **Interactive Demo** (widgets where possible)
# MAGIC - 🔗 **Real-World Pattern** (email assistant mein kaise use hota hai)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏗️ Final Architecture
# MAGIC
# MAGIC ```
# MAGIC Email → FastAPI Endpoint → Planner Agent (Orchestrator)
# MAGIC                               ↓
# MAGIC                      Intent Classification (LLM)
# MAGIC                               ↓
# MAGIC                     ┌─────────┴─────────┐
# MAGIC                     ↓                   ↓
# MAGIC             ID Card Agent      Address Change Agent
# MAGIC                     ↓                   ↓
# MAGIC               PEGA Queue          PEGA Queue
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Prerequisites Check
# MAGIC
# MAGIC Yeh tutorial ke baad aap samajh paoge:
# MAGIC - ✅ Async email processing kaise hoti hai
# MAGIC - ✅ LLM se structured output kaise nikalta hai
# MAGIC - ✅ Planner agent routing kaise decide karta hai
# MAGIC - ✅ Multiple executor agents kaise coordinate karte hain
# MAGIC - ✅ OAuth tokens ka lifecycle
# MAGIC - ✅ FastAPI REST API design
# MAGIC - ✅ Pydantic validation patterns
# MAGIC - ✅ State management + logging
# MAGIC
# MAGIC **Interview Ready:** Yeh sab topics GenAI/LLM engineer interviews mein frequently aate hain!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Let's dive in! 🚀

# COMMAND ----------

# DBTITLE 1,🐍 Topic 1: Python Advanced - Theory
# MAGIC %md
# MAGIC # 1️⃣ Python Advanced - Foundation 🐍
# MAGIC
# MAGIC ## Kyun Zaroori Hai?
# MAGIC
# MAGIC Email assistant **concurrent** emails process karta hai (1 email = 1 async task). Agar blocking code likha to system slow ho jaayega!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 5 Key Concepts
# MAGIC
# MAGIC ### 1️⃣ **async/await** — Non-blocking Code
# MAGIC
# MAGIC **Problem:** Email processing mein LLM API call (2-3 seconds), database query (500ms), external API (1s) - sab sequential chale to **5+ seconds!**
# MAGIC
# MAGIC **Solution:** Async/await se parallel processing:
# MAGIC
# MAGIC ```python
# MAGIC # ❌ BAD: Sequential (5 seconds total)
# MAGIC result1 = call_llm()        # 2s
# MAGIC result2 = query_database()  # 0.5s
# MAGIC result3 = call_pega_api()   # 1s
# MAGIC
# MAGIC # ✅ GOOD: Concurrent (2s total - max of all)
# MAGIC import asyncio
# MAGIC result1, result2, result3 = await asyncio.gather(
# MAGIC     call_llm_async(),
# MAGIC     query_database_async(),
# MAGIC     call_pega_api_async()
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Mein:** 
# MAGIC - `async def process_email()` — main handler
# MAGIC - `await llm.invoke()` — LLM call
# MAGIC - `await db.save()` — state save
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ **dataclasses** — Config Objects
# MAGIC
# MAGIC **Problem:** Config dictionaries messy ho jaate hain:
# MAGIC
# MAGIC ```python
# MAGIC # ❌ BAD
# MAGIC config = {
# MAGIC     "llm_model": "gpt-4o",
# MAGIC     "temperature": 0.7,
# MAGIC     "max_tokens": 500
# MAGIC }
# MAGIC model = config["llm_model"]  # typo = runtime error!
# MAGIC ```
# MAGIC
# MAGIC **Solution:** dataclass with type hints:
# MAGIC
# MAGIC ```python
# MAGIC from dataclasses import dataclass
# MAGIC
# MAGIC @dataclass
# MAGIC class LLMConfig:
# MAGIC     llm_model: str = "gpt-4o"
# MAGIC     temperature: float = 0.7
# MAGIC     max_tokens: int = 500
# MAGIC
# MAGIC config = LLMConfig()
# MAGIC print(config.llm_model)  # Auto-complete + type checking!
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Mein:**
# MAGIC - `AgentConfig` — agent settings
# MAGIC - `ExecutorConfig` — executor agent config
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ **Type Hints** — Code Clarity
# MAGIC
# MAGIC **Benefit:** IDE auto-complete + early error detection
# MAGIC
# MAGIC ```python
# MAGIC # Without type hints
# MAGIC def process_email(email):
# MAGIC     return email.split()  # What is email? string? dict? list?
# MAGIC
# MAGIC # With type hints
# MAGIC def process_email(email: str) -> list[str]:
# MAGIC     return email.split()  # Clear! Takes string, returns list of strings
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Mein:** Har function fully typed:
# MAGIC
# MAGIC ```python
# MAGIC async def classify_intent(
# MAGIC     email_text: str, 
# MAGIC     llm: HorizonLlmChat
# MAGIC ) -> IntentClassificationResponse:
# MAGIC     ...
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ **Decorators** — Reusable Wrappers
# MAGIC
# MAGIC **Use Case:** Logging, retry logic, timing
# MAGIC
# MAGIC ```python
# MAGIC import functools
# MAGIC import time
# MAGIC
# MAGIC def log_execution(func):
# MAGIC     @functools.wraps(func)
# MAGIC     async def wrapper(*args, **kwargs):
# MAGIC         start = time.time()
# MAGIC         print(f"▶️ Starting {func.__name__}")
# MAGIC         result = await func(*args, **kwargs)
# MAGIC         print(f"✅ Finished in {time.time()-start:.2f}s")
# MAGIC         return result
# MAGIC     return wrapper
# MAGIC
# MAGIC @log_execution
# MAGIC async def call_llm(prompt: str):
# MAGIC     # LLM call here
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Mein:**
# MAGIC - `@retry(max_attempts=3)` — LLM failures pe retry
# MAGIC - `@log_step` — observability
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5️⃣ **Exceptions** — Error Handling
# MAGIC
# MAGIC **Pattern:** Custom exceptions for different failures
# MAGIC
# MAGIC ```python
# MAGIC class LLMTimeoutError(Exception):
# MAGIC     pass
# MAGIC
# MAGIC class InvalidEmailFormatError(Exception):
# MAGIC     pass
# MAGIC
# MAGIC try:
# MAGIC     result = await process_email(email)
# MAGIC except LLMTimeoutError:
# MAGIC     # Retry or fallback
# MAGIC     result = use_rule_based_classifier()
# MAGIC except InvalidEmailFormatError:
# MAGIC     # Send to human review queue
# MAGIC     send_to_human_queue(email)
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Mein:**
# MAGIC - `TokenRefreshError` — OAuth token issues
# MAGIC - `PlanningError` — planner agent failures
# MAGIC - `ExecutionError` — executor failures
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Feature | Email Assistant Use Case |
# MAGIC |---------|-------------------------|
# MAGIC | async/await | Parallel LLM + DB + API calls |
# MAGIC | dataclasses | Config objects (AgentConfig, ExecutorConfig) |
# MAGIC | Type hints | Full typing for clarity + IDE support |
# MAGIC | Decorators | @log_step, @retry |
# MAGIC | Exceptions | Custom errors for each failure type |
# MAGIC
# MAGIC Next: Practical examples! 🚀

# COMMAND ----------

# DBTITLE 1,Setup - Install Libraries
# Install required libraries
%pip install plotly ipywidgets pydantic fastapi python-jose[cryptography] passlib aiohttp -q

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import functools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown, widgets
from IPython.display import display, clear_output

print("✅ Libraries installed!")
print("🚀 Ready to learn Agentic LLM Systems!")

# COMMAND ----------

# DBTITLE 1,💻 Example 1.1: async/await - Sequential vs Concurrent
import asyncio
import time
import random

print("💻 Example: Sequential vs Concurrent Email Processing\n")

# Simulate API calls (LLM, DB, PEGA)
async def call_llm(email_id: int) -> str:
    """Simulate LLM API call (2 seconds)"""
    await asyncio.sleep(2)
    return f"Email {email_id}: Intent classified as ID_CARD_REQUEST"

async def save_to_db(email_id: int) -> str:
    """Simulate database save (0.5 seconds)"""
    await asyncio.sleep(0.5)
    return f"Email {email_id}: Saved to DB"

async def send_to_pega(email_id: int) -> str:
    """Simulate PEGA API call (1 second)"""
    await asyncio.sleep(1)
    return f"Email {email_id}: Sent to PEGA queue"

# ❌ BAD: Sequential processing
async def process_email_sequential(email_id: int):
    start = time.time()
    
    result1 = await call_llm(email_id)
    result2 = await save_to_db(email_id)
    result3 = await send_to_pega(email_id)
    
    elapsed = time.time() - start
    return f"Sequential: {elapsed:.2f}s", [result1, result2, result3]

# ✅ GOOD: Concurrent processing
async def process_email_concurrent(email_id: int):
    start = time.time()
    
    # Run all three in parallel!
    results = await asyncio.gather(
        call_llm(email_id),
        save_to_db(email_id),
        send_to_pega(email_id)
    )
    
    elapsed = time.time() - start
    return f"Concurrent: {elapsed:.2f}s", results

# Test both approaches
print("⏱️ Processing Email #1 (Sequential)...")
timing1, results1 = await process_email_sequential(1)
print(f"   {timing1}")
for r in results1:
    print(f"   - {r}")

print("\n⏱️ Processing Email #2 (Concurrent)...")
timing2, results2 = await process_email_concurrent(2)
print(f"   {timing2}")
for r in results2:
    print(f"   - {r}")

print("\n📊 Comparison:")
print(f"   Sequential: ~3.5 seconds (2 + 0.5 + 1)")
print(f"   Concurrent: ~2.0 seconds (max of 2, 0.5, 1)")
print(f"   ✅ Speedup: ~1.75x faster!")

print("\n💡 Real Email Assistant: 100s of emails can be processed concurrently!")

# COMMAND ----------

# DBTITLE 1,💻 Example 1.2: dataclasses - Config Management
from dataclasses import dataclass, field
from typing import List, Optional

print("💻 Example: dataclasses for Configuration\n")

# Email Assistant Config using dataclass
@dataclass
class LLMConfig:
    """LLM Configuration"""
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 500
    timeout_seconds: int = 30
    
@dataclass
class AgentConfig:
    """Agent Configuration"""
    agent_name: str
    llm_config: LLMConfig
    retry_attempts: int = 3
    confidence_threshold: float = 0.8
    fallback_to_human: bool = True
    
@dataclass
class ExecutorAgent:
    """Executor Agent (ID Card, Address, Inquiry)"""
    executor_type: str  # "id_card", "address_change", "inquiry"
    config: AgentConfig
    pega_queue: str
    supported_intents: List[str] = field(default_factory=list)

# Create configurations
llm_config = LLMConfig(
    model_name="gpt-4o",
    temperature=0.3,  # Lower for more deterministic output
    max_tokens=300
)

agent_config = AgentConfig(
    agent_name="ID_Card_Agent",
    llm_config=llm_config,
    confidence_threshold=0.85
)

id_card_agent = ExecutorAgent(
    executor_type="id_card",
    config=agent_config,
    pega_queue="QUEUE_ID_CARD_REQUESTS",
    supported_intents=["id_card_request", "id_card_replacement"]
)

print("✅ Configurations Created:\n")
print(f"Agent: {id_card_agent.config.agent_name}")
print(f"LLM Model: {id_card_agent.config.llm_config.model_name}")
print(f"Temperature: {id_card_agent.config.llm_config.temperature}")
print(f"PEGA Queue: {id_card_agent.pega_queue}")
print(f"Supported Intents: {', '.join(id_card_agent.supported_intents)}")
print(f"Confidence Threshold: {id_card_agent.config.confidence_threshold}")
print(f"Retry Attempts: {id_card_agent.config.retry_attempts}")

print("\n💡 Benefits:")
print("   ✅ Type safety (IDE catches typos)")
print("   ✅ Auto-complete in IDE")
print("   ✅ Default values")
print("   ✅ Nested configs (LLMConfig inside AgentConfig)")
print("   ✅ Easy serialization to JSON/dict")

# COMMAND ----------

# MAGIC %debug
# MAGIC import functools
# MAGIC import time
# MAGIC import random
# MAGIC import asyncio
# MAGIC
# MAGIC print("💻 Example: Decorators for Logging & Retry\n")
# MAGIC
# MAGIC # 1. Logging Decorator
# MAGIC def log_execution(func):
# MAGIC     """Decorator to log function execution time"""
# MAGIC     @functools.wraps(func)
# MAGIC     async def wrapper(*args, **kwargs):
# MAGIC         start = time.time()
# MAGIC         func_name = func.__name__
# MAGIC         print(f"  ▶️ [{func_name}] Starting...")
# MAGIC         
# MAGIC         try:
# MAGIC             result = await func(*args, **kwargs)
# MAGIC             elapsed = time.time() - start
# MAGIC             print(f"  ✅ [{func_name}] Completed in {elapsed:.2f}s")
# MAGIC             return result
# MAGIC         except Exception as e:
# MAGIC             elapsed = time.time() - start
# MAGIC             print(f"  ❌ [{func_name}] Failed after {elapsed:.2f}s: {e}")
# MAGIC             raise
# MAGIC     
# MAGIC     return wrapper
# MAGIC
# MAGIC # 2. Retry Decorator
# MAGIC def retry(max_attempts: int = 3, delay: float = 1.0):
# MAGIC     """Decorator to retry on failure"""
# MAGIC     def decorator(func):
# MAGIC         @functools.wraps(func)
# MAGIC         async def wrapper(*args, **kwargs):
# MAGIC             for attempt in range(1, max_attempts + 1):
# MAGIC                 try:
# MAGIC                     return await func(*args, **kwargs)
# MAGIC                 except Exception as e:
# MAGIC                     if attempt == max_attempts:
# MAGIC                         print(f"    ❌ All {max_attempts} attempts failed!")
# MAGIC                         raise
# MAGIC                     print(f"    ⚠️ Attempt {attempt}/{max_attempts} failed: {e}")
# MAGIC                     print(f"    ⏳ Retrying in {delay}s...")
# MAGIC                     await asyncio.sleep(delay)
# MAGIC             return None
# MAGIC         return wrapper
# MAGIC     return decorator
# MAGIC
# MAGIC # Simulate unreliable LLM API
# MAGIC @log_execution
# MAGIC @retry(max_attempts=3, delay=0.5)
# MAGIC async def call_unreliable_llm(prompt: str) -> str:
# MAGIC     """Simulates LLM API that fails randomly"""
# MAGIC     await asyncio.sleep(0.5)
# MAGIC     
# MAGIC     # 60% chance of failure
# MAGIC     if random.random() < 0.6:
# MAGIC         raise Exception("LLM API timeout")
# MAGIC     
# MAGIC     return f"LLM Response: Intent classified successfully"
# MAGIC
# MAGIC # Debugging tip: Add print statements inside decorators and function to trace flow
# MAGIC # For example, print attempt number in retry, and print prompt in call_unreliable_llm
# MAGIC
# MAGIC print("🧪 Testing unreliable LLM with retry logic:\n")
# MAGIC
# MAGIC try:
# MAGIC     result = await call_unreliable_llm("What is the intent of this email?")
# MAGIC     print(f"\n🎉 Success: {result}")
# MAGIC except Exception as e:
# MAGIC     print(f"\n🚫 Final failure: {e}")
# MAGIC
# MAGIC print("\n💡 Real Email Assistant:")
# MAGIC print("   - @log_step decorator tracks each step for observability")
# MAGIC print("   - @retry decorator handles transient LLM/API failures")
# MAGIC print("   - Multiple decorators can be stacked!")

# COMMAND ----------

func

# COMMAND ----------

result

# COMMAND ----------

# DBTITLE 1,💻 Example 1.3: Decorators - Logging & Retry


# COMMAND ----------

# DBTITLE 1,🎮 Interactive 1.1: Async Performance Comparison
print("🎮 Interactive: Async Performance Comparison\n")

async def simulate_email_processing(num_emails: int, use_concurrent: bool = True):
    """Simulate processing multiple emails"""
    
    async def process_one_email(email_id: int):
        # Simulate LLM (2s), DB (0.5s), API (1s)
        if use_concurrent:
            await asyncio.gather(
                asyncio.sleep(2),
                asyncio.sleep(0.5),
                asyncio.sleep(1)
            )
        else:
            await asyncio.sleep(2)
            await asyncio.sleep(0.5)
            await asyncio.sleep(1)
        return f"Email {email_id} processed"
    
    start = time.time()
    
    # Process all emails concurrently
    results = await asyncio.gather(*[
        process_one_email(i) for i in range(1, num_emails + 1)
    ])
    
    elapsed = time.time() - start
    return elapsed, results

def interactive_async_demo(num_emails=5, concurrent=True):
    """Interactive async performance demo"""
    
    # Run async function using asyncio.run()
    elapsed, results = asyncio.run(simulate_email_processing(num_emails, concurrent))
    
    # Calculate theoretical times
    if concurrent:
        theoretical_time = 2.0  # Max of (2, 0.5, 1)
        method = "Concurrent"
    else:
        theoretical_time = 3.5 * num_emails  # Sequential: (2+0.5+1) * N
        method = "Sequential"
    
    # Create visualization
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=["Actual Time", "Theoretical Time"],
        y=[elapsed, theoretical_time],
        marker=dict(
            color=['#4ECDC4', '#95E1D3'],
            line=dict(width=2, color='black')
        ),
        text=[f"{elapsed:.2f}s", f"{theoretical_time:.2f}s"],
        textposition='outside',
        hovertemplate='%{x}: %{y:.2f}s<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{method} Processing: {num_emails} Emails",
        yaxis_title="Time (seconds)",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    fig.show()
    
    # Print stats
    print(f"\n📊 Results:")
    print(f"   Method: {method}")
    print(f"   Emails processed: {num_emails}")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Time per email: {elapsed/num_emails:.2f}s")
    
    if concurrent:
        sequential_time = 3.5 * num_emails
        speedup = sequential_time / elapsed
        print(f"\n⚡ Speedup vs Sequential: {speedup:.2f}x faster!")
        print(f"   (Would take {sequential_time:.2f}s sequentially)")
    else:
        concurrent_time = 2.0
        slowdown = elapsed / concurrent_time
        print(f"\n🐌 Slowdown vs Concurrent: {slowdown:.2f}x slower!")
        print(f"   (Would take {concurrent_time:.2f}s concurrently)")

# Create interactive widget
interact(
    interactive_async_demo,
    num_emails=IntSlider(value=5, min=1, max=20, step=1, description='# Emails:', continuous_update=False),
    concurrent=widgets.Checkbox(value=True, description='Use Concurrent')
);

# COMMAND ----------

# DBTITLE 1,✅ Topic 2: Pydantic - Theory
# MAGIC %md
# MAGIC # 2️⃣ Pydantic - Data Validation ✅
# MAGIC
# MAGIC ## Kyun Zaroori Hai?
# MAGIC
# MAGIC Email assistant **har request/response** ko validate karna hota hai:
# MAGIC - Email format sahi hai?
# MAGIC - HCID (Health Care ID) 9 digits ka hai?
# MAGIC - LLM se aayi JSON valid hai?
# MAGIC - Confidence score 0-1 range mein hai?
# MAGIC
# MAGIC **Pydantic = Python + Validation + Type Safety**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Key Concepts
# MAGIC
# MAGIC ### 1️⃣ **BaseModel** — The Foundation
# MAGIC
# MAGIC Sare schemas `BaseModel` se inherit karte hain:
# MAGIC
# MAGIC ```python
# MAGIC from pydantic import BaseModel
# MAGIC
# MAGIC class EmailRequest(BaseModel):
# MAGIC     email_id: str
# MAGIC     sender: str
# MAGIC     subject: str
# MAGIC     body: str
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - Auto type conversion (`"123"` → `123` agar int chahiye)
# MAGIC - Validation (missing field = error)
# MAGIC - Serialization (`.dict()`, `.json()`)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ **Field Validation** — Custom Rules
# MAGIC
# MAGIC ```python
# MAGIC from pydantic import BaseModel, Field, validator
# MAGIC
# MAGIC class EmailRequest(BaseModel):
# MAGIC     email_id: str
# MAGIC     sender: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')  # Email format
# MAGIC     hcid: str = Field(..., min_length=9, max_length=9)  # Exactly 9 digits
# MAGIC     confidence: float = Field(..., ge=0.0, le=1.0)  # Between 0 and 1
# MAGIC ```
# MAGIC
# MAGIC **Field constraints:**
# MAGIC - `regex` — pattern matching
# MAGIC - `min_length`, `max_length` — string length
# MAGIC - `ge`, `le` — greater/less than or equal
# MAGIC - `gt`, `lt` — greater/less than
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ **Custom Validators** — Business Rules
# MAGIC
# MAGIC ```python
# MAGIC class EmailRequest(BaseModel):
# MAGIC     hcid: str
# MAGIC     
# MAGIC     @validator('hcid')
# MAGIC     def validate_hcid(cls, v):
# MAGIC         if not v.isdigit():
# MAGIC             raise ValueError('HCID must be numeric')
# MAGIC         if len(v) != 9:
# MAGIC             raise ValueError('HCID must be 9 digits')
# MAGIC         return v
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Mein:**
# MAGIC - HCID format validation
# MAGIC - Intent enum validation
# MAGIC - Address format validation
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ **Nested Models** — Complex Structures
# MAGIC
# MAGIC ```python
# MAGIC class Address(BaseModel):
# MAGIC     street: str
# MAGIC     city: str
# MAGIC     state: str
# MAGIC     zipcode: str
# MAGIC
# MAGIC class AddressChangeRequest(BaseModel):
# MAGIC     email_id: str
# MAGIC     hcid: str
# MAGIC     old_address: Address  # Nested!
# MAGIC     new_address: Address
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5️⃣ **LLM Response Parsing** — Structured Output
# MAGIC
# MAGIC **Most Important!** LLM se JSON nikalna:
# MAGIC
# MAGIC ```python
# MAGIC class IntentClassificationResponse(BaseModel):
# MAGIC     intent: str  # "id_card_request", "address_change", "inquiry"
# MAGIC     confidence: float
# MAGIC     entities: dict
# MAGIC     reasoning: str
# MAGIC
# MAGIC # LLM response
# MAGIC llm_output = '{"intent": "id_card_request", "confidence": 0.92, ...}'
# MAGIC
# MAGIC # Parse with Pydantic (auto-validates!)
# MAGIC response = IntentClassificationResponse.parse_raw(llm_output)
# MAGIC print(response.intent)  # "id_card_request"
# MAGIC ```
# MAGIC
# MAGIC **Agar LLM invalid JSON deta hai → Pydantic error throw karega → retry logic!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏗️ Email Assistant Architecture
# MAGIC
# MAGIC ```
# MAGIC FastAPI Request → EmailRequest (Pydantic)
# MAGIC        ↓
# MAGIC    LLM Call → Prompt
# MAGIC        ↓
# MAGIC    LLM Response (JSON string)
# MAGIC        ↓
# MAGIC    Pydantic Parse → IntentClassificationResponse
# MAGIC        ↓
# MAGIC    Route to Executor Agent
# MAGIC        ↓
# MAGIC    PEGA API Call → PEGARequest (Pydantic)
# MAGIC        ↓
# MAGIC    Response → EmailProcessingResponse (Pydantic)
# MAGIC ```
# MAGIC
# MAGIC **Har arrow pe Pydantic validation!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Feature | Email Assistant Use Case |
# MAGIC |---------|-------------------------|
# MAGIC | BaseModel | All request/response schemas |
# MAGIC | Field validation | Email format, HCID format, confidence range |
# MAGIC | Custom validators | Business rules (HCID 9 digits, valid intents) |
# MAGIC | Nested models | Address inside AddressChangeRequest |
# MAGIC | JSON parsing | **LLM output → Pydantic object** (core!) |
# MAGIC
# MAGIC **Without Pydantic:** Har jagah manual validation + type checking = messy code!
# MAGIC
# MAGIC Next: Real examples! 🚀

# COMMAND ----------

# DBTITLE 1,💻 Example 2.1: Pydantic Basics
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Optional, List
from enum import Enum

print("💻 Example: Pydantic for Email Assistant\n")

# 1. Intent Enum
class IntentType(str, Enum):
    ID_CARD_REQUEST = "id_card_request"
    ADDRESS_CHANGE = "address_change"
    INQUIRY = "inquiry"
    CLAIM_STATUS = "claim_status"
    UNKNOWN = "unknown"

# 2. Basic Email Request Model
class EmailRequest(BaseModel):
    email_id: str
    sender: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')  # Email validation
    subject: str
    body: str
    hcid: Optional[str] = Field(None, min_length=9, max_length=9)  # Optional, 9 digits
    
    @validator('hcid')
    def validate_hcid_numeric(cls, v):
        if v is not None and not v.isdigit():
            raise ValueError('HCID must contain only digits')
        return v

# 3. Intent Classification Response (from LLM)
class IntentClassificationResponse(BaseModel):
    intent: IntentType
    confidence: float = Field(..., ge=0.0, le=1.0)  # Between 0 and 1
    entities: dict
    reasoning: str

# 4. Address Model (nested)
class Address(BaseModel):
    street: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)  # 2-letter state code
    zipcode: str = Field(..., regex=r'^\d{5}$')  # 5 digits

# Test valid data
print("✅ Test 1: Valid Email Request")
try:
    email = EmailRequest(
        email_id="email_001",
        sender="john.doe@example.com",
        subject="Need ID Card",
        body="Please send me a new ID card",
        hcid="123456789"
    )
    print(f"   ✅ Valid! HCID: {email.hcid}")
except ValidationError as e:
    print(f"   ❌ Error: {e}")

# Test invalid email
print("\n❌ Test 2: Invalid Email Format")
try:
    email = EmailRequest(
        email_id="email_002",
        sender="invalid-email",  # Missing @ and domain
        subject="Test",
        body="Test body"
    )
    print(f"   ✅ Valid: {email.sender}")
except ValidationError as e:
    print(f"   ❌ Validation Error: {e.errors()[0]['msg']}")

# Test invalid HCID
print("\n❌ Test 3: Invalid HCID (not numeric)")
try:
    email = EmailRequest(
        email_id="email_003",
        sender="john@example.com",
        subject="Test",
        body="Test",
        hcid="ABC123456"  # Contains letters
    )
    print(f"   ✅ Valid: {email.hcid}")
except ValidationError as e:
    print(f"   ❌ Validation Error: {e.errors()[0]['msg']}")

# Test LLM response parsing
print("\n✅ Test 4: Parse LLM Response (JSON string)")
llm_json = '''
{
    "intent": "id_card_request",
    "confidence": 0.92,
    "entities": {"hcid": "123456789"},
    "reasoning": "Email mentions 'need ID card'"
}
'''

try:
    response = IntentClassificationResponse.parse_raw(llm_json)
    print(f"   ✅ Parsed successfully!")
    print(f"   Intent: {response.intent.value}")
    print(f"   Confidence: {response.confidence}")
    print(f"   Entities: {response.entities}")
except ValidationError as e:
    print(f"   ❌ Parse Error: {e}")

print("\n💡 Real Email Assistant:")
print("   - Every API request/response uses Pydantic")
print("   - LLM output is parsed to Pydantic models")
print("   - Invalid data = clear error messages")
print("   - Type safety throughout the system!")

# COMMAND ----------

# DBTITLE 1,⚡ Topic 3: FastAPI - Theory
# MAGIC %md
# MAGIC # 3️⃣ FastAPI - REST API Framework ⚡
# MAGIC
# MAGIC ## Kyun Zaroori Hai?
# MAGIC
# MAGIC Email assistant ko external system (frontend, email server, PEGA) se communicate karna hota hai - FastAPI iske liye perfect hai!
# MAGIC
# MAGIC **Why FastAPI?**
# MAGIC - ⚡ Fast (async support built-in)
# MAGIC - 📝 Auto documentation (Swagger UI)
# MAGIC - ✅ Pydantic integration (auto-validation)
# MAGIC - 🔒 Security (JWT, OAuth)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Key Concepts
# MAGIC
# MAGIC ### 1️⃣ **Routes** — API Endpoints
# MAGIC
# MAGIC ```python
# MAGIC from fastapi import FastAPI
# MAGIC
# MAGIC app = FastAPI()
# MAGIC
# MAGIC @app.post("/process-email")
# MAGIC async def process_email(request: EmailRequest):
# MAGIC     # Process email
# MAGIC     return {"status": "success"}
# MAGIC
# MAGIC @app.get("/status/{email_id}")
# MAGIC async def get_status(email_id: str):
# MAGIC     # Get processing status
# MAGIC     return {"email_id": email_id, "status": "processing"}
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Endpoints:**
# MAGIC - `POST /process-email` — Main email processing
# MAGIC - `GET /status/{email_id}` — Check processing status
# MAGIC - `POST /retry/{email_id}` — Retry failed email
# MAGIC - `GET /health` — Health check
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ **Request/Response Models** — Pydantic Integration
# MAGIC
# MAGIC ```python
# MAGIC class EmailRequest(BaseModel):
# MAGIC     email_id: str
# MAGIC     sender: str
# MAGIC     body: str
# MAGIC
# MAGIC class EmailResponse(BaseModel):
# MAGIC     email_id: str
# MAGIC     status: str
# MAGIC     intent: str
# MAGIC     queue: str
# MAGIC
# MAGIC @app.post("/process-email", response_model=EmailResponse)
# MAGIC async def process_email(request: EmailRequest) -> EmailResponse:
# MAGIC     # FastAPI auto-validates request using Pydantic!
# MAGIC     # And auto-serializes response
# MAGIC     return EmailResponse(
# MAGIC         email_id=request.email_id,
# MAGIC         status="completed",
# MAGIC         intent="id_card_request",
# MAGIC         queue="QUEUE_ID_CARD"
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ Auto request validation
# MAGIC - ✅ Auto response serialization
# MAGIC - ✅ Auto API docs generation
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ **JWT Authentication** — Secure APIs
# MAGIC
# MAGIC ```python
# MAGIC from fastapi import Depends, HTTPException, status
# MAGIC from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# MAGIC from jose import JWTError, jwt
# MAGIC
# MAGIC security = HTTPBearer()
# MAGIC
# MAGIC async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
# MAGIC     token = credentials.credentials
# MAGIC     try:
# MAGIC         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
# MAGIC         return payload
# MAGIC     except JWTError:
# MAGIC         raise HTTPException(status_code=401, detail="Invalid token")
# MAGIC
# MAGIC @app.post("/process-email")
# MAGIC async def process_email(
# MAGIC     request: EmailRequest,
# MAGIC     user = Depends(verify_token)  # Token required!
# MAGIC ):
# MAGIC     # Only authenticated requests allowed
# MAGIC     return {"status": "success"}
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant:** External systems need JWT token to call API
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ **Dependency Injection** — Shared Resources
# MAGIC
# MAGIC ```python
# MAGIC # Database connection as dependency
# MAGIC async def get_db():
# MAGIC     db = OracleDBConnector()
# MAGIC     try:
# MAGIC         yield db
# MAGIC     finally:
# MAGIC         await db.close()
# MAGIC
# MAGIC @app.post("/process-email")
# MAGIC async def process_email(
# MAGIC     request: EmailRequest,
# MAGIC     db = Depends(get_db)  # Auto-inject DB!
# MAGIC ):
# MAGIC     # Use db here
# MAGIC     await db.save_email(request)
# MAGIC     return {"status": "saved"}
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ Reusable dependencies (DB, LLM, TokenManager)
# MAGIC - ✅ Auto cleanup
# MAGIC - ✅ Easy testing (mock dependencies)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5️⃣ **Async Support** — Non-blocking I/O
# MAGIC
# MAGIC ```python
# MAGIC @app.post("/process-email")
# MAGIC async def process_email(request: EmailRequest):
# MAGIC     # All I/O operations are non-blocking!
# MAGIC     intent = await classify_intent_llm(request.body)
# MAGIC     await save_to_db(intent)
# MAGIC     await send_to_pega(intent)
# MAGIC     return {"status": "success"}
# MAGIC ```
# MAGIC
# MAGIC FastAPI natively supports `async/await` → high concurrency!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏗️ Email Assistant API Architecture
# MAGIC
# MAGIC ```
# MAGIC External Request (JWT token)
# MAGIC        ↓
# MAGIC    FastAPI Endpoint (/process-email)
# MAGIC        ↓
# MAGIC    Pydantic Validation (EmailRequest)
# MAGIC        ↓
# MAGIC    Planner Agent (orchestrator)
# MAGIC        ↓
# MAGIC    LLM + DB + PEGA (all async!)
# MAGIC        ↓
# MAGIC    Pydantic Response (EmailResponse)
# MAGIC        ↓
# MAGIC    JSON Response to Client
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Feature | Email Assistant Use Case |
# MAGIC |---------|-------------------------|
# MAGIC | Routes | /process-email, /status, /retry |
# MAGIC | Pydantic Models | Auto request/response validation |
# MAGIC | JWT Auth | Secure API access |
# MAGIC | Dependencies | DB, LLM, TokenManager injection |
# MAGIC | Async | Concurrent email processing |
# MAGIC
# MAGIC **FastAPI = Perfect for production-grade async APIs with Pydantic!**
# MAGIC
# MAGIC Next: Real API examples! 🚀

# COMMAND ----------

# DBTITLE 1,💻 Example 3.1: FastAPI Email Assistant
# Note: This is a simplified example (won't run a full server in notebook)
# Real email assistant would use: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import asyncio

print("💻 Example: FastAPI Email Assistant API\n")

# Models (Pydantic)
class IntentType(str, Enum):
    ID_CARD = "id_card_request"
    ADDRESS_CHANGE = "address_change"
    INQUIRY = "inquiry"

class EmailRequest(BaseModel):
    email_id: str
    sender: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    subject: str
    body: str
    hcid: Optional[str] = Field(None, min_length=9, max_length=9)

class EmailResponse(BaseModel):
    email_id: str
    status: str  # "processing", "completed", "failed"
    intent: Optional[IntentType] = None
    confidence: Optional[float] = None
    queue: Optional[str] = None
    message: str

# Create FastAPI app
app = FastAPI(
    title="Email Assistant API",
    description="Automated email triage for health insurance",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Simulated services
class EmailService:
    async def classify_intent(self, body: str) -> tuple[IntentType, float]:
        """Simulate LLM intent classification"""
        await asyncio.sleep(0.5)  # Simulate LLM call
        
        # Simple rule-based for demo
        if "id card" in body.lower():
            return IntentType.ID_CARD, 0.92
        elif "address" in body.lower():
            return IntentType.ADDRESS_CHANGE, 0.88
        else:
            return IntentType.INQUIRY, 0.75
    
    async def route_to_pega(self, intent: IntentType) -> str:
        """Simulate PEGA queue routing"""
        await asyncio.sleep(0.3)  # Simulate API call
        
        queue_map = {
            IntentType.ID_CARD: "QUEUE_ID_CARD_REQUESTS",
            IntentType.ADDRESS_CHANGE: "QUEUE_ADDRESS_CHANGES",
            IntentType.INQUIRY: "QUEUE_GENERAL_INQUIRY"
        }
        return queue_map[intent]

email_service = EmailService()

# Dependency: Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    
    # Simplified validation (real app would decode JWT)
    if token != "valid-token-123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return {"user": "api_client"}

# Main endpoint
@app.post("/process-email", response_model=EmailResponse)
async def process_email(
    request: EmailRequest,
    auth = Depends(verify_token)
) -> EmailResponse:
    """
    Process incoming email:
    1. Classify intent using LLM
    2. Route to appropriate PEGA queue
    3. Return processing result
    """
    try:
        # Step 1: Classify intent
        intent, confidence = await email_service.classify_intent(request.body)
        
        # Step 2: Route to PEGA
        queue = await email_service.route_to_pega(intent)
        
        # Step 3: Return response
        return EmailResponse(
            email_id=request.email_id,
            status="completed",
            intent=intent,
            confidence=confidence,
            queue=queue,
            message=f"Email successfully routed to {queue}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/status/{email_id}")
async def get_status(
    email_id: str,
    auth = Depends(verify_token)
):
    """Get processing status of an email"""
    # In real app, would query database
    return {
        "email_id": email_id,
        "status": "completed",
        "timestamp": "2026-06-12T10:30:00Z"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)"""
    return {
        "status": "healthy",
        "service": "email-assistant",
        "version": "1.0.0"
    }

print("✅ FastAPI app defined!")
print("\n📝 API Endpoints:")
print("   POST /process-email - Process email (requires JWT)")
print("   GET  /status/{email_id} - Get status (requires JWT)")
print("   GET  /health - Health check (public)")

print("\n💡 To run: uvicorn main:app --reload")
print("   Swagger docs: http://localhost:8000/docs")
print("   ReDoc: http://localhost:8000/redoc")

# COMMAND ----------

# DBTITLE 1,🧠 Topic 4: LLM Basics - Theory (THE HEART!)
# MAGIC %md
# MAGIC # 4️⃣ LLM Basics - The Brain 🧠
# MAGIC
# MAGIC ## **YEH PROJECT KA DIL HAI! ❤️**
# MAGIC
# MAGIC Email assistant ka **core capability** = LLM se **structured JSON output** nikalna!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Key Concepts
# MAGIC
# MAGIC ### 1️⃣ **System vs User Messages** — Conversation Structure
# MAGIC
# MAGIC **System Message:** LLM ko instructions dena ("You are an expert email classifier")
# MAGIC **User Message:** Actual input ("Classify this email: ...")
# MAGIC
# MAGIC ```python
# MAGIC messages = [
# MAGIC     {
# MAGIC         "role": "system",
# MAGIC         "content": "You are an expert email classifier for health insurance."
# MAGIC     },
# MAGIC     {
# MAGIC         "role": "user",
# MAGIC         "content": "Classify this email: I need a new ID card."
# MAGIC     }
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant Pattern:**
# MAGIC ```python
# MAGIC System: "You are an email intent classifier. Extract: intent, confidence, entities."
# MAGIC User: "Email: {email_body}"
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ **Temperature** — Creativity Control
# MAGIC
# MAGIC **Temperature** = LLM ka "creativity" level (0.0 to 2.0)
# MAGIC
# MAGIC - **0.0** → Deterministic (same input = same output)
# MAGIC - **0.7** → Balanced (default)
# MAGIC - **1.5+** → Creative (random, unpredictable)
# MAGIC
# MAGIC **Email Assistant:**
# MAGIC ```python
# MAGIC temperature = 0.3  # Low! We want consistent classification
# MAGIC ```
# MAGIC
# MAGIC Kyun? Classification predictable hona chahiye, creative nahi!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ **Tokens** — Cost & Context
# MAGIC
# MAGIC **Token** = LLM ka basic unit (~0.75 words)
# MAGIC
# MAGIC **Example:**
# MAGIC - "Hello world" = 2 tokens
# MAGIC - "I need an ID card" = 5 tokens
# MAGIC
# MAGIC **Why important:**
# MAGIC - 💰 **Cost:** APIs charge per token
# MAGIC - 📈 **Context Window:** GPT-4o = 128K tokens max
# MAGIC - ⏱️ **Speed:** More tokens = slower response
# MAGIC
# MAGIC **Email Assistant:**
# MAGIC ```python
# MAGIC max_tokens = 500  # Limit response length
# MAGIC # Typical email: 200-500 tokens
# MAGIC # Response: 100-200 tokens
# MAGIC # Total: ~300-700 tokens per request
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ **Structured JSON Output** — **THE HEART! ❤️**
# MAGIC
# MAGIC **Problem:** LLM free-form text deta hai:
# MAGIC ```
# MAGIC "This email is about an ID card request. The user wants a replacement card."
# MAGIC ```
# MAGIC
# MAGIC **Solution:** JSON format mein force karo!
# MAGIC
# MAGIC #### **Method 1: Prompt Engineering**
# MAGIC
# MAGIC ```python
# MAGIC system_prompt = """
# MAGIC You are an email classifier. ALWAYS respond in JSON format:
# MAGIC {
# MAGIC     "intent": "id_card_request" | "address_change" | "inquiry",
# MAGIC     "confidence": 0.0-1.0,
# MAGIC     "entities": {"key": "value"},
# MAGIC     "reasoning": "why you classified this way"
# MAGIC }
# MAGIC """
# MAGIC ```
# MAGIC
# MAGIC #### **Method 2: Function Calling (GPT-4)**
# MAGIC
# MAGIC ```python
# MAGIC functions = [
# MAGIC     {
# MAGIC         "name": "classify_email",
# MAGIC         "parameters": {
# MAGIC             "type": "object",
# MAGIC             "properties": {
# MAGIC                 "intent": {"type": "string", "enum": [...]},
# MAGIC                 "confidence": {"type": "number"},
# MAGIC                 "entities": {"type": "object"}
# MAGIC             },
# MAGIC             "required": ["intent", "confidence"]
# MAGIC         }
# MAGIC     }
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC #### **Method 3: JSON Mode (GPT-4 Turbo+)**
# MAGIC
# MAGIC ```python
# MAGIC response = llm.invoke(
# MAGIC     messages,
# MAGIC     response_format={"type": "json_object"}  # Forces JSON!
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5️⃣ **Email Assistant Flow**
# MAGIC
# MAGIC ```
# MAGIC Email Body
# MAGIC     ↓
# MAGIC [System Prompt: "You are a classifier. Output JSON."]
# MAGIC     ↓
# MAGIC [User Prompt: "Classify: {email_body}"]
# MAGIC     ↓
# MAGIC LLM (GPT-4o via Horizon)
# MAGIC     ↓
# MAGIC JSON String: '{"intent": "id_card_request", ...}'
# MAGIC     ↓
# MAGIC Pydantic.parse_raw() → IntentClassificationResponse object
# MAGIC     ↓
# MAGIC if valid → Route to executor agent
# MAGIC if invalid → Retry or fallback
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎁 Complete Example
# MAGIC
# MAGIC ```python
# MAGIC # System prompt (detailed instructions)
# MAGIC system_prompt = """
# MAGIC You are an expert email classifier for Elevance Health insurance.
# MAGIC
# MAGIC Your task:
# MAGIC 1. Classify email intent
# MAGIC 2. Extract entities (HCID, dates, addresses)
# MAGIC 3. Assign confidence score
# MAGIC
# MAGIC Intents:
# MAGIC - id_card_request: User needs ID card
# MAGIC - address_change: User changing address
# MAGIC - inquiry: General question
# MAGIC
# MAGIC OUTPUT FORMAT (JSON only):
# MAGIC {
# MAGIC     "intent": "<intent>",
# MAGIC     "confidence": <0.0-1.0>,
# MAGIC     "entities": {"hcid": "...", ...},
# MAGIC     "reasoning": "<why>"
# MAGIC }
# MAGIC """
# MAGIC
# MAGIC # User message
# MAGIC email_body = "I moved to a new address. My HCID is 123456789."
# MAGIC user_message = f"Classify this email:\n\n{email_body}"
# MAGIC
# MAGIC # Call LLM
# MAGIC response = await llm.invoke(
# MAGIC     messages=[
# MAGIC         {"role": "system", "content": system_prompt},
# MAGIC         {"role": "user", "content": user_message}
# MAGIC     ],
# MAGIC     temperature=0.3,
# MAGIC     max_tokens=300
# MAGIC )
# MAGIC
# MAGIC # Parse response
# MAGIC llm_output = response.content  # JSON string
# MAGIC result = IntentClassificationResponse.parse_raw(llm_output)
# MAGIC
# MAGIC print(result.intent)  # "address_change"
# MAGIC print(result.confidence)  # 0.91
# MAGIC print(result.entities)  # {"hcid": "123456789"}
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Concept | Email Assistant Use |
# MAGIC |---------|--------------------|
# MAGIC | System Message | Detailed classification instructions |
# MAGIC | User Message | Email body to classify |
# MAGIC | Temperature | 0.3 (low for consistency) |
# MAGIC | Tokens | ~500 per request (email + response) |
# MAGIC | **Structured Output** | **Force JSON → Pydantic → Type-safe!** |
# MAGIC
# MAGIC **Without structured output:** Free-form text → messy parsing → errors!
# MAGIC
# MAGIC **With structured output:** JSON → Pydantic validation → clean code! ✅
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Insight
# MAGIC
# MAGIC Email assistant = **LLM as structured function:**
# MAGIC
# MAGIC ```
# MAGIC f(email_body) → IntentClassificationResponse
# MAGIC ```
# MAGIC
# MAGIC Not a chatbot! It's a **function** that takes text and returns structured data!
# MAGIC
# MAGIC Next: Real LLM examples! 🚀

# COMMAND ----------

# DBTITLE 1,💻 Example 4.1: Structured JSON Output Simulation
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Optional
from enum import Enum

print("💻 Example: LLM Structured Output Pattern\n")

# Define response schema (Pydantic)
class IntentType(str, Enum):
    ID_CARD_REQUEST = "id_card_request"
    ADDRESS_CHANGE = "address_change"
    INQUIRY = "inquiry"
    CLAIM_STATUS = "claim_status"

class IntentClassificationResponse(BaseModel):
    intent: IntentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: Dict[str, str]
    reasoning: str

# Simulated LLM responses (what GPT-4o would return)
def simulate_llm_call(email_body: str, temperature: float = 0.3) -> str:
    """
    Simulate LLM API call
    In real app: HorizonLlmChat.invoke() returns JSON string
    """
    
    # Email 1: ID Card Request
    if "id card" in email_body.lower():
        return json.dumps({
            "intent": "id_card_request",
            "confidence": 0.95,
            "entities": {
                "hcid": "123456789",
                "request_type": "replacement"
            },
            "reasoning": "Email explicitly mentions 'need a new ID card'"
        })
    
    # Email 2: Address Change
    elif "address" in email_body.lower() or "moved" in email_body.lower():
        return json.dumps({
            "intent": "address_change",
            "confidence": 0.91,
            "entities": {
                "hcid": "987654321",
                "new_address": "123 Main St"
            },
            "reasoning": "Email mentions address change and provides HCID"
        })
    
    # Email 3: General Inquiry
    else:
        return json.dumps({
            "intent": "inquiry",
            "confidence": 0.78,
            "entities": {},
            "reasoning": "No specific action requested, appears to be a question"
        })

# Test emails
test_emails = [
    {
        "id": 1,
        "body": "Hi, I need a new ID card. My HCID is 123456789. Please send it to my address."
    },
    {
        "id": 2,
        "body": "I moved to a new address. HCID: 987654321. New address is 123 Main St, Austin, TX."
    },
    {
        "id": 3,
        "body": "What is the status of my claim? Can someone help me?"
    }
]

print("🧪 Simulating LLM Structured Output...\n")

for email in test_emails:
    print(f"📧 Email #{email['id']}:")
    print(f"   Body: {email['body'][:60]}..." if len(email['body']) > 60 else f"   Body: {email['body']}")
    
    # Step 1: Call LLM (simulated)
    llm_json_output = simulate_llm_call(email['body'])
    
    # Step 2: Parse with Pydantic
    try:
        response = IntentClassificationResponse.parse_raw(llm_json_output)
        
        print(f"   ✅ Intent: {response.intent.value}")
        print(f"   ✅ Confidence: {response.confidence:.2f}")
        print(f"   ✅ Entities: {response.entities}")
        print(f"   💬 Reasoning: {response.reasoning}")
        
        # Routing decision
        if response.confidence >= 0.85:
            print(f"   ✅ Auto-route to executor agent")
        else:
            print(f"   ⚠️ Send to human review (low confidence)")
            
    except ValidationError as e:
        print(f"   ❌ Validation Error: {e.errors()[0]['msg']}")
        print(f"   🔄 Retry LLM call or use fallback...")
    
    print()

print("💡 Real Email Assistant Flow:")
print("   1. Email → LLM with structured prompt")
print("   2. LLM → JSON string")
print("   3. Pydantic.parse_raw() → validated object")
print("   4. If valid + high confidence → route to executor")
print("   5. If invalid or low confidence → human review")

print("\n❤️ This is THE CORE of the email assistant!")

# COMMAND ----------

# DBTITLE 1,🎮 Interactive 4.1: Temperature Effect
import random

print("🎮 Interactive: Temperature Effect on LLM Output\n")

def simulate_temperature_effect(email: str, temperature: float, num_runs: int = 5):
    """
    Simulate how temperature affects LLM output consistency
    """
    
    # Base confidence for this email
    base_confidence = 0.85
    
    results = []
    
    for run in range(num_runs):
        # Temperature adds randomness
        # Low temp (0.0-0.3) = consistent
        # High temp (1.0+) = random
        noise = random.uniform(-temperature, temperature)
        confidence = max(0.0, min(1.0, base_confidence + noise * 0.3))
        
        # Intent might change at high temperature!
        if temperature > 1.0 and random.random() < 0.3:
            intent = random.choice(["id_card_request", "address_change", "inquiry"])
        else:
            intent = "id_card_request"  # Correct intent
        
        results.append({
            "run": run + 1,
            "intent": intent,
            "confidence": confidence
        })
    
    return results

def interactive_temperature_demo(temperature=0.3):
    """
    Interactive temperature comparison
    """
    
    email = "I need a new ID card. My HCID is 123456789."
    
    # Run simulation
    results = simulate_temperature_effect(email, temperature, num_runs=10)
    
    # Extract data
    runs = [r['run'] for r in results]
    confidences = [r['confidence'] for r in results]
    intents = [r['intent'] for r in results]
    
    # Count intent consistency
    correct_intents = sum(1 for i in intents if i == "id_card_request")
    consistency = (correct_intents / len(intents)) * 100
    
    # Create visualization
    fig = go.Figure()
    
    # Plot confidence scores
    colors = ['#4ECDC4' if i == "id_card_request" else '#FF6B6B' for i in intents]
    
    fig.add_trace(go.Scatter(
        x=runs,
        y=confidences,
        mode='markers+lines',
        marker=dict(
            size=12,
            color=colors,
            line=dict(width=2, color='black')
        ),
        line=dict(color='gray', width=1, dash='dash'),
        name='Confidence',
        hovertemplate='Run %{x}<br>Confidence: %{y:.3f}<br>Intent: ' + 
                     '<br>'.join([f'{i}' for i in intents]) + '<extra></extra>'
    ))
    
    # Add mean line
    mean_conf = sum(confidences) / len(confidences)
    fig.add_hline(
        y=mean_conf,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_conf:.3f}"
    )
    
    # Update layout
    fig.update_layout(
        title=f'Temperature: {temperature} | Intent Consistency: {consistency:.0f}%',
        xaxis_title='Run Number',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 1.05]),
        height=450,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.show()
    
    # Print analysis
    std_dev = (sum((c - mean_conf)**2 for c in confidences) / len(confidences)) ** 0.5
    
    print(f"\n📊 Analysis:")
    print(f"   Temperature: {temperature}")
    print(f"   Mean confidence: {mean_conf:.3f}")
    print(f"   Std deviation: {std_dev:.3f}")
    print(f"   Intent consistency: {consistency:.0f}%")
    print(f"   Correct intents: {correct_intents}/10")
    
    if temperature < 0.5:
        print(f"\n✅ Low temperature: Consistent, predictable output")
        print(f"   💡 Good for classification tasks!")
    elif temperature < 1.0:
        print(f"\n⚠️ Medium temperature: Some variation")
        print(f"   ⚖️ Balance between consistency and creativity")
    else:
        print(f"\n❌ High temperature: Unpredictable, random")
        print(f"   🎨 Good for creative tasks, NOT classification!")
    
    print(f"\n💡 Email Assistant uses: temperature = 0.3")
    print(f"   Why? Classification needs to be CONSISTENT!")

# Create interactive widget
interact(
    interactive_temperature_demo,
    temperature=FloatSlider(
        value=0.3,
        min=0.0,
        max=2.0,
        step=0.1,
        description='Temperature:',
        continuous_update=False
    )
);

# COMMAND ----------

# DBTITLE 1,🔗 Topic 5: LangChain - Theory
# MAGIC %md
# MAGIC # 5️⃣ LangChain Basics - LLM Framework 🔗
# MAGIC
# MAGIC ## Kyun Zaroori Hai?
# MAGIC
# MAGIC LangChain = LLM applications banane ka framework (prompts, chains, agents)
# MAGIC
# MAGIC **Email Assistant:** Horizon wrapper = LangChain pattern follow karta hai!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Key Concepts
# MAGIC
# MAGIC ### 1️⃣ **invoke() Method** — Core API
# MAGIC
# MAGIC ```python
# MAGIC from langchain.chat_models import ChatOpenAI
# MAGIC
# MAGIC llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
# MAGIC
# MAGIC # invoke() = synchronous call
# MAGIC response = llm.invoke([
# MAGIC     {"role": "system", "content": "You are a classifier"},
# MAGIC     {"role": "user", "content": "Classify: ..."}
# MAGIC ])
# MAGIC
# MAGIC print(response.content)  # JSON string
# MAGIC ```
# MAGIC
# MAGIC **Horizon Pattern:**
# MAGIC ```python
# MAGIC from horizon_langchain import HorizonLlmChat
# MAGIC
# MAGIC llm = HorizonLlmChat(
# MAGIC     model_name="gpt-4o",
# MAGIC     temperature=0.3
# MAGIC )
# MAGIC
# MAGIC response = await llm.ainvoke(messages)  # Async version
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ **AIMessage Format** — Response Structure
# MAGIC
# MAGIC LangChain response = `AIMessage` object:
# MAGIC
# MAGIC ```python
# MAGIC response = llm.invoke(messages)
# MAGIC
# MAGIC print(type(response))  # AIMessage
# MAGIC print(response.content)  # The actual text/JSON
# MAGIC print(response.additional_kwargs)  # Extra metadata
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant:**
# MAGIC ```python
# MAGIC response = await llm.ainvoke(messages)
# MAGIC json_str = response.content  # Extract JSON string
# MAGIC result = IntentClassificationResponse.parse_raw(json_str)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ **Message Format** — Standard Structure
# MAGIC
# MAGIC LangChain messages = list of dicts:
# MAGIC
# MAGIC ```python
# MAGIC messages = [
# MAGIC     {"role": "system", "content": "System instructions"},
# MAGIC     {"role": "user", "content": "User input"},
# MAGIC     {"role": "assistant", "content": "Previous response"},  # Optional
# MAGIC     {"role": "user", "content": "Follow-up"}  # Optional
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC **Email Assistant:** Usually just system + user (no conversation history)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ **Horizon Wrapper Pattern**
# MAGIC
# MAGIC Email assistant uses **proprietary Horizon wrapper** (Elevance Health):
# MAGIC
# MAGIC ```python
# MAGIC # horizon_langchain.py (simplified)
# MAGIC class HorizonLlmChat:
# MAGIC     def __init__(self, model_name: str, temperature: float):
# MAGIC         self.model_name = model_name
# MAGIC         self.temperature = temperature
# MAGIC         self.token_manager = TokenManager()  # OAuth tokens
# MAGIC     
# MAGIC     async def ainvoke(self, messages: list) -> AIMessage:
# MAGIC         # Get OAuth token
# MAGIC         token = await self.token_manager.get_token()
# MAGIC         
# MAGIC         # Call Horizon API (which calls GPT-4o)
# MAGIC         response = await self._call_horizon_api(
# MAGIC             messages=messages,
# MAGIC             token=token,
# MAGIC             model=self.model_name,
# MAGIC             temperature=self.temperature
# MAGIC         )
# MAGIC         
# MAGIC         # Return LangChain-compatible AIMessage
# MAGIC         return AIMessage(content=response["content"])
# MAGIC ```
# MAGIC
# MAGIC **Key Points:**
# MAGIC - Follows LangChain interface (`.ainvoke()`, `AIMessage`)
# MAGIC - Internally uses OAuth (TokenManager)
# MAGIC - Calls Horizon API → which calls GPT-4o
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5️⃣ **Why LangChain Pattern?**
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ Standard interface (easy to swap LLM providers)
# MAGIC - ✅ Well-documented patterns
# MAGIC - ✅ Async support
# MAGIC - ✅ Community ecosystem
# MAGIC
# MAGIC **Email Assistant:**
# MAGIC - Horizon wrapper = LangChain-compatible
# MAGIC - Easy to test (mock LLM responses)
# MAGIC - Could switch to OpenAI directly if needed
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏗️ Complete Flow
# MAGIC
# MAGIC ```python
# MAGIC # 1. Initialize
# MAGIC llm = HorizonLlmChat(model_name="gpt-4o", temperature=0.3)
# MAGIC
# MAGIC # 2. Prepare messages
# MAGIC system_msg = "You are an email classifier. Output JSON."
# MAGIC user_msg = f"Classify: {email_body}"
# MAGIC
# MAGIC messages = [
# MAGIC     {"role": "system", "content": system_msg},
# MAGIC     {"role": "user", "content": user_msg}
# MAGIC ]
# MAGIC
# MAGIC # 3. Call LLM
# MAGIC response = await llm.ainvoke(messages)
# MAGIC
# MAGIC # 4. Extract JSON
# MAGIC json_str = response.content
# MAGIC
# MAGIC # 5. Parse with Pydantic
# MAGIC result = IntentClassificationResponse.parse_raw(json_str)
# MAGIC
# MAGIC # 6. Use result
# MAGIC if result.confidence >= 0.85:
# MAGIC     await route_to_executor(result.intent)
# MAGIC else:
# MAGIC     await send_to_human_review()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Concept | Email Assistant Use |
# MAGIC |---------|--------------------|
# MAGIC | invoke() | Core method to call LLM |
# MAGIC | AIMessage | LangChain response format |
# MAGIC | Message format | [{"role": "system", "content": ...}] |
# MAGIC | Horizon wrapper | Custom LangChain-compatible wrapper |
# MAGIC | Pattern | Standard interface, proprietary backend |
# MAGIC
# MAGIC **Key Insight:** LangChain = interface standard, Horizon = implementation!
# MAGIC
# MAGIC Next: OAuth & TokenManager! 🔐

# COMMAND ----------

# DBTITLE 1,🔐 Topic 6: OAuth Client-Credentials - Theory
# MAGIC %md
# MAGIC # 6️⃣ OAuth Client-Credentials - Authentication 🔐
# MAGIC
# MAGIC ## Kyun Zaroori Hai?
# MAGIC
# MAGIC Horizon API (aur PEGA) ko call karne ke liye **OAuth token** chahiye!
# MAGIC
# MAGIC **Client-Credentials Flow** = Server-to-server authentication (no user involved)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 OAuth Flow
# MAGIC
# MAGIC ### **Step-by-Step:**
# MAGIC
# MAGIC ```
# MAGIC 1. Application starts
# MAGIC        ↓
# MAGIC 2. TokenManager initialized with client_id + client_secret
# MAGIC        ↓
# MAGIC 3. Request token from OAuth server:
# MAGIC    POST /oauth/token
# MAGIC    Body: {
# MAGIC        "grant_type": "client_credentials",
# MAGIC        "client_id": "...",
# MAGIC        "client_secret": "...",
# MAGIC        "scope": "llm.api"
# MAGIC    }
# MAGIC        ↓
# MAGIC 4. OAuth server responds:
# MAGIC    {
# MAGIC        "access_token": "eyJhbG...",
# MAGIC        "token_type": "Bearer",
# MAGIC        "expires_in": 3600  # 1 hour
# MAGIC    }
# MAGIC        ↓
# MAGIC 5. Store token + expiry time
# MAGIC        ↓
# MAGIC 6. Use token in API calls:
# MAGIC    Authorization: Bearer eyJhbG...
# MAGIC        ↓
# MAGIC 7. Before expiry → refresh (go to step 3)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 TokenManager Class
# MAGIC
# MAGIC ```python
# MAGIC class TokenManager:
# MAGIC     def __init__(self, client_id: str, client_secret: str, token_url: str):
# MAGIC         self.client_id = client_id
# MAGIC         self.client_secret = client_secret
# MAGIC         self.token_url = token_url
# MAGIC         
# MAGIC         self.access_token: Optional[str] = None
# MAGIC         self.expires_at: Optional[float] = None
# MAGIC     
# MAGIC     async def get_token(self) -> str:
# MAGIC         """
# MAGIC         Get valid access token (refresh if expired)
# MAGIC         """
# MAGIC         # Check if token exists and is still valid
# MAGIC         if self.access_token and time.time() < self.expires_at:
# MAGIC             return self.access_token  # Use cached token
# MAGIC         
# MAGIC         # Token expired or doesn't exist → get new one
# MAGIC         await self._refresh_token()
# MAGIC         return self.access_token
# MAGIC     
# MAGIC     async def _refresh_token(self):
# MAGIC         """
# MAGIC         Request new token from OAuth server
# MAGIC         """
# MAGIC         response = await http_client.post(
# MAGIC             self.token_url,
# MAGIC             data={
# MAGIC                 "grant_type": "client_credentials",
# MAGIC                 "client_id": self.client_id,
# MAGIC                 "client_secret": self.client_secret,
# MAGIC                 "scope": "llm.api"
# MAGIC             }
# MAGIC         )
# MAGIC         
# MAGIC         data = response.json()
# MAGIC         self.access_token = data["access_token"]
# MAGIC         expires_in = data["expires_in"]  # seconds
# MAGIC         
# MAGIC         # Set expiry time (with 5-minute buffer)
# MAGIC         self.expires_at = time.time() + expires_in - 300
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Token Lifecycle
# MAGIC
# MAGIC ### **Scenario 1: First Call**
# MAGIC
# MAGIC ```
# MAGIC Email arrives → process_email()
# MAGIC     ↓
# MAGIC LLM call needed
# MAGIC     ↓
# MAGIC token_manager.get_token()
# MAGIC     ↓
# MAGIC No cached token → _refresh_token()
# MAGIC     ↓
# MAGIC OAuth server → new token
# MAGIC     ↓
# MAGIC Cache token + expiry
# MAGIC     ↓
# MAGIC Return token to LLM call
# MAGIC     ↓
# MAGIC LLM call succeeds
# MAGIC ```
# MAGIC
# MAGIC ### **Scenario 2: Subsequent Calls (within 1 hour)**
# MAGIC
# MAGIC ```
# MAGIC Another email → process_email()
# MAGIC     ↓
# MAGIC LLM call needed
# MAGIC     ↓
# MAGIC token_manager.get_token()
# MAGIC     ↓
# MAGIC Cached token still valid → return immediately
# MAGIC     ↓
# MAGIC LLM call succeeds (fast!)
# MAGIC ```
# MAGIC
# MAGIC ### **Scenario 3: Token Expired**
# MAGIC
# MAGIC ```
# MAGIC 1 hour passes...
# MAGIC New email → process_email()
# MAGIC     ↓
# MAGIC LLM call needed
# MAGIC     ↓
# MAGIC token_manager.get_token()
# MAGIC     ↓
# MAGIC Cached token expired → _refresh_token()
# MAGIC     ↓
# MAGIC Get new token
# MAGIC     ↓
# MAGIC LLM call succeeds
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Error Handling
# MAGIC
# MAGIC ```python
# MAGIC class TokenRefreshError(Exception):
# MAGIC     pass
# MAGIC
# MAGIC async def get_token(self) -> str:
# MAGIC     try:
# MAGIC         # ... token logic ...
# MAGIC         return self.access_token
# MAGIC     except Exception as e:
# MAGIC         raise TokenRefreshError(f"Failed to get OAuth token: {e}")
# MAGIC
# MAGIC # Usage in LLM call
# MAGIC try:
# MAGIC     token = await token_manager.get_token()
# MAGIC     response = await llm_api_call(token=token)
# MAGIC except TokenRefreshError:
# MAGIC     # Log error
# MAGIC     # Retry or fail gracefully
# MAGIC     logger.error("OAuth token refresh failed")
# MAGIC     raise
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏗️ Email Assistant Integration
# MAGIC
# MAGIC ```python
# MAGIC # 1. Initialize TokenManager at startup
# MAGIC token_manager = TokenManager(
# MAGIC     client_id=os.getenv("HORIZON_CLIENT_ID"),
# MAGIC     client_secret=os.getenv("HORIZON_CLIENT_SECRET"),
# MAGIC     token_url="https://oauth.elevance.com/token"
# MAGIC )
# MAGIC
# MAGIC # 2. Pass to HorizonLlmChat
# MAGIC llm = HorizonLlmChat(
# MAGIC     model_name="gpt-4o",
# MAGIC     token_manager=token_manager  # Injected!
# MAGIC )
# MAGIC
# MAGIC # 3. HorizonLlmChat uses it internally
# MAGIC class HorizonLlmChat:
# MAGIC     async def ainvoke(self, messages):
# MAGIC         # Get token (auto-refreshes if needed)
# MAGIC         token = await self.token_manager.get_token()
# MAGIC         
# MAGIC         # Use in API call
# MAGIC         headers = {"Authorization": f"Bearer {token}"}
# MAGIC         response = await http.post(api_url, headers=headers, ...)
# MAGIC         
# MAGIC         return AIMessage(content=response["content"])
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Concept | Email Assistant Use |
# MAGIC |---------|--------------------|
# MAGIC | OAuth Flow | Client-credentials (server-to-server) |
# MAGIC | TokenManager | Handles token lifecycle |
# MAGIC | Caching | Stores token for 1 hour (with buffer) |
# MAGIC | Auto-refresh | Checks expiry, refreshes if needed |
# MAGIC | Error handling | TokenRefreshError for failures |
# MAGIC
# MAGIC **Key Insight:** TokenManager = invisible to business logic, handles auth automatically!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Why This Pattern?
# MAGIC
# MAGIC - ✅ **Efficient:** Reuse tokens (don't request every time)
# MAGIC - ✅ **Automatic:** Business code doesn't worry about tokens
# MAGIC - ✅ **Secure:** Tokens expire (not permanent credentials)
# MAGIC - ✅ **Resilient:** Auto-refresh handles expiry gracefully
# MAGIC
# MAGIC Next: Agentic concepts! 🤝

# COMMAND ----------

# DBTITLE 1,🤝 Topic 7: Agentic Concepts - Theory (ARCHITECTURE!)
# MAGIC %md
# MAGIC # 7️⃣ Agentic Concepts - System Architecture 🤝
# MAGIC
# MAGIC ## **YEH EMAIL ASSISTANT KA DESIGN HAI! 🏗️**
# MAGIC
# MAGIC **Agentic System** = Multiple specialized agents working together!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 The 3-Agent Architecture
# MAGIC
# MAGIC ### **1. Orchestrator (FastAPI)**
# MAGIC
# MAGIC ```
# MAGIC FastAPI Endpoint
# MAGIC    │
# MAGIC    ├─ Receives email request
# MAGIC    ├─ Validates with Pydantic
# MAGIC    ├─ Calls Planner Agent
# MAGIC    └─ Returns response
# MAGIC ```
# MAGIC
# MAGIC **Role:** Entry point, HTTP handling, request/response management
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **2. Planner Agent (Decision Maker)**
# MAGIC
# MAGIC ```
# MAGIC Planner Agent
# MAGIC    │
# MAGIC    ├─ Calls LLM for intent classification
# MAGIC    ├─ Decides which executor agent to use
# MAGIC    ├─ Creates ExecutionPlan (list of steps)
# MAGIC    ├─ Routes to appropriate executor
# MAGIC    └─ Returns results
# MAGIC ```
# MAGIC
# MAGIC **Role:** 
# MAGIC - **Intent Classification:** "Is this ID card request, address change, or inquiry?"
# MAGIC - **Routing Decision:** "Which executor agent should handle this?"
# MAGIC - **Plan Creation:** "What steps are needed?"
# MAGIC
# MAGIC **Not an LLM itself!** → Uses LLM for classification, but is routing/planning logic
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **3. Executor Agents (Action Takers)**
# MAGIC
# MAGIC Three specialized executors:
# MAGIC
# MAGIC #### **A. ID Card Agent**
# MAGIC ```
# MAGIC ID Card Executor
# MAGIC    │
# MAGIC    ├─ Extracts: HCID, member info
# MAGIC    ├─ Validates: HCID format
# MAGIC    ├─ Calls: PEGA API (ID_CARD queue)
# MAGIC    └─ Logs: Action taken
# MAGIC ```
# MAGIC
# MAGIC #### **B. Address Change Agent**
# MAGIC ```
# MAGIC Address Change Executor
# MAGIC    │
# MAGIC    ├─ Extracts: HCID, old address, new address
# MAGIC    ├─ Validates: Address format
# MAGIC    ├─ Calls: SOA mainframe API
# MAGIC    └─ Logs: Address updated
# MAGIC ```
# MAGIC
# MAGIC #### **C. Inquiry Agent**
# MAGIC ```
# MAGIC Inquiry Executor
# MAGIC    │
# MAGIC    ├─ Extracts: Question type
# MAGIC    ├─ Routes to: Human queue (PEGA)
# MAGIC    └─ Logs: Sent to human review
# MAGIC ```
# MAGIC
# MAGIC **Role:** Execute specific actions based on intent
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Complete Flow Example
# MAGIC
# MAGIC ### **Email: "I need a new ID card. My HCID is 123456789."**
# MAGIC
# MAGIC ```
# MAGIC Step 1: Orchestrator (FastAPI)
# MAGIC    ↓
# MAGIC    POST /process-email
# MAGIC    Request validated (Pydantic)
# MAGIC    ↓
# MAGIC
# MAGIC Step 2: Planner Agent
# MAGIC    ↓
# MAGIC    Calls LLM for intent classification
# MAGIC    LLM returns: {
# MAGIC        "intent": "id_card_request",
# MAGIC        "confidence": 0.95,
# MAGIC        "entities": {"hcid": "123456789"}
# MAGIC    }
# MAGIC    ↓
# MAGIC    Planner decides: "Use ID Card Executor"
# MAGIC    Creates plan: [
# MAGIC        PlanStep(action="extract_entities"),
# MAGIC        PlanStep(action="validate_hcid"),
# MAGIC        PlanStep(action="send_to_pega")
# MAGIC    ]
# MAGIC    ↓
# MAGIC
# MAGIC Step 3: ID Card Executor
# MAGIC    ↓
# MAGIC    Executes plan steps:
# MAGIC    1. Extract entities → {"hcid": "123456789"}
# MAGIC    2. Validate HCID → ✅ Valid
# MAGIC    3. Send to PEGA → Queue: "QUEUE_ID_CARD_REQUESTS"
# MAGIC    ↓
# MAGIC
# MAGIC Step 4: Response
# MAGIC    ↓
# MAGIC    Return to orchestrator:
# MAGIC    {
# MAGIC        "status": "completed",
# MAGIC        "intent": "id_card_request",
# MAGIC        "queue": "QUEUE_ID_CARD_REQUESTS",
# MAGIC        "message": "ID card request processed"
# MAGIC    }
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Key Concepts Explained
# MAGIC
# MAGIC ### **1. Intent Classification**
# MAGIC
# MAGIC **What:** Determine what the user wants
# MAGIC
# MAGIC **How:** LLM reads email, outputs structured JSON:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC     "intent": "id_card_request",
# MAGIC     "confidence": 0.92,
# MAGIC     "entities": {"hcid": "123456789"},
# MAGIC     "reasoning": "Email explicitly mentions 'need ID card'"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Intents:**
# MAGIC - `id_card_request` → ID Card Agent
# MAGIC - `address_change` → Address Change Agent
# MAGIC - `inquiry` → Inquiry Agent (human review)
# MAGIC - `claim_status` → Inquiry Agent
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **2. Entity Extraction**
# MAGIC
# MAGIC **What:** Pull out specific data from email
# MAGIC
# MAGIC **Examples:**
# MAGIC - **HCID:** "123456789" (Health Care ID)
# MAGIC - **Address:** "123 Main St, Austin, TX 78701"
# MAGIC - **Date:** "June 15, 2026"
# MAGIC - **Name:** "John Doe"
# MAGIC
# MAGIC **How:** LLM extracts + Pydantic validates format
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **3. Routing (Intent → Queue)**
# MAGIC
# MAGIC **Mapping:**
# MAGIC
# MAGIC ```python
# MAGIC ROUTING_MAP = {
# MAGIC     "id_card_request": {
# MAGIC         "executor": IDCardExecutor,
# MAGIC         "pega_queue": "QUEUE_ID_CARD_REQUESTS"
# MAGIC     },
# MAGIC     "address_change": {
# MAGIC         "executor": AddressChangeExecutor,
# MAGIC         "pega_queue": "QUEUE_ADDRESS_CHANGES"
# MAGIC     },
# MAGIC     "inquiry": {
# MAGIC         "executor": InquiryExecutor,
# MAGIC         "pega_queue": "QUEUE_GENERAL_INQUIRY"
# MAGIC     }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Planner Agent uses this map to decide routing!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **4. Human-in-the-Loop**
# MAGIC
# MAGIC **When to send to human:**
# MAGIC
# MAGIC ```python
# MAGIC if intent_response.confidence < 0.85:
# MAGIC     # Low confidence → human review
# MAGIC     await send_to_human_queue(email)
# MAGIC else:
# MAGIC     # High confidence → auto-route to executor
# MAGIC     await route_to_executor(intent_response.intent)
# MAGIC ```
# MAGIC
# MAGIC **Threshold:** 0.85 (85% confidence)
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ Safety: Uncertain emails reviewed by humans
# MAGIC - ✅ Quality: Humans handle edge cases
# MAGIC - ✅ Learning: Human corrections improve system
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏗️ Architecture Diagram
# MAGIC
# MAGIC ```
# MAGIC                     ┌─────────────────┐
# MAGIC                     │  Email Server   │
# MAGIC                     └─────────┬───────┘
# MAGIC                              │
# MAGIC                              │ HTTP POST
# MAGIC                              │
# MAGIC                     ┌────────┴────────┐
# MAGIC                     │  FastAPI        │ ← Orchestrator
# MAGIC                     │  /process-email │
# MAGIC                     └────────┬────────┘
# MAGIC                              │
# MAGIC                              │
# MAGIC                     ┌────────┴────────┐
# MAGIC                     │  Planner Agent  │ ← Decision Maker
# MAGIC                     │  (Orchestrator) │
# MAGIC                     └────────┬────────┘
# MAGIC                              │
# MAGIC               ┌──────────┼──────────┐
# MAGIC               │              │              │
# MAGIC     ┌───────┴──────┐  ┌──┴──┐  ┌─────┴─────┐
# MAGIC     │ ID Card      │  │ LLM │  │  Address   │ ← Executors
# MAGIC     │ Executor     │  └──┬──┘  │  Change    │
# MAGIC     └─────┬──────┘     │     └───┬─────┘
# MAGIC           │              │          │
# MAGIC           │         ┌────┴────┐   │
# MAGIC           │         │  Inquiry │   │
# MAGIC           │         │  Executor│   │
# MAGIC           │         └────┬────┘   │
# MAGIC           │              │          │
# MAGIC           └──────────────┼──────────┘
# MAGIC                          │
# MAGIC                 ┌────────┴────────┐
# MAGIC                 │   PEGA Queues   │ ← External System
# MAGIC                 │  (Workflow)     │
# MAGIC                 └─────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Agent | Role | Uses LLM? |
# MAGIC |-------|------|----------|
# MAGIC | **Orchestrator** | HTTP entry point | No |
# MAGIC | **Planner Agent** | Decision maker, routing | Yes (for classification) |
# MAGIC | **Executor Agents** | Action takers | No (business logic) |
# MAGIC
# MAGIC **Key Insight:**
# MAGIC - **Planner** = Brain (uses LLM to decide)
# MAGIC - **Executors** = Hands (execute specific actions)
# MAGIC - **Orchestrator** = Mouth (talks to external world)
# MAGIC
# MAGIC Next: Planner Pattern deep dive! 📋

# COMMAND ----------

# DBTITLE 1,📋 Topic 8: Planner Pattern - Theory
# MAGIC %md
# MAGIC # 8️⃣ Planner Pattern - Execution Planning 📋
# MAGIC
# MAGIC ## **PLANNER AGENT KA INTERNAL LOGIC! 🧠**
# MAGIC
# MAGIC **Planner Pattern** = Steps ko data structure banake execute karna!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Core Data Structures
# MAGIC
# MAGIC ### **1. PlanStep** — Single Step
# MAGIC
# MAGIC ```python
# MAGIC @dataclass
# MAGIC class PlanStep:
# MAGIC     step_id: str
# MAGIC     action: str  # "classify_intent", "extract_entities", "validate", "route_to_pega"
# MAGIC     executor_type: Optional[str]  # "id_card", "address_change", "inquiry"
# MAGIC     input_data: dict
# MAGIC     output_data: Optional[dict] = None
# MAGIC     status: str = "pending"  # "pending", "running", "completed", "failed"
# MAGIC     error: Optional[str] = None
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC ```python
# MAGIC step1 = PlanStep(
# MAGIC     step_id="step_1",
# MAGIC     action="classify_intent",
# MAGIC     executor_type=None,  # LLM step, no executor yet
# MAGIC     input_data={"email_body": "I need an ID card"},
# MAGIC     status="pending"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **2. ExecutionPlan** — List of Steps
# MAGIC
# MAGIC ```python
# MAGIC @dataclass
# MAGIC class ExecutionPlan:
# MAGIC     plan_id: str
# MAGIC     email_id: str
# MAGIC     steps: List[PlanStep]
# MAGIC     context: dict  # Shared context across steps
# MAGIC     status: str = "pending"
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC ```python
# MAGIC plan = ExecutionPlan(
# MAGIC     plan_id="plan_001",
# MAGIC     email_id="email_123",
# MAGIC     steps=[
# MAGIC         PlanStep(step_id="1", action="classify_intent", ...),
# MAGIC         PlanStep(step_id="2", action="extract_entities", ...),
# MAGIC         PlanStep(step_id="3", action="route_to_pega", ...)
# MAGIC     ],
# MAGIC     context={}  # Shared data
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **3. Shared Context (ctx)** — Data Flow
# MAGIC
# MAGIC **Problem:** Steps need to share data!
# MAGIC
# MAGIC ```
# MAGIC Step 1: Classify intent → Result: "id_card_request"
# MAGIC Step 2: Extract entities → Needs intent from Step 1!
# MAGIC Step 3: Route to PEGA → Needs entities from Step 2!
# MAGIC ```
# MAGIC
# MAGIC **Solution:** Shared `context` dict!
# MAGIC
# MAGIC ```python
# MAGIC context = {
# MAGIC     "email_body": "I need an ID card. HCID: 123456789",
# MAGIC     "intent": None,  # Filled by step 1
# MAGIC     "confidence": None,  # Filled by step 1
# MAGIC     "entities": None,  # Filled by step 2
# MAGIC     "pega_queue": None  # Filled by step 3
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Planner Agent Methods
# MAGIC
# MAGIC ### **Method 1: create_plan()**
# MAGIC
# MAGIC ```python
# MAGIC class PlannerAgent:
# MAGIC     async def create_plan(self, email: EmailRequest) -> ExecutionPlan:
# MAGIC         """
# MAGIC         Create execution plan based on email
# MAGIC         """
# MAGIC         # Step 1: Always classify intent first
# MAGIC         steps = [
# MAGIC             PlanStep(
# MAGIC                 step_id="1",
# MAGIC                 action="classify_intent",
# MAGIC                 executor_type=None,
# MAGIC                 input_data={"email_body": email.body}
# MAGIC             )
# MAGIC         ]
# MAGIC         
# MAGIC         # Steps 2-N will be added after classification
# MAGIC         # (we don't know which executor yet!)
# MAGIC         
# MAGIC         plan = ExecutionPlan(
# MAGIC             plan_id=f"plan_{email.email_id}",
# MAGIC             email_id=email.email_id,
# MAGIC             steps=steps,
# MAGIC             context={"email": email}
# MAGIC         )
# MAGIC         
# MAGIC         return plan
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Method 2: execute_plan()**
# MAGIC
# MAGIC ```python
# MAGIC async def execute_plan(self, plan: ExecutionPlan) -> dict:
# MAGIC     """
# MAGIC     Execute all steps in sequence
# MAGIC     """
# MAGIC     for step in plan.steps:
# MAGIC         # Log step start
# MAGIC         await self.log_step(step, "started")
# MAGIC         
# MAGIC         try:
# MAGIC             # Execute step based on action
# MAGIC             if step.action == "classify_intent":
# MAGIC                 result = await self._classify_intent(step, plan.context)
# MAGIC             elif step.action == "extract_entities":
# MAGIC                 result = await self._extract_entities(step, plan.context)
# MAGIC             elif step.action == "route_to_pega":
# MAGIC                 result = await self._route_to_pega(step, plan.context)
# MAGIC             
# MAGIC             # Update step
# MAGIC             step.output_data = result
# MAGIC             step.status = "completed"
# MAGIC             
# MAGIC             # Update context (for next steps!)
# MAGIC             plan.context.update(result)
# MAGIC             
# MAGIC             # Log success
# MAGIC             await self.log_step(step, "completed")
# MAGIC             
# MAGIC         except Exception as e:
# MAGIC             step.status = "failed"
# MAGIC             step.error = str(e)
# MAGIC             await self.log_step(step, "failed", error=str(e))
# MAGIC             raise
# MAGIC     
# MAGIC     plan.status = "completed"
# MAGIC     return plan.context
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Method 3: _classify_intent()** — LLM Step
# MAGIC
# MAGIC ```python
# MAGIC async def _classify_intent(self, step: PlanStep, ctx: dict) -> dict:
# MAGIC     """
# MAGIC     Step 1: Classify email intent using LLM
# MAGIC     """
# MAGIC     email_body = step.input_data["email_body"]
# MAGIC     
# MAGIC     # Call LLM
# MAGIC     system_prompt = "You are an email classifier. Output JSON."
# MAGIC     user_prompt = f"Classify: {email_body}"
# MAGIC     
# MAGIC     response = await self.llm.ainvoke([
# MAGIC         {"role": "system", "content": system_prompt},
# MAGIC         {"role": "user", "content": user_prompt}
# MAGIC     ])
# MAGIC     
# MAGIC     # Parse response
# MAGIC     result = IntentClassificationResponse.parse_raw(response.content)
# MAGIC     
# MAGIC     # Return data for context
# MAGIC     return {
# MAGIC         "intent": result.intent,
# MAGIC         "confidence": result.confidence,
# MAGIC         "entities": result.entities,
# MAGIC         "reasoning": result.reasoning
# MAGIC     }
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Method 4: decide_executor()** — Routing Logic
# MAGIC
# MAGIC ```python
# MAGIC async def decide_executor(self, intent: str, confidence: float) -> str:
# MAGIC     """
# MAGIC     Decide which executor to use based on intent
# MAGIC     """
# MAGIC     # Check confidence threshold
# MAGIC     if confidence < 0.85:
# MAGIC         return "human_review"  # Low confidence → human
# MAGIC     
# MAGIC     # Route based on intent
# MAGIC     routing_map = {
# MAGIC         "id_card_request": "id_card",
# MAGIC         "address_change": "address_change",
# MAGIC         "inquiry": "inquiry",
# MAGIC         "claim_status": "inquiry"  # Claims also go to inquiry
# MAGIC     }
# MAGIC     
# MAGIC     executor_type = routing_map.get(intent, "inquiry")  # Default to inquiry
# MAGIC     
# MAGIC     return executor_type
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Method 5: add_executor_steps()** — Dynamic Planning
# MAGIC
# MAGIC ```python
# MAGIC async def add_executor_steps(
# MAGIC     self,
# MAGIC     plan: ExecutionPlan,
# MAGIC     executor_type: str
# MAGIC ):
# MAGIC     """
# MAGIC     Add executor-specific steps to plan
# MAGIC     """
# MAGIC     if executor_type == "id_card":
# MAGIC         plan.steps.extend([
# MAGIC             PlanStep(
# MAGIC                 step_id="2",
# MAGIC                 action="extract_entities",
# MAGIC                 executor_type="id_card",
# MAGIC                 input_data={}
# MAGIC             ),
# MAGIC             PlanStep(
# MAGIC                 step_id="3",
# MAGIC                 action="validate_hcid",
# MAGIC                 executor_type="id_card",
# MAGIC                 input_data={}
# MAGIC             ),
# MAGIC             PlanStep(
# MAGIC                 step_id="4",
# MAGIC                 action="send_to_pega",
# MAGIC                 executor_type="id_card",
# MAGIC                 input_data={"queue": "QUEUE_ID_CARD_REQUESTS"}
# MAGIC             )
# MAGIC         ])
# MAGIC     
# MAGIC     elif executor_type == "address_change":
# MAGIC         plan.steps.extend([
# MAGIC             PlanStep(step_id="2", action="extract_address", ...),
# MAGIC             PlanStep(step_id="3", action="validate_address", ...),
# MAGIC             PlanStep(step_id="4", action="call_soa_api", ...)
# MAGIC         ])
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Complete Flow
# MAGIC
# MAGIC ```python
# MAGIC # Usage in FastAPI endpoint
# MAGIC @app.post("/process-email")
# MAGIC async def process_email(email: EmailRequest):
# MAGIC     planner = PlannerAgent(llm=llm)
# MAGIC     
# MAGIC     # Step 1: Create initial plan
# MAGIC     plan = await planner.create_plan(email)
# MAGIC     
# MAGIC     # Step 2: Execute first step (classify intent)
# MAGIC     await planner.execute_plan(plan)
# MAGIC     
# MAGIC     # Step 3: Get intent from context
# MAGIC     intent = plan.context["intent"]
# MAGIC     confidence = plan.context["confidence"]
# MAGIC     
# MAGIC     # Step 4: Decide executor
# MAGIC     executor_type = await planner.decide_executor(intent, confidence)
# MAGIC     
# MAGIC     # Step 5: Add executor steps
# MAGIC     await planner.add_executor_steps(plan, executor_type)
# MAGIC     
# MAGIC     # Step 6: Execute remaining steps
# MAGIC     result = await planner.execute_plan(plan)
# MAGIC     
# MAGIC     # Step 7: Return response
# MAGIC     return EmailResponse(
# MAGIC         email_id=email.email_id,
# MAGIC         status="completed",
# MAGIC         intent=result["intent"],
# MAGIC         queue=result.get("pega_queue")
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary
# MAGIC
# MAGIC | Concept | Purpose |
# MAGIC |---------|--------|
# MAGIC | **PlanStep** | Single action with input/output |
# MAGIC | **ExecutionPlan** | List of steps + shared context |
# MAGIC | **Context (ctx)** | Shared data across steps |
# MAGIC | **create_plan()** | Initial plan creation |
# MAGIC | **execute_plan()** | Run all steps in sequence |
# MAGIC | **decide_executor()** | Routing logic |
# MAGIC | **add_executor_steps()** | Dynamic step addition |
# MAGIC
# MAGIC **Key Insights:**
# MAGIC - ✅ **Data-driven:** Steps = data structures (not just functions)
# MAGIC - ✅ **Observable:** Each step logged separately
# MAGIC - ✅ **Flexible:** Can add steps dynamically
# MAGIC - ✅ **Testable:** Easy to mock/test individual steps
# MAGIC - ✅ **Resumable:** Can save plan state, resume later
# MAGIC
# MAGIC Next: State management & logging! 💾

# COMMAND ----------

# DBTITLE 1,💾 Topic 9 & 🏥 Topic 10: Quick Reference
# MAGIC %md
# MAGIC # 9️⃣ State Management & 1️⃣0️⃣ Domain Knowledge
# MAGIC
# MAGIC ## 💾 Topic 9: State Management (Quick Overview)
# MAGIC
# MAGIC ### **SQLAlchemy + OracleDB**
# MAGIC
# MAGIC ```python
# MAGIC from sqlalchemy import create_engine, Column, String, DateTime
# MAGIC from sqlalchemy.ext.declarative import declarative_base
# MAGIC
# MAGIC Base = declarative_base()
# MAGIC
# MAGIC class EmailTask(Base):
# MAGIC     __tablename__ = "email_tasks"
# MAGIC     
# MAGIC     task_id = Column(String, primary_key=True)
# MAGIC     email_id = Column(String)
# MAGIC     status = Column(String)  # "pending", "processing", "completed", "failed"
# MAGIC     intent = Column(String)
# MAGIC     executor_type = Column(String)
# MAGIC     created_at = Column(DateTime)
# MAGIC     completed_at = Column(DateTime)
# MAGIC
# MAGIC # Save task
# MAGIC async def save_task(email_id: str, status: str):
# MAGIC     task = EmailTask(
# MAGIC         task_id=f"task_{email_id}",
# MAGIC         email_id=email_id,
# MAGIC         status=status
# MAGIC     )
# MAGIC     session.add(task)
# MAGIC     await session.commit()
# MAGIC
# MAGIC # Query task
# MAGIC async def get_task_status(email_id: str):
# MAGIC     task = session.query(EmailTask).filter_by(email_id=email_id).first()
# MAGIC     return task.status
# MAGIC ```
# MAGIC
# MAGIC ### **log_step Pattern (Observability)**
# MAGIC
# MAGIC ```python
# MAGIC async def log_step(
# MAGIC     step: PlanStep,
# MAGIC     status: str,
# MAGIC     error: Optional[str] = None
# MAGIC ):
# MAGIC     """
# MAGIC     Log each step execution for debugging
# MAGIC     """
# MAGIC     log_entry = {
# MAGIC         "timestamp": datetime.now().isoformat(),
# MAGIC         "step_id": step.step_id,
# MAGIC         "action": step.action,
# MAGIC         "status": status,
# MAGIC         "error": error
# MAGIC     }
# MAGIC     
# MAGIC     # Save to database
# MAGIC     await db.save_log(log_entry)
# MAGIC     
# MAGIC     # Also print for debugging
# MAGIC     logger.info(f"Step {step.step_id} ({step.action}): {status}")
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ Track every step execution
# MAGIC - ✅ Debug failures easily
# MAGIC - ✅ Audit trail for compliance
# MAGIC - ✅ Performance monitoring
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏥 Topic 10: Domain Knowledge
# MAGIC
# MAGIC ### **Health Insurance Terminology**
# MAGIC
# MAGIC | Term | Meaning | Example |
# MAGIC |------|---------|--------|
# MAGIC | **HCID** | Health Care ID (9 digits) | 123456789 |
# MAGIC | **Member** | Insurance policy holder | John Doe |
# MAGIC | **ID Card** | Physical/digital insurance card | Requested when lost |
# MAGIC | **Claim** | Request for payment | Medical bill reimbursement |
# MAGIC | **EOB** | Explanation of Benefits | Claim status document |
# MAGIC | **SOA** | Service-Oriented Architecture | Mainframe API |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **PEGA Queues**
# MAGIC
# MAGIC **PEGA** = Workflow management system (routes tasks to humans)
# MAGIC
# MAGIC | Queue | Purpose | Example Email |
# MAGIC |-------|---------|---------------|
# MAGIC | `QUEUE_ID_CARD_REQUESTS` | ID card requests | "I need a new card" |
# MAGIC | `QUEUE_ADDRESS_CHANGES` | Address updates | "I moved to..." |
# MAGIC | `QUEUE_GENERAL_INQUIRY` | Questions | "What's my coverage?" |
# MAGIC | `QUEUE_CLAIM_STATUS` | Claim questions | "Where's my claim?" |
# MAGIC | `QUEUE_HUMAN_REVIEW` | Low confidence | (Uncertain emails) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Intent Types**
# MAGIC
# MAGIC ```python
# MAGIC class IntentType(str, Enum):
# MAGIC     ID_CARD_REQUEST = "id_card_request"
# MAGIC     # User needs ID card (lost, damaged, new member)
# MAGIC     
# MAGIC     ADDRESS_CHANGE = "address_change"
# MAGIC     # User changed address, needs update
# MAGIC     
# MAGIC     INQUIRY = "inquiry"
# MAGIC     # General questions, coverage info
# MAGIC     
# MAGIC     CLAIM_STATUS = "claim_status"
# MAGIC     # Questions about claim processing
# MAGIC     
# MAGIC     UNKNOWN = "unknown"
# MAGIC     # Could not determine intent
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **SOA Mainframe Integration**
# MAGIC
# MAGIC **SOA** = Old mainframe system for address changes
# MAGIC
# MAGIC ```python
# MAGIC async def update_address_soa(hcid: str, new_address: Address):
# MAGIC     """
# MAGIC     Call SOA mainframe API to update address
# MAGIC     """
# MAGIC     payload = {
# MAGIC         "hcid": hcid,
# MAGIC         "street": new_address.street,
# MAGIC         "city": new_address.city,
# MAGIC         "state": new_address.state,
# MAGIC         "zipcode": new_address.zipcode
# MAGIC     }
# MAGIC     
# MAGIC     # SOAP API call (old-school XML)
# MAGIC     response = await soa_client.call(
# MAGIC         endpoint="/UpdateMemberAddress",
# MAGIC         payload=payload
# MAGIC     )
# MAGIC     
# MAGIC     return response.success
# MAGIC ```
# MAGIC
# MAGIC **Why separate from PEGA?**
# MAGIC - Address changes hit mainframe database directly
# MAGIC - ID card requests go through PEGA workflow
# MAGIC - Different systems for historical reasons
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Complete Tech Stack Summary
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │           EMAIL ASSISTANT STACK              │
# MAGIC ├─────────────────────────────────────────┤
# MAGIC │ 🐍 Python 3.11                          │
# MAGIC │ ⚡ FastAPI (REST API)                     │
# MAGIC │ ✅ Pydantic (validation)                  │
# MAGIC │ 🧠 LLM: GPT-4o (via Horizon/Elevance)    │
# MAGIC │ 🔗 LangChain (LLM framework)              │
# MAGIC │ 🔐 OAuth Client-Credentials (TokenManager)│
# MAGIC │ 💾 SQLAlchemy + OracleDB (state)         │
# MAGIC │ 🏛️ PEGA (workflow queues)                 │
# MAGIC │ 💻 SOA Mainframe (address updates)       │
# MAGIC └─────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Interview Topics Covered
# MAGIC
# MAGIC Yeh sab topics GenAI/LLM engineer interviews mein frequently aate hain:
# MAGIC
# MAGIC ✅ Async/await & concurrency  
# MAGIC ✅ Pydantic validation  
# MAGIC ✅ FastAPI REST APIs  
# MAGIC ✅ **LLM structured output (most important!)**  
# MAGIC ✅ LangChain basics  
# MAGIC ✅ OAuth authentication  
# MAGIC ✅ **Agentic architecture (orchestrator + executors)**  
# MAGIC ✅ **Planner pattern (data-driven steps)**  
# MAGIC ✅ State management  
# MAGIC ✅ Domain knowledge (health insurance)  
# MAGIC
# MAGIC **You're now ready for interviews AND understanding the real project! 🎉**

# COMMAND ----------

# DBTITLE 1,🎉 FINAL: Complete System Example
print("🎉 COMPLETE EMAIL ASSISTANT SIMULATION\n")
print("Bringing it all together!\n")

# Simulate complete system
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pydantic import BaseModel

# ========== MODELS (Pydantic) ==========
class IntentType(str, Enum):
    ID_CARD = "id_card_request"
    ADDRESS_CHANGE = "address_change"
    INQUIRY = "inquiry"

class EmailRequest(BaseModel):
    email_id: str
    sender: str
    subject: str
    body: str

# ========== PLANNER PATTERN ==========
@dataclass
class PlanStep:
    step_id: str
    action: str
    status: str = "pending"
    result: Optional[dict] = None

@dataclass
class ExecutionPlan:
    plan_id: str
    email_id: str
    steps: List[PlanStep] = field(default_factory=list)
    context: dict = field(default_factory=dict)

# ========== SIMULATED SERVICES ==========
class SimulatedLLM:
    """Simulates HorizonLlmChat"""
    
    async def ainvoke(self, messages: list) -> dict:
        await asyncio.sleep(0.5)  # Simulate API call
        
        # Extract email body from messages
        user_msg = next(m for m in messages if m["role"] == "user")["content"]
        
        # Simple classification logic
        if "id card" in user_msg.lower():
            return {
                "content": json.dumps({
                    "intent": "id_card_request",
                    "confidence": 0.95,
                    "entities": {"hcid": "123456789"},
                    "reasoning": "Email explicitly mentions ID card"
                })
            }
        elif "address" in user_msg.lower() or "moved" in user_msg.lower():
            return {
                "content": json.dumps({
                    "intent": "address_change",
                    "confidence": 0.91,
                    "entities": {"hcid": "987654321", "new_address": "123 Main St"},
                    "reasoning": "Email mentions address change"
                })
            }
        else:
            return {
                "content": json.dumps({
                    "intent": "inquiry",
                    "confidence": 0.78,
                    "entities": {},
                    "reasoning": "General question"
                })
            }

class PlannerAgent:
    """Orchestrator that plans and executes"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def process_email(self, email: EmailRequest) -> dict:
        """Main entry point"""
        print(f"📧 Processing Email: {email.email_id}")
        print(f"   From: {email.sender}")
        print(f"   Subject: {email.subject}")
        print(f"   Body: {email.body[:60]}..." if len(email.body) > 60 else f"   Body: {email.body}")
        print()
        
        # Step 1: Create plan
        plan = ExecutionPlan(
            plan_id=f"plan_{email.email_id}",
            email_id=email.email_id,
            context={"email": email}
        )
        
        # Step 2: Classify intent
        print("  ▶️ Step 1: Classifying intent with LLM...")
        step1 = PlanStep(step_id="1", action="classify_intent")
        plan.steps.append(step1)
        
        response = await self.llm.ainvoke([
            {"role": "system", "content": "You are an email classifier. Output JSON."},
            {"role": "user", "content": f"Classify: {email.body}"}
        ])
        
        result = json.loads(response["content"])
        step1.result = result
        step1.status = "completed"
        plan.context.update(result)
        
        print(f"     ✅ Intent: {result['intent']}")
        print(f"     ✅ Confidence: {result['confidence']}")
        print()
        
        # Step 3: Route decision
        print("  ▶️ Step 2: Deciding routing...")
        intent = result["intent"]
        confidence = result["confidence"]
        
        if confidence < 0.85:
            executor = "human_review"
            queue = "QUEUE_HUMAN_REVIEW"
            print(f"     ⚠️ Low confidence → Human review")
        else:
            routing = {
                "id_card_request": ("id_card", "QUEUE_ID_CARD_REQUESTS"),
                "address_change": ("address_change", "QUEUE_ADDRESS_CHANGES"),
                "inquiry": ("inquiry", "QUEUE_GENERAL_INQUIRY")
            }
            executor, queue = routing.get(intent, ("inquiry", "QUEUE_GENERAL_INQUIRY"))
            print(f"     ✅ Route to: {executor} executor")
        
        plan.context["executor"] = executor
        plan.context["queue"] = queue
        print(f"     ✅ PEGA Queue: {queue}")
        print()
        
        # Step 4: Execute (simulated)
        print("  ▶️ Step 3: Sending to PEGA...")
        await asyncio.sleep(0.3)  # Simulate API call
        print(f"     ✅ Sent to {queue}")
        print()
        
        print("✅ Email processing completed!\n")
        
        return {
            "email_id": email.email_id,
            "status": "completed",
            "intent": intent,
            "confidence": confidence,
            "executor": executor,
            "queue": queue
        }

# ========== TEST ==========
llm = SimulatedLLM()
planner = PlannerAgent(llm)

test_emails = [
    EmailRequest(
        email_id="email_001",
        sender="john@example.com",
        subject="ID Card Request",
        body="Hi, I need a new ID card. My HCID is 123456789. Thanks!"
    ),
    EmailRequest(
        email_id="email_002",
        sender="jane@example.com",
        subject="Address Change",
        body="I moved to 123 Main St, Austin, TX. HCID: 987654321."
    ),
    EmailRequest(
        email_id="email_003",
        sender="bob@example.com",
        subject="Question",
        body="What is my coverage for dental services?"
    )
]

print("="*70)
print(" SIMULATING EMAIL ASSISTANT - 3 EMAILS")
print("="*70)
print()

results = []
for email in test_emails:
    result = await planner.process_email(email)
    results.append(result)
    print("-"*70)
    print()

print("\n" + "="*70)
print(" SUMMARY")
print("="*70)
for i, result in enumerate(results, 1):
    print(f"\n{i}. Email {result['email_id']}:")
    print(f"   Intent: {result['intent']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Routed to: {result['queue']}")

print("\n🎉 All emails processed successfully!")
print("\n💡 This is how the real email assistant works!")

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Architecture Visualization
print("🎮 Interactive: Email Assistant Architecture Flowchart\n")

def create_architecture_diagram(show_detail_level="high"):
    """
    Create interactive architecture diagram
    """
    
    if show_detail_level == "high":
        # High-level view
        fig = go.Figure()
        
        # Nodes
        nodes = [
            {"name": "Email\nServer", "x": 0.5, "y": 1.0, "color": "#FF6B6B"},
            {"name": "FastAPI\nOrchestrator", "x": 0.5, "y": 0.8, "color": "#4ECDC4"},
            {"name": "Planner\nAgent", "x": 0.5, "y": 0.6, "color": "#95E1D3"},
            {"name": "LLM\n(GPT-4o)", "x": 0.5, "y": 0.4, "color": "#FFA07A"},
            {"name": "Executor\nAgents", "x": 0.5, "y": 0.2, "color": "#FFD93D"},
            {"name": "PEGA\nQueues", "x": 0.5, "y": 0.0, "color": "#6BCB77"}
        ]
        
        # Draw nodes
        for node in nodes:
            fig.add_shape(
                type="rect",
                x0=node["x"]-0.15, y0=node["y"]-0.05,
                x1=node["x"]+0.15, y1=node["y"]+0.05,
                fillcolor=node["color"],
                line=dict(color="black", width=2)
            )
            fig.add_annotation(
                x=node["x"], y=node["y"],
                text=node["name"],
                showarrow=False,
                font=dict(size=12, color="black", family="Arial Black")
            )
        
        # Draw arrows
        arrows = [
            (0.5, 0.95, 0.5, 0.85),  # Email → FastAPI
            (0.5, 0.75, 0.5, 0.65),  # FastAPI → Planner
            (0.5, 0.55, 0.5, 0.45),  # Planner → LLM
            (0.5, 0.35, 0.5, 0.25),  # LLM → Executors
            (0.5, 0.15, 0.5, 0.05),  # Executors → PEGA
        ]
        
        for x0, y0, x1, y1 in arrows:
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="black"
            )
        
        title = "High-Level Architecture"
    
    else:  # detailed
        # Detailed view with all components
        fig = go.Figure()
        
        # More detailed nodes
        nodes = [
            {"name": "Email", "x": 0.5, "y": 1.0, "color": "#FF6B6B"},
            {"name": "FastAPI", "x": 0.5, "y": 0.9, "color": "#4ECDC4"},
            {"name": "Pydantic\nValidation", "x": 0.3, "y": 0.8, "color": "#95E1D3"},
            {"name": "JWT\nAuth", "x": 0.7, "y": 0.8, "color": "#95E1D3"},
            {"name": "Planner\nAgent", "x": 0.5, "y": 0.7, "color": "#FFD93D"},
            {"name": "LLM\nGPT-4o", "x": 0.5, "y": 0.55, "color": "#FFA07A"},
            {"name": "Token\nManager", "x": 0.2, "y": 0.55, "color": "#B4A5A5"},
            {"name": "ID Card\nExecutor", "x": 0.25, "y": 0.4, "color": "#6BCB77"},
            {"name": "Address\nExecutor", "x": 0.5, "y": 0.4, "color": "#6BCB77"},
            {"name": "Inquiry\nExecutor", "x": 0.75, "y": 0.4, "color": "#6BCB77"},
            {"name": "SQLAlchemy\nDB", "x": 0.15, "y": 0.25, "color": "#D4A5A5"},
            {"name": "PEGA\nQueues", "x": 0.5, "y": 0.1, "color": "#FF6B6B"},
            {"name": "SOA\nMainframe", "x": 0.85, "y": 0.25, "color": "#D4A5A5"}
        ]
        
        # Draw all nodes
        for node in nodes:
            fig.add_shape(
                type="rect",
                x0=node["x"]-0.08, y0=node["y"]-0.04,
                x1=node["x"]+0.08, y1=node["y"]+0.04,
                fillcolor=node["color"],
                line=dict(color="black", width=1.5)
            )
            fig.add_annotation(
                x=node["x"], y=node["y"],
                text=node["name"],
                showarrow=False,
                font=dict(size=9, color="black", family="Arial")
            )
        
        # Draw connections (simplified)
        arrows = [
            (0.5, 0.96, 0.5, 0.94),  # Email → FastAPI
            (0.5, 0.86, 0.3, 0.84),  # FastAPI → Pydantic
            (0.5, 0.86, 0.7, 0.84),  # FastAPI → JWT
            (0.5, 0.86, 0.5, 0.74),  # FastAPI → Planner
            (0.5, 0.66, 0.5, 0.59),  # Planner → LLM
            (0.2, 0.51, 0.45, 0.58),  # Token → LLM
            (0.5, 0.51, 0.25, 0.44),  # LLM → ID Card
            (0.5, 0.51, 0.5, 0.44),   # LLM → Address
            (0.5, 0.51, 0.75, 0.44),  # LLM → Inquiry
            (0.25, 0.36, 0.5, 0.14),  # ID Card → PEGA
            (0.5, 0.36, 0.5, 0.14),   # Address → PEGA
            (0.5, 0.36, 0.85, 0.29),  # Address → SOA
            (0.75, 0.36, 0.5, 0.14),  # Inquiry → PEGA
        ]
        
        for x0, y0, x1, y1 in arrows:
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="gray"
            )
        
        title = "Detailed Architecture"
    
    # Update layout
    fig.update_layout(
        title=f"Email Assistant {title}",
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 1.05], showticklabels=False, showgrid=False, zeroline=False),
        height=700,
        template='plotly_white',
        showlegend=False,
        hovermode=False
    )
    
    fig.show()
    
    # Print legend
    if show_detail_level == "high":
        print("\n📊 Flow:")
        print("   1. Email arrives → FastAPI endpoint")
        print("   2. FastAPI → Planner Agent (orchestrator)")
        print("   3. Planner → LLM (classify intent)")
        print("   4. LLM result → Route to appropriate Executor")
        print("   5. Executor → Send to PEGA queue")
    else:
        print("\n📊 Components:")
        print("   🔴 Red: External systems (Email, PEGA)")
        print("   🔵 Blue/Cyan: API layer (FastAPI, Auth, Validation)")
        print("   🟡 Yellow: Planning layer (Planner Agent)")
        print("   🟠 Orange: AI layer (LLM + Token Manager)")
        print("   🟢 Green: Execution layer (Executor Agents)")
        print("   🟪 Purple: Data layer (DB, SOA)")

# Create interactive widget
interact(
    create_architecture_diagram,
    show_detail_level=Dropdown(
        options=[('High-Level', 'high'), ('Detailed', 'detailed')],
        value='high',
        description='Detail:'
    )
);