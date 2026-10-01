# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Layered Agent Architecture
# MAGIC %md
# MAGIC # Layered Agent Architecture
# MAGIC
# MAGIC This notebook implements a production-ready multi-agent system with clear separation of concerns:
# MAGIC
# MAGIC ## Architecture Layers
# MAGIC
# MAGIC **1. Planner / Routing Agent**
# MAGIC * Uses LLM for reasoning and decision-making only
# MAGIC * Decides which agent to call based on user request
# MAGIC * Does NOT execute anything directly
# MAGIC * Output: Agent selection decision
# MAGIC
# MAGIC **2. Execution Controller** (Safety Layer)
# MAGIC * Sits between planner and JSON RPC calls
# MAGIC * Enforces safety mechanisms:
# MAGIC   * **Timeouts**: Prevent hanging operations
# MAGIC   * **Retries**: Only for transient failures
# MAGIC   * **Circuit Breakers**: Stop hammering downed services
# MAGIC   * **Loop Detection**: Prevent infinite cycles
# MAGIC   * **Max Iterations**: Hard limit on execution rounds
# MAGIC * Provides ops visibility through structured logging
# MAGIC
# MAGIC **3. JSON RPC Layer**
# MAGIC * Converts decisions into standardized remote calls
# MAGIC * Handles agent invocation protocol
# MAGIC
# MAGIC **4. Remote Agent Execution**
# MAGIC * Agent's execute method runs tools
# MAGIC * Returns status: `submitted`, `queued`, `in_progress`, `requires_action`, `completed`, `failed`, `cancelled`
# MAGIC * Planner polls and decides next action based on status
# MAGIC
# MAGIC ## Key Principle
# MAGIC **The LLM is used only for reasoning. The controller enforces retries, timeouts, and circuit breakers before invoking remote agents.**

# COMMAND ----------

# DBTITLE 1,Layer 1: Planner (LLM Reasoning Only)
import time
import logging
from collections import defaultdict
from enum import Enum
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)

# Configuration
MAX_ITERATIONS = 5
MAX_SAME_AGENT_CALLS = 3
RPC_TIMEOUT_SECONDS = 30
CIRCUIT_BREAKER_THRESHOLD = 3
MAX_RETRIES = 2


class AgentStatus(Enum):
    """Status returned by remote agent execution"""
    SUBMITTED = "submitted"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    REQUIRES_ACTION = "requires_action"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStopped(Exception):
    """Raised when execution must stop"""
    pass


class Planner:
    """
    Layer 1: Planner / Routing Agent
    
    Uses LLM for reasoning and decision-making ONLY.
    Does NOT execute anything.
    """
    
    def __init__(self, llm_function):
        self.llm = llm_function
    
    def decide_next_action(self, user_request: str, execution_history: List[Dict]) -> Dict[str, Any]:
        """
        Use LLM to decide which agent to call next.
        
        Returns:
            {
                "action": "call_agent" | "finish",
                "agent_name": "data_quality_agent",
                "parameters": {"table": "sales.orders"},
                "reasoning": "Need to check data quality first"
            }
        """
        decision = self.llm(user_request, execution_history)
        
        logging.info(f"Planner decision: {decision.get('action')} - {decision.get('reasoning', '')}")
        
        return decision

# COMMAND ----------

# DBTITLE 1,Layer 2: Execution Controller (Safety Mechanisms)
class CircuitBreaker:
    """
    Prevents repeatedly calling a failing agent.
    Stops hammering a downed service.
    """
    
    def __init__(self, failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD):
        self.failure_threshold = failure_threshold
        self.failures = defaultdict(int)
        self.open_circuits = set()
    
    def record_failure(self, agent_name: str):
        self.failures[agent_name] += 1
        
        if self.failures[agent_name] >= self.failure_threshold:
            self.open_circuits.add(agent_name)
            logging.error(f"Circuit breaker OPEN for agent: {agent_name}")
    
    def record_success(self, agent_name: str):
        self.failures[agent_name] = 0
        if agent_name in self.open_circuits:
            self.open_circuits.remove(agent_name)
            logging.info(f"Circuit breaker CLOSED for agent: {agent_name}")
    
    def is_open(self, agent_name: str) -> bool:
        return agent_name in self.open_circuits


class ExecutionController:
    """
    Layer 2: Execution Controller
    
    Sits between the Planner and JSON RPC layer.
    Enforces safety mechanisms:
    - Timeouts (prevent hanging)
    - Retries (only for transient failures)
    - Circuit breakers (stop hammering downed services)
    - Loop detection (prevent infinite cycles)
    - Max iterations (hard execution limit)
    """
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.agent_call_count = defaultdict(int)
        self.iteration_count = 0
    
    def execute_with_safety(self, agent_name: str, rpc_call_func, parameters: Dict) -> Dict[str, Any]:
        """
        Execute remote agent call with safety mechanisms.
        
        Args:
            agent_name: Name of the agent to call
            rpc_call_func: Function that makes the JSON RPC call
            parameters: Parameters to pass to the agent
        
        Returns:
            Agent execution result with status
        """
        
        # Safety Check 1: Circuit breaker
        if self.circuit_breaker.is_open(agent_name):
            raise ExecutionStopped(
                f"Circuit breaker is OPEN for agent '{agent_name}'. "
                f"Agent has failed {CIRCUIT_BREAKER_THRESHOLD}+ times."
            )
        
        # Safety Check 2: Loop detection
        self.agent_call_count[agent_name] += 1
        if self.agent_call_count[agent_name] > MAX_SAME_AGENT_CALLS:
            raise ExecutionStopped(
                f"Agent '{agent_name}' called {MAX_SAME_AGENT_CALLS}+ times. "
                f"Possible infinite loop detected."
            )
        
        # Execute with timeout and retry logic
        last_exception = None
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                logging.info(f"Calling agent '{agent_name}' (attempt {attempt + 1}/{MAX_RETRIES + 1})")
                
                start_time = time.time()
                
                # Make the JSON RPC call with timeout
                result = self._execute_with_timeout(rpc_call_func, parameters, RPC_TIMEOUT_SECONDS)
                
                duration = time.time() - start_time
                logging.info(f"Agent '{agent_name}' completed in {duration:.2f}s")
                
                # Record success
                self.circuit_breaker.record_success(agent_name)
                
                return result
            
            except TimeoutError as e:
                last_exception = e
                logging.warning(f"Agent '{agent_name}' timed out after {RPC_TIMEOUT_SECONDS}s (attempt {attempt + 1})")
                # Timeout is not retryable - fail immediately
                break
            
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Only retry for transient failures
                if self._is_transient_failure(e):
                    logging.warning(f"Transient failure for agent '{agent_name}': {error_type} - {e} (attempt {attempt + 1})")
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                else:
                    # Permanent failure - don't retry
                    logging.error(f"Permanent failure for agent '{agent_name}': {error_type} - {e}")
                    break
        
        # All retries exhausted or permanent failure
        self.circuit_breaker.record_failure(agent_name)
        raise ExecutionStopped(f"Agent '{agent_name}' failed: {last_exception}")
    
    def _execute_with_timeout(self, func, parameters: Dict, timeout: int) -> Dict:
        """Execute function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation exceeded {timeout}s timeout")
        
        # Set timeout (Unix-based systems)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            result = func(parameters)
            signal.alarm(0)  # Cancel alarm
            return result
        except AttributeError:
            # Fallback for systems without signal (e.g., Windows)
            return func(parameters)
    
    def _is_transient_failure(self, exception: Exception) -> bool:
        """Determine if failure is transient (retryable)"""
        transient_errors = (
            ConnectionError,
            TimeoutError,
        )
        
        error_msg = str(exception).lower()
        transient_keywords = ['timeout', 'connection', 'temporary', 'unavailable', 'throttled']
        
        return (
            isinstance(exception, transient_errors) or
            any(keyword in error_msg for keyword in transient_keywords)
        )
    
    def check_max_iterations(self):
        """Check if max iterations reached"""
        self.iteration_count += 1
        
        if self.iteration_count > MAX_ITERATIONS:
            raise ExecutionStopped(
                f"Exceeded maximum {MAX_ITERATIONS} iterations. "
                f"Possible runaway execution."
            )

# COMMAND ----------

# DBTITLE 1,Layer 3: JSON RPC & Layer 4: Remote Agent
import json
from abc import ABC, abstractmethod


class JsonRpcClient:
    """
    Layer 3: JSON RPC Layer
    
    Converts planner decisions into standardized remote calls.
    Handles agent invocation protocol.
    """
    
    def __init__(self, agent_registry: Dict[str, 'RemoteAgent']):
        self.agent_registry = agent_registry
    
    def call(self, agent_name: str, parameters: Dict) -> Dict[str, Any]:
        """
        Make JSON RPC call to remote agent.
        
        In production, this would:
        - Serialize request to JSON
        - Make HTTP/gRPC call to remote service
        - Deserialize response
        - Handle network errors
        
        For this example, we directly invoke the agent.
        """
        
        if agent_name not in self.agent_registry:
            raise ValueError(f"Agent '{agent_name}' not found in registry")
        
        agent = self.agent_registry[agent_name]
        
        # Simulate JSON RPC request/response
        request = {
            "jsonrpc": "2.0",
            "method": "execute",
            "params": parameters,
            "id": f"req_{int(time.time())}"
        }
        
        logging.info(f"JSON RPC -> {agent_name}: {json.dumps(parameters)}")
        
        # Execute agent
        result = agent.execute(parameters)
        
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": request["id"]
        }
        
        return response["result"]


class RemoteAgent(ABC):
    """
    Layer 4: Remote Agent Base Class
    
    Agent's execute method runs tools and returns status.
    Planner polls and decides next action based on status.
    """
    
    @abstractmethod
    def execute(self, parameters: Dict) -> Dict[str, Any]:
        """
        Execute agent logic.
        
        Returns:
            {
                "status": AgentStatus,
                "result": Any,
                "message": str,
                "requires_action": Optional[Dict]  # If status is REQUIRES_ACTION
            }
        """
        pass


# Example Remote Agents

class DataQualityAgent(RemoteAgent):
    """Agent that checks data quality"""
    
    def execute(self, parameters: Dict) -> Dict[str, Any]:
        table = parameters.get("table")
        
        logging.info(f"DataQualityAgent: Checking quality for {table}")
        
        # Simulate work
        time.sleep(0.5)
        
        # Simulate quality check
        quality_score = 0.92
        
        if quality_score >= 0.9:
            return {
                "status": AgentStatus.COMPLETED.value,
                "result": {
                    "table": table,
                    "quality_score": quality_score,
                    "issues": []
                },
                "message": f"Quality check passed for {table}"
            }
        else:
            return {
                "status": AgentStatus.REQUIRES_ACTION.value,
                "result": {
                    "table": table,
                    "quality_score": quality_score,
                    "issues": ["Missing values in column X"]
                },
                "message": f"Quality issues found in {table}",
                "requires_action": {
                    "action": "fix_data_quality",
                    "details": "Resolve missing values before proceeding"
                }
            }


class PipelineAgent(RemoteAgent):
    """Agent that runs pipelines"""
    
    def execute(self, parameters: Dict) -> Dict[str, Any]:
        pipeline_id = parameters.get("pipeline_id")
        
        logging.info(f"PipelineAgent: Running pipeline {pipeline_id}")
        
        # Simulate work
        time.sleep(0.3)
        
        return {
            "status": AgentStatus.COMPLETED.value,
            "result": {
                "pipeline_id": pipeline_id,
                "records_processed": 1500,
                "duration_seconds": 45
            },
            "message": f"Pipeline {pipeline_id} completed successfully"
        }


class AnalyticsAgent(RemoteAgent):
    """Agent that generates analytics"""
    
    def execute(self, parameters: Dict) -> Dict[str, Any]:
        query = parameters.get("query")
        
        logging.info(f"AnalyticsAgent: Running query: {query}")
        
        # Simulate work
        time.sleep(0.4)
        
        return {
            "status": AgentStatus.COMPLETED.value,
            "result": {
                "query": query,
                "rows_returned": 250,
                "execution_time_ms": 320
            },
            "message": "Query executed successfully"
        }

# COMMAND ----------

# DBTITLE 1,Layered Workflow
# MAGIC %md
# MAGIC ## Layered Workflow
# MAGIC
# MAGIC ```
# MAGIC                     User Request
# MAGIC                          ↓
# MAGIC          ┌───────────────────────────────┐
# MAGIC          │   Layer 1: PLANNER            │
# MAGIC          │   (LLM Reasoning Only)        │
# MAGIC          │   • Decides which agent       │
# MAGIC          │   • Does NOT execute          │
# MAGIC          └───────────────────────────────┘
# MAGIC                          ↓
# MAGIC                   Decision (JSON)
# MAGIC                          ↓
# MAGIC          ┌───────────────────────────────┐
# MAGIC          │ Layer 2: EXECUTION CONTROLLER │
# MAGIC          │ (Safety Mechanisms)           │
# MAGIC          │ • Circuit breaker             │
# MAGIC          │ • Timeout enforcement         │
# MAGIC          │ • Retry (transient only)      │
# MAGIC          │ • Loop detection              │
# MAGIC          │ • Max iterations              │
# MAGIC          └───────────────────────────────┘
# MAGIC                          ↓
# MAGIC                 Safe to proceed?
# MAGIC                          ↓
# MAGIC          ┌───────────────────────────────┐
# MAGIC          │   Layer 3: JSON RPC           │
# MAGIC          │   (Protocol Handler)          │
# MAGIC          │   • Serialize request         │
# MAGIC          │   • Invoke remote agent       │
# MAGIC          │   • Deserialize response      │
# MAGIC          └───────────────────────────────┘
# MAGIC                          ↓
# MAGIC          ┌───────────────────────────────┐
# MAGIC          │   Layer 4: REMOTE AGENT       │
# MAGIC          │   (Execution Layer)           │
# MAGIC          │   • Run tools                 │
# MAGIC          │   • Return status:            │
# MAGIC          │     - submitted               │
# MAGIC          │     - queued                  │
# MAGIC          │     - in_progress             │
# MAGIC          │     - requires_action         │
# MAGIC          │     - completed               │
# MAGIC          │     - failed                  │
# MAGIC          │     - cancelled               │
# MAGIC          └───────────────────────────────┘
# MAGIC                          ↓
# MAGIC               Status returned to Planner
# MAGIC                          ↓
# MAGIC                   ┌──────┴──────┐
# MAGIC               completed      requires_action
# MAGIC                   ↓                 ↓
# MAGIC                 Done         Planner decides
# MAGIC                              next action
# MAGIC ```
# MAGIC
# MAGIC **Key Interview Point:**  
# MAGIC *"The planner uses the LLM only for reasoning, while a controller enforces retries, timeouts, and circuit breakers before invoking remote agents."*

# COMMAND ----------

# DBTITLE 1,Full Orchestration: Multi-Agent System
class MultiAgentOrchestrator:
    """
    Orchestrates the entire multi-agent system.
    Coordinates: Planner -> Controller -> RPC -> Remote Agents
    """
    
    def __init__(self, planner: Planner, controller: ExecutionController, rpc_client: JsonRpcClient):
        self.planner = planner
        self.controller = controller
        self.rpc_client = rpc_client
        self.execution_history = []
    
    def run(self, user_request: str) -> Dict[str, Any]:
        """
        Main orchestration loop.
        
        Returns:
            Final result or raises ExecutionStopped
        """
        
        logging.info(f"=" * 60)
        logging.info(f"Starting multi-agent orchestration")
        logging.info(f"User request: {user_request}")
        logging.info(f"=" * 60)
        
        while True:
            try:
                # Check max iterations
                self.controller.check_max_iterations()
                
                # Layer 1: Planner decides next action (LLM reasoning)
                decision = self.planner.decide_next_action(user_request, self.execution_history)
                
                # Check if done
                if decision.get("action") == "finish":
                    final_result = decision.get("result", "Task completed")
                    logging.info(f"Orchestration complete: {final_result}")
                    return {
                        "status": "success",
                        "result": final_result,
                        "execution_history": self.execution_history
                    }
                
                # Layer 2: Controller enforces safety before RPC call
                agent_name = decision.get("agent_name")
                parameters = decision.get("parameters", {})
                
                # This is where Controller sits BETWEEN Planner and RPC
                result = self.controller.execute_with_safety(
                    agent_name=agent_name,
                    rpc_call_func=lambda params: self.rpc_client.call(agent_name, params),
                    parameters=parameters
                )
                
                # Record execution
                self.execution_history.append({
                    "agent": agent_name,
                    "parameters": parameters,
                    "result": result,
                    "status": result.get("status"),
                    "timestamp": time.time()
                })
                
                # Handle agent status
                status = result.get("status")
                
                if status == AgentStatus.REQUIRES_ACTION.value:
                    logging.warning(f"Agent '{agent_name}' requires action: {result.get('requires_action')}")
                    # Planner will decide next action in next iteration
                
                elif status == AgentStatus.FAILED.value:
                    raise ExecutionStopped(f"Agent '{agent_name}' failed: {result.get('message')}")
                
                elif status == AgentStatus.COMPLETED.value:
                    logging.info(f"Agent '{agent_name}' completed: {result.get('message')}")
                
            except ExecutionStopped as e:
                logging.error(f"Execution stopped: {e}")
                self._escalate_to_human(str(e))
                raise
    
    def _escalate_to_human(self, reason: str):
        """Escalate to human operator"""
        logging.warning(
            f"\n" + "="*60 +
            f"\nHUMAN ESCALATION REQUIRED" +
            f"\nReason: {reason}" +
            f"\nExecution History: {len(self.execution_history)} steps" +
            f"\n" + "="*60
        )
        # In production:
        # - Create Jira ticket
        # - Send Slack/Teams notification
        # - Trigger ServiceNow incident
        # - Put in manual approval queue


# Mock LLM function
def mock_llm(user_request: str, execution_history: List[Dict]) -> Dict[str, Any]:
    """
    Mock LLM that simulates decision-making.
    In production, this would call GPT-4, Claude, etc.
    """
    step = len(execution_history)
    
    if step == 0:
        return {
            "action": "call_agent",
            "agent_name": "data_quality_agent",
            "parameters": {"table": "sales.orders"},
            "reasoning": "First check data quality before analysis"
        }
    elif step == 1:
        return {
            "action": "call_agent",
            "agent_name": "pipeline_agent",
            "parameters": {"pipeline_id": "etl_pipeline_001"},
            "reasoning": "Run ETL pipeline to refresh data"
        }
    elif step == 2:
        return {
            "action": "call_agent",
            "agent_name": "analytics_agent",
            "parameters": {"query": "SELECT SUM(revenue) FROM sales.orders WHERE date >= '2024-01-01'"},
            "reasoning": "Generate analytics report"
        }
    else:
        return {
            "action": "finish",
            "result": "Data pipeline executed, quality verified, analytics generated"
        }


# Example: Run the full multi-agent system
if __name__ == "__main__":
    
    # Setup agent registry
    agent_registry = {
        "data_quality_agent": DataQualityAgent(),
        "pipeline_agent": PipelineAgent(),
        "analytics_agent": AnalyticsAgent()
    }
    
    # Initialize layers
    planner = Planner(llm_function=mock_llm)
    controller = ExecutionController()
    rpc_client = JsonRpcClient(agent_registry=agent_registry)
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(
        planner=planner,
        controller=controller,
        rpc_client=rpc_client
    )
    
    # Run
    try:
        result = orchestrator.run("Analyze sales data and generate report")
        print("\n" + "="*60)
        print("SUCCESS")
        print("="*60)
        print(f"Result: {result['result']}")
        print(f"Steps executed: {len(result['execution_history'])}")
    except ExecutionStopped as e:
        print("\n" + "="*60)
        print("EXECUTION STOPPED")
        print("="*60)
        print(f"Reason: {e}")

# COMMAND ----------

# DBTITLE 1,Interview Talking Points
# MAGIC %md
# MAGIC ## Key Interview Talking Points
# MAGIC
# MAGIC ### Architecture Overview
# MAGIC "In a production multi-agent system, I separate concerns across four layers:
# MAGIC
# MAGIC 1. **Planner/Routing Agent** - Uses the LLM purely for reasoning and decision-making. It decides which agent to call but never executes anything directly.
# MAGIC
# MAGIC 2. **Execution Controller** - This is the safety layer that sits between the planner and remote agent calls. It enforces:
# MAGIC    - **Circuit breakers** to stop hammering a downed service
# MAGIC    - **Timeouts** to prevent operations from hanging
# MAGIC    - **Retries** but only for transient failures (network issues, temporary unavailability)
# MAGIC    - **Loop detection** to catch infinite cycles
# MAGIC    - **Max iteration limits** as a hard execution boundary
# MAGIC
# MAGIC 3. **JSON RPC Layer** - Converts the planner's decision into a standardized remote call. This handles serialization, network transport, and protocol details.
# MAGIC
# MAGIC 4. **Remote Agent Execution** - The agent's execute method runs its tools and returns a status code: submitted, queued, in_progress, requires_action, completed, failed, or cancelled. The planner polls this status and decides what to do next.
# MAGIC
# MAGIC ### Key Principle
# MAGIC **The LLM is used only for reasoning. The controller enforces retries, timeouts, and circuit breakers before invoking remote agents.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Why This Matters
# MAGIC
# MAGIC **Reliability**
# MAGIC - Circuit breakers prevent cascade failures
# MAGIC - Timeouts prevent resource exhaustion
# MAGIC - Retries handle transient issues gracefully
# MAGIC
# MAGIC **Observability**
# MAGIC - Every layer logs structured events
# MAGIC - Execution history provides full audit trail
# MAGIC - Operators can see exactly where failures occur
# MAGIC
# MAGIC **Safety**
# MAGIC - Loop detection catches runaway agents
# MAGIC - Max iterations provide hard boundaries
# MAGIC - Human escalation ensures manual review for edge cases
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Common Interview Questions
# MAGIC
# MAGIC **Q: Why not let the LLM retry failed operations?**  
# MAGIC A: The LLM doesn't know if a failure is transient or permanent. The controller has explicit retry logic that only retries transient errors (network timeouts, connection issues) and fails fast on permanent errors (invalid parameters, authorization failures). This prevents wasted API calls and reduces latency.
# MAGIC
# MAGIC **Q: Where do you put the circuit breaker - in the prompt or the code?**  
# MAGIC A: Always in code, in the execution controller. The circuit breaker needs to track failure counts across requests and maintain state. Prompts are stateless. Plus, the LLM would burn tokens "deciding" whether a circuit is open when this should be a deterministic check.
# MAGIC
# MAGIC **Q: What's the difference between timeout and max iterations?**  
# MAGIC A: Timeout is per-operation (e.g., "this RPC call must complete within 30 seconds"). Max iterations is across the entire workflow (e.g., "the planner can only call agents 5 times total"). Both are needed - timeout prevents hanging, max iterations prevents infinite loops.
# MAGIC
# MAGIC **Q: How do you handle requires_action status?**  
# MAGIC A: The remote agent returns requires_action when it needs human input or another agent to unblock it. The planner receives this status and decides the next action - maybe call a different agent, escalate to human, or abort. The controller doesn't interpret status - it just enforces safety boundaries.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Production Considerations
# MAGIC
# MAGIC **Monitoring & Alerting**
# MAGIC - Track circuit breaker state changes
# MAGIC - Alert on high retry rates
# MAGIC - Monitor timeout frequency
# MAGIC - Dashboard for execution step durations
# MAGIC
# MAGIC **Graceful Degradation**
# MAGIC - When an agent is down (circuit open), route to fallback
# MAGIC - Cache previous results when appropriate
# MAGIC - Provide partial results instead of total failure
# MAGIC
# MAGIC **Cost Control**
# MAGIC - LLM calls are expensive - avoid unnecessary planning rounds
# MAGIC - Cache planner decisions when inputs haven't changed
# MAGIC - Use cheaper models for simple routing decisions
# MAGIC
# MAGIC **Testing Strategy**
# MAGIC - Unit test each layer independently
# MAGIC - Integration tests with mock agents
# MAGIC - Chaos testing: inject random failures to validate circuit breakers
# MAGIC - Load testing: verify timeouts under pressure

# COMMAND ----------

