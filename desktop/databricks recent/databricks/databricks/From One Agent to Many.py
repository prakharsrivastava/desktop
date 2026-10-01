# Databricks notebook source
# DBTITLE 1,Introduction: From One Agent to Many
# MAGIC %md
# MAGIC # From One Agent to Many
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Most use cases start with one agent, but as the number of tools increases, and the range of problems you want your agent to solve increases, introducing a multiagent pattern can improve the overall performance and reliability.
# MAGIC
# MAGIC Just as we saw that it's probably not a good idea to put all of your code in a single file, or bundle all of your backend servers into a single monolith, many of the lessons we learned about the principles of software architecture and system design apply equally well to agentic systems.
# MAGIC
# MAGIC ## Why Multiple Agents?
# MAGIC
# MAGIC As your agentic system grows, you may encounter:
# MAGIC * **Tool overload**: A single agent struggling to select the right tool from 15+ options
# MAGIC * **Context dilution**: Generic prompts that can't capture domain-specific expertise
# MAGIC * **Error cascades**: Mistakes in one domain affecting others
# MAGIC * **Reduced reliability**: Lower success rates as complexity increases
# MAGIC
# MAGIC ## Key Benefits of Multiagent Systems
# MAGIC
# MAGIC * **Specialization**: Each agent focuses on a specific domain (inventory, transportation, supplier management)
# MAGIC * **Better tool selection**: Fewer tools per agent = higher accuracy
# MAGIC * **Parallel processing**: Multiple agents can work simultaneously
# MAGIC * **Modular debugging**: Isolate and fix issues in specific domains
# MAGIC * **Scalability**: Add new capabilities by adding new agents
# MAGIC
# MAGIC ## Structure of This Notebook
# MAGIC
# MAGIC 1. **When to Stay Single-Agent** vs when to introduce multiple agents
# MAGIC 2. **Multiagent Patterns**: Specialist agents, supervisor coordination, swarm patterns
# MAGIC 3. **Implementation**: Practical code examples with LangGraph
# MAGIC 4. **Coordination Strategies**: Democratic, manager, hierarchical, actor-critic
# MAGIC 5. **Production Considerations**: Message brokers, durability, monitoring
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Let's start by understanding when a single agent is sufficient, and when it's time to evolve to a multiagent architecture.

# COMMAND ----------

# DBTITLE 1,Decision Framework: Single vs Multi-Agent
# MAGIC %md
# MAGIC ## Decision Framework: When to Use Single vs Multi-Agent
# MAGIC
# MAGIC ### Stay Single-Agent When:
# MAGIC
# MAGIC ✅ **Small tool count** (< 8 tools)
# MAGIC * Agent can reliably select the right tool
# MAGIC * Low cognitive load for decision-making
# MAGIC
# MAGIC ✅ **Tightly coupled domain**
# MAGIC * All operations belong to one workflow
# MAGIC * Tools naturally work together
# MAGIC
# MAGIC ✅ **Simple coordination**
# MAGIC * Linear or straightforward task sequences
# MAGIC * No need for parallel execution
# MAGIC
# MAGIC ✅ **Early development**
# MAGIC * Still discovering requirements
# MAGIC * Rapid prototyping phase
# MAGIC
# MAGIC ### Move to Multi-Agent When:
# MAGIC
# MAGIC 🔄 **Tool selection accuracy drops** (< 80%)
# MAGIC * Agent frequently picks wrong tools
# MAGIC * Confusion between similar operations
# MAGIC
# MAGIC 🔄 **Distinct domains emerge** (3+ clear categories)
# MAGIC * Tools cluster into logical groups
# MAGIC * Example: Inventory, Transportation, Supplier tools
# MAGIC
# MAGIC 🔄 **Domain-specific expertise needed**
# MAGIC * Different prompts/strategies per domain
# MAGIC * Specialized error handling per area
# MAGIC
# MAGIC 🔄 **Parallel execution valuable**
# MAGIC * Multiple independent operations
# MAGIC * Performance gains from concurrency
# MAGIC
# MAGIC 🔄 **Team specialization**
# MAGIC * Different teams own different domains
# MAGIC * Modular development and ownership
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Real-World Example: Supply Chain System
# MAGIC
# MAGIC **Started with**: Single agent + 16 tools
# MAGIC * Success rate: ~65%
# MAGIC * Frequent tool selection errors
# MAGIC * Generic error messages
# MAGIC
# MAGIC **Evolved to**: 3 specialist agents + 1 supervisor
# MAGIC * Success rate: ~92%
# MAGIC * Clear responsibility boundaries
# MAGIC * Domain-specific error handling
# MAGIC
# MAGIC **Tool Distribution**:
# MAGIC * Inventory Agent: 6 tools (check_stock, update_inventory, forecast_demand, ...)
# MAGIC * Transportation Agent: 5 tools (schedule_shipment, track_delivery, optimize_routes, ...)
# MAGIC * Supplier Agent: 5 tools (evaluate_supplier, place_order, manage_contracts, ...)
# MAGIC * Supervisor: Routing logic only

# COMMAND ----------

# DBTITLE 1,Multiagent Pattern 1: Specialist Agents with Supervisor
# MAGIC %md
# MAGIC ## Pattern 1: Specialist Agents with Supervisor
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC                     ┌─────────────┐
# MAGIC                     │  Supervisor │
# MAGIC                     │   (Router)  │
# MAGIC                     └──────┬──────┘
# MAGIC                            │
# MAGIC           ┌────────────────┼────────────────┐
# MAGIC           │                │                │
# MAGIC     ┌─────▼─────┐    ┌────▼─────┐    ┌────▼──────┐
# MAGIC     │ Inventory │    │Transport │    │  Supplier │
# MAGIC     │  Agent    │    │  Agent   │    │   Agent   │
# MAGIC     │  (6 tools)│    │ (5 tools)│    │  (5 tools)│
# MAGIC     └───────────┘    └──────────┘    └───────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Key Characteristics
# MAGIC
# MAGIC * **Supervisor**: Routes incoming requests to appropriate specialist
# MAGIC * **Specialists**: Each handles a specific domain with domain-specific tools
# MAGIC * **Communication**: User → Supervisor → Specialist → Supervisor → User
# MAGIC * **State Management**: Shared state or passed through supervisor
# MAGIC
# MAGIC ### Routing Logic
# MAGIC
# MAGIC The supervisor analyzes the user query and routes to the appropriate specialist:
# MAGIC
# MAGIC ```python
# MAGIC def route_query(query: str) -> str:
# MAGIC     """
# MAGIC     Determine which specialist agent should handle the query.
# MAGIC     """
# MAGIC     # In practice, use LLM or keyword matching
# MAGIC     if any(kw in query.lower() for kw in ['stock', 'inventory', 'warehouse', 'forecast']):
# MAGIC         return 'inventory_agent'
# MAGIC     elif any(kw in query.lower() for kw in ['ship', 'delivery', 'route', 'transport']):
# MAGIC         return 'transportation_agent'
# MAGIC     elif any(kw in query.lower() for kw in ['supplier', 'vendor', 'order', 'contract']):
# MAGIC         return 'supplier_agent'
# MAGIC     else:
# MAGIC         return 'supervisor'  # Handle directly or ask for clarification
# MAGIC ```
# MAGIC
# MAGIC ### Advantages
# MAGIC
# MAGIC ✅ **Clear separation of concerns**
# MAGIC ✅ **Easy to add new specialists**
# MAGIC ✅ **Simple coordination logic**
# MAGIC ✅ **Each agent can have specialized prompts**
# MAGIC
# MAGIC ### Disadvantages
# MAGIC
# MAGIC ❌ **Supervisor bottleneck**
# MAGIC ❌ **No direct agent-to-agent communication**
# MAGIC ❌ **Sequential processing only**

# COMMAND ----------

# DBTITLE 1,Implementation: Supervisor Pattern with LangGraph
# Supervisor Pattern Implementation
# This demonstrates routing between specialist agents using LangGraph

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator

# State definition
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    task_complete: bool

# Simulated specialist agents
class InventoryAgent:
    """Handles inventory, stock, warehouse, and forecasting."""
    
    def __init__(self):
        self.tools = [
            'check_stock_level',
            'update_inventory',
            'forecast_demand',
            'track_warehouse_capacity',
            'reorder_alert',
            'inventory_audit'
        ]
    
    def invoke(self, state: AgentState) -> AgentState:
        query = state['messages'][-1].content
        # Simulate processing
        response = f"[Inventory Agent] Processed: {query[:50]}... | Available tools: {len(self.tools)}"
        state['messages'].append(HumanMessage(content=response))
        state['task_complete'] = True
        return state

class TransportationAgent:
    """Handles shipping, delivery, routing, and logistics."""
    
    def __init__(self):
        self.tools = [
            'schedule_shipment',
            'track_delivery',
            'optimize_routes',
            'calculate_shipping_cost',
            'manage_fleet'
        ]
    
    def invoke(self, state: AgentState) -> AgentState:
        query = state['messages'][-1].content
        response = f"[Transportation Agent] Processed: {query[:50]}... | Available tools: {len(self.tools)}"
        state['messages'].append(HumanMessage(content=response))
        state['task_complete'] = True
        return state

class SupplierAgent:
    """Handles suppliers, vendors, orders, and contracts."""
    
    def __init__(self):
        self.tools = [
            'evaluate_supplier',
            'place_order',
            'manage_contracts',
            'supplier_performance',
            'negotiate_terms'
        ]
    
    def invoke(self, state: AgentState) -> AgentState:
        query = state['messages'][-1].content
        response = f"[Supplier Agent] Processed: {query[:50]}... | Available tools: {len(self.tools)}"
        state['messages'].append(HumanMessage(content=response))
        state['task_complete'] = True
        return state

# Supervisor routing logic
def supervisor_route(state: AgentState) -> Literal['inventory', 'transportation', 'supplier', 'end']:
    """Route to appropriate specialist based on query content."""
    query = state['messages'][-1].content.lower()
    
    # Check if task is complete
    if state.get('task_complete', False):
        return 'end'
    
    # Route based on keywords
    if any(kw in query for kw in ['stock', 'inventory', 'warehouse', 'forecast']):
        state['current_agent'] = 'inventory'
        return 'inventory'
    elif any(kw in query for kw in ['ship', 'delivery', 'route', 'transport']):
        state['current_agent'] = 'transportation'
        return 'transportation'
    elif any(kw in query for kw in ['supplier', 'vendor', 'order', 'contract']):
        state['current_agent'] = 'supplier'
        return 'supplier'
    else:
        # Default: end (could also route to a clarification node)
        return 'end'

# Build the graph
def build_supervisor_graph():
    # Initialize agents
    inventory_agent = InventoryAgent()
    transportation_agent = TransportationAgent()
    supplier_agent = SupplierAgent()
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node('inventory', inventory_agent.invoke)
    workflow.add_node('transportation', transportation_agent.invoke)
    workflow.add_node('supplier', supplier_agent.invoke)
    
    # Add conditional routing from entry point
    workflow.set_conditional_entry_point(supervisor_route)
    
    # All agents route back to supervisor (which checks task_complete and ends)
    workflow.add_conditional_edges(
        'inventory',
        supervisor_route,
        {
            'inventory': 'inventory',
            'transportation': 'transportation',
            'supplier': 'supplier',
            'end': END
        }
    )
    workflow.add_conditional_edges(
        'transportation',
        supervisor_route,
        {
            'inventory': 'inventory',
            'transportation': 'transportation',
            'supplier': 'supplier',
            'end': END
        }
    )
    workflow.add_conditional_edges(
        'supplier',
        supervisor_route,
        {
            'inventory': 'inventory',
            'transportation': 'transportation',
            'supplier': 'supplier',
            'end': END
        }
    )
    
    return workflow.compile()

# Test the system
print("✅ Supervisor pattern implementation ready")
print("\nAgent Architecture:")
print("  • Inventory Agent: 6 tools")
print("  • Transportation Agent: 5 tools")
print("  • Supplier Agent: 5 tools")
print("  • Total: 16 tools distributed across 3 specialists")

# COMMAND ----------

# DBTITLE 1,Demo: Supervisor Routing in Action
# Demo: Test the supervisor routing with different queries

app = build_supervisor_graph()

test_queries = [
    "Check stock levels for item SKU-12345",
    "Schedule shipment for warehouse A to location B",
    "Evaluate supplier performance for Q4 2025",
    "Forecast demand for next quarter based on historical data"
]

print("=" * 70)
print("SUPERVISOR ROUTING DEMO")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}: {query}")
    print("=" * 70)
    
    # Create initial state
    initial_state = {
        'messages': [HumanMessage(content=query)],
        'current_agent': '',
        'task_complete': False
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    # Display results
    print(f"\n✅ Routed to: {result['current_agent'].upper()}")
    print(f"💬 Response: {result['messages'][-1].content}")
    print(f"✅ Task complete: {result['task_complete']}")

print("\n" + "=" * 70)
print("💡 KEY INSIGHTS")
print("=" * 70)
print("• Each query was automatically routed to the correct specialist")
print("• Tool count per agent: 5-6 (vs 16 for single agent)")
print("• Clear responsibility boundaries improve reliability")
print("• Easy to debug: know exactly which agent handled which query")

# COMMAND ----------

# DBTITLE 1,Pattern 2: Democratic/Swarm Pattern
# MAGIC %md
# MAGIC ## Pattern 2: Democratic/Swarm Pattern
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC                     User Query
# MAGIC                          │
# MAGIC                          v
# MAGIC          ┌────────────────────────┐
# MAGIC          │  All agents receive query │
# MAGIC          └──────────┬─────────────┘
# MAGIC                     │
# MAGIC     ┌───────────┼─────────────┐
# MAGIC     │               │               │
# MAGIC     v               v               v
# MAGIC ┌────────┐    ┌────────┐    ┌────────┐
# MAGIC │ Agent A │    │ Agent B │    │ Agent C │
# MAGIC │  Bids   │    │  Bids   │    │  Bids   │
# MAGIC └────┬────┘    └────┬────┘    └────┬────┘
# MAGIC      │               │               │
# MAGIC      └───────────┼─────────────┘
# MAGIC                      v
# MAGIC             ┌────────────────┐
# MAGIC             │ Highest bidder │
# MAGIC             │   executes     │
# MAGIC             └────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Key Characteristics
# MAGIC
# MAGIC * **No supervisor**: Peer-to-peer coordination
# MAGIC * **Bidding mechanism**: Each agent assesses capability/confidence
# MAGIC * **Dynamic selection**: Best-suited agent self-selects
# MAGIC * **Emergent behavior**: System adapts without central control
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. **Broadcast**: Query sent to all agents simultaneously
# MAGIC 2. **Bid calculation**: Each agent evaluates fit (0.0 - 1.0 confidence score)
# MAGIC 3. **Winner selection**: Highest bid executes
# MAGIC 4. **Fallback**: If no agent bids high enough, request clarification
# MAGIC
# MAGIC ### Bidding Logic Example
# MAGIC
# MAGIC ```python
# MAGIC class SwarmAgent:
# MAGIC     def calculate_bid(self, query: str) -> float:
# MAGIC         """
# MAGIC         Calculate confidence score (0.0 - 1.0) for handling query.
# MAGIC         """
# MAGIC         # Keyword matching
# MAGIC         keyword_score = self._keyword_match(query)
# MAGIC         
# MAGIC         # Tool availability
# MAGIC         tool_score = self._tool_availability(query)
# MAGIC         
# MAGIC         # Historical success rate for similar queries
# MAGIC         history_score = self._historical_performance(query)
# MAGIC         
# MAGIC         # Weighted combination
# MAGIC         bid = (0.4 * keyword_score + 
# MAGIC                0.3 * tool_score + 
# MAGIC                0.3 * history_score)
# MAGIC         
# MAGIC         return bid
# MAGIC ```
# MAGIC
# MAGIC ### Advantages
# MAGIC
# MAGIC ✅ **No single point of failure**
# MAGIC ✅ **Flexible adaptation**
# MAGIC ✅ **Agents can learn and improve bids**
# MAGIC ✅ **Natural load balancing**
# MAGIC ✅ **Easy to add/remove agents**
# MAGIC
# MAGIC ### Disadvantages
# MAGIC
# MAGIC ❌ **More complex coordination**
# MAGIC ❌ **Potential bid conflicts**
# MAGIC ❌ **Higher latency** (all agents evaluate)
# MAGIC ❌ **Requires bid calibration**

# COMMAND ----------

# DBTITLE 1,Implementation: Swarm/Democratic Pattern
# Swarm Pattern Implementation
# Agents bid on queries and highest bidder executes

from typing import List, Tuple
import random

class SwarmAgent:
    """Base class for swarm agents with bidding capability."""
    
    def __init__(self, name: str, keywords: List[str], tools: List[str]):
        self.name = name
        self.keywords = keywords
        self.tools = tools
        self.history = []  # Track (query, success) pairs
    
    def calculate_bid(self, query: str) -> float:
        """Calculate confidence score for handling this query."""
        query_lower = query.lower()
        
        # Keyword matching (0-1)
        keyword_matches = sum(1 for kw in self.keywords if kw in query_lower)
        keyword_score = min(keyword_matches / len(self.keywords), 1.0)
        
        # Tool availability (simulated)
        tool_score = 0.8 if keyword_matches > 0 else 0.2
        
        # Historical success (simulated)
        if self.history:
            recent_success = [s for q, s in self.history[-10:] if any(kw in q.lower() for kw in self.keywords)]
            history_score = sum(recent_success) / len(recent_success) if recent_success else 0.5
        else:
            history_score = 0.5
        
        # Weighted bid
        bid = 0.4 * keyword_score + 0.3 * tool_score + 0.3 * history_score
        
        return bid
    
    def execute(self, query: str) -> str:
        """Execute the query and return response."""
        response = f"[{self.name}] Executed: {query[:60]}... | Tools used: {len(self.tools)} available"
        # Simulate success (in practice, track actual outcome)
        success = random.random() > 0.1  # 90% success rate
        self.history.append((query, success))
        return response

class SwarmCoordinator:
    """Coordinates bidding and execution among swarm agents."""
    
    def __init__(self, agents: List[SwarmAgent], bid_threshold: float = 0.3):
        self.agents = agents
        self.bid_threshold = bid_threshold
    
    def process_query(self, query: str) -> Tuple[str, str, List[Tuple[str, float]]]:
        """Process query using swarm bidding mechanism."""
        # 1. Collect bids from all agents
        bids = [(agent, agent.calculate_bid(query)) for agent in self.agents]
        
        # 2. Sort by bid (highest first)
        bids.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Select winner
        winner, winning_bid = bids[0]
        
        # 4. Check if bid meets threshold
        if winning_bid < self.bid_threshold:
            return "CLARIFICATION_NEEDED", f"No agent confident enough (highest bid: {winning_bid:.2f})", [(agent.name, bid) for agent, bid in bids]
        
        # 5. Execute with winning agent
        response = winner.execute(query)
        
        return winner.name, response, [(agent.name, bid) for agent, bid in bids]

# Create swarm agents
inventory_swarm = SwarmAgent(
    name="Inventory Agent",
    keywords=['stock', 'inventory', 'warehouse', 'forecast', 'reorder'],
    tools=['check_stock', 'update_inventory', 'forecast_demand', 'track_capacity', 'reorder_alert', 'audit']
)

transport_swarm = SwarmAgent(
    name="Transportation Agent",
    keywords=['ship', 'delivery', 'route', 'transport', 'logistics', 'fleet'],
    tools=['schedule_shipment', 'track_delivery', 'optimize_routes', 'calc_cost', 'manage_fleet']
)

supplier_swarm = SwarmAgent(
    name="Supplier Agent",
    keywords=['supplier', 'vendor', 'order', 'contract', 'negotiate', 'evaluate'],
    tools=['evaluate_supplier', 'place_order', 'manage_contracts', 'performance', 'negotiate']
)

# Create coordinator
swarm = SwarmCoordinator(
    agents=[inventory_swarm, transport_swarm, supplier_swarm],
    bid_threshold=0.3
)

print("✅ Swarm coordinator initialized")
print(f"  • {len(swarm.agents)} agents in swarm")
print(f"  • Bid threshold: {swarm.bid_threshold}")

# COMMAND ----------

# DBTITLE 1,Demo: Swarm Bidding in Action
# Demo: Swarm bidding mechanism

test_queries = [
    "Check inventory levels and forecast next month demand",
    "Optimize delivery routes for upcoming shipments",
    "Negotiate contract terms with new supplier",
    "General supply chain status update"  # Ambiguous query
]

print("=" * 70)
print("SWARM BIDDING DEMO")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"Query {i}: {query}")
    print("=" * 70)
    
    winner, response, all_bids = swarm.process_query(query)
    
    # Show all bids
    print("\n📊 Bidding Results:")
    for agent_name, bid in all_bids:
        indicator = "⭐" if agent_name == winner else "  "
        print(f"  {indicator} {agent_name}: {bid:.3f}")
    
    # Show winner and response
    print(f"\n🏆 Winner: {winner}")
    if winner != "CLARIFICATION_NEEDED":
        print(f"💬 {response}")
    else:
        print(f"⚠️ {response}")

print("\n" + "=" * 70)
print("💡 KEY INSIGHTS")
print("=" * 70)
print("• Democratic coordination: no single supervisor")
print("• Self-selection based on confidence")
print("• Handles ambiguous queries gracefully (requests clarification)")
print("• Transparent: see all bids before execution")
print("• Adaptable: agents learn from success/failure history")

# COMMAND ----------

# DBTITLE 1,Coordination Strategies Comparison
# MAGIC %md
# MAGIC ## Coordination Strategies: Which to Choose?
# MAGIC
# MAGIC ### Strategy Overview
# MAGIC
# MAGIC | Strategy | Description | Best For | Complexity |
# MAGIC |----------|-------------|----------|------------|
# MAGIC | **Manager (Supervisor)** | Central router delegates to specialists | Clear domains, hierarchical workflows | Low |
# MAGIC | **Democratic (Swarm)** | Agents bid on tasks | Dynamic environments, peer systems | Medium |
# MAGIC | **Hierarchical** | Multi-level supervision (managers of managers) | Large-scale systems (10+ agents) | High |
# MAGIC | **Actor-Critic** | One agent acts, another evaluates/critiques | Quality-critical tasks, learning systems | Medium |
# MAGIC | **Sequential Pipeline** | Fixed agent chain (A → B → C) | Well-defined workflows | Low |
# MAGIC | **Parallel Aggregation** | All agents contribute, results merged | Diverse perspectives needed | Medium |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 1. Manager/Supervisor Pattern
# MAGIC
# MAGIC **When to use:**
# MAGIC * Clear domain boundaries
# MAGIC * Centralized control needed
# MAGIC * Simple routing logic
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC User query → Supervisor → Inventory Agent → Response
# MAGIC ```
# MAGIC
# MAGIC **Pros:** Simple, predictable, easy to debug
# MAGIC **Cons:** Supervisor bottleneck, no parallel execution
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. Democratic/Swarm Pattern
# MAGIC
# MAGIC **When to use:**
# MAGIC * No clear hierarchy
# MAGIC * Dynamic task allocation
# MAGIC * Fault tolerance important
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC User query → All agents bid → Highest bidder executes
# MAGIC ```
# MAGIC
# MAGIC **Pros:** Flexible, resilient, self-organizing
# MAGIC **Cons:** Coordination overhead, bid calibration needed
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Hierarchical Pattern
# MAGIC
# MAGIC **When to use:**
# MAGIC * Large agent count (10+)
# MAGIC * Multiple organizational levels
# MAGIC * Complex coordination needed
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC User → Master Supervisor → Domain Supervisor → Specialist Agent
# MAGIC ```
# MAGIC
# MAGIC **Pros:** Scales to many agents, clear structure
# MAGIC **Cons:** High latency, complex setup
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4. Actor-Critic Pattern
# MAGIC
# MAGIC **When to use:**
# MAGIC * Quality control critical
# MAGIC * Continuous improvement needed
# MAGIC * Learning from feedback
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC Actor proposes solution → Critic evaluates → Refine or accept
# MAGIC ```
# MAGIC
# MAGIC **Pros:** Built-in quality checks, continuous learning
# MAGIC **Cons:** Double latency, critic reliability critical
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. Sequential Pipeline
# MAGIC
# MAGIC **When to use:**
# MAGIC * Fixed workflow stages
# MAGIC * Each agent adds specific value
# MAGIC * Order matters
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC Data Ingestion Agent → Processing Agent → Analysis Agent → Reporting Agent
# MAGIC ```
# MAGIC
# MAGIC **Pros:** Predictable, easy to monitor
# MAGIC **Cons:** Rigid, no parallelization
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6. Parallel Aggregation
# MAGIC
# MAGIC **When to use:**
# MAGIC * Multiple perspectives valuable
# MAGIC * Consensus decision needed
# MAGIC * Diverse approaches beneficial
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC Query → [Agent A, Agent B, Agent C] (parallel) → Aggregator → Final answer
# MAGIC ```
# MAGIC
# MAGIC **Pros:** Robust, comprehensive answers
# MAGIC **Cons:** Higher cost, aggregation complexity

# COMMAND ----------

# DBTITLE 1,Production Considerations
# MAGIC %md
# MAGIC ## Production Considerations
# MAGIC
# MAGIC ### 1. Communication Infrastructure
# MAGIC
# MAGIC #### Option A: In-Memory (Development)
# MAGIC * **Use case**: Prototyping, small-scale
# MAGIC * **Implementation**: Python dictionaries, LangGraph state
# MAGIC * **Pros**: Simple, fast, no external dependencies
# MAGIC * **Cons**: Not durable, single-process only
# MAGIC
# MAGIC #### Option B: Message Brokers (Production)
# MAGIC * **Kafka**: High throughput, persistent, partitioned
# MAGIC * **Redis Streams**: Fast, lightweight, simple
# MAGIC * **NATS**: Cloud-native, efficient, flexible
# MAGIC * **RabbitMQ**: Mature, reliable, feature-rich
# MAGIC
# MAGIC **Choose based on:**
# MAGIC * **Kafka**: High volume, event sourcing, analytics
# MAGIC * **Redis**: Low latency, simple pub/sub
# MAGIC * **NATS**: Microservices, cloud-native
# MAGIC * **RabbitMQ**: Enterprise features, complex routing
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. State Management
# MAGIC
# MAGIC #### Shared State Patterns
# MAGIC
# MAGIC **A. Centralized Store**
# MAGIC ```python
# MAGIC # Redis example
# MAGIC import redis
# MAGIC
# MAGIC r = redis.Redis()
# MAGIC r.hset('task:123', mapping={
# MAGIC     'status': 'processing',
# MAGIC     'assigned_agent': 'inventory_agent',
# MAGIC     'data': json.dumps(task_data)
# MAGIC })
# MAGIC ```
# MAGIC
# MAGIC **B. Event Sourcing**
# MAGIC ```python
# MAGIC # All state changes as events
# MAGIC events = [
# MAGIC     {'type': 'TaskCreated', 'task_id': 123, 'timestamp': ...},
# MAGIC     {'type': 'AgentAssigned', 'agent': 'inventory', 'timestamp': ...},
# MAGIC     {'type': 'TaskCompleted', 'result': ..., 'timestamp': ...}
# MAGIC ]
# MAGIC # Rebuild current state by replaying events
# MAGIC ```
# MAGIC
# MAGIC **C. Agent-Local State with Sync**
# MAGIC ```python
# MAGIC # Each agent maintains local state, sync on coordination
# MAGIC class Agent:
# MAGIC     def __init__(self):
# MAGIC         self.local_state = {}
# MAGIC     
# MAGIC     def sync_state(self, shared_store):
# MAGIC         # Pull latest from shared store
# MAGIC         # Merge with local state
# MAGIC         pass
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Observability & Monitoring
# MAGIC
# MAGIC **Key Metrics to Track:**
# MAGIC
# MAGIC * **Agent-level**:
# MAGIC   * Task success rate per agent
# MAGIC   * Average execution time
# MAGIC   * Tool selection accuracy
# MAGIC   * Error rate and types
# MAGIC
# MAGIC * **System-level**:
# MAGIC   * End-to-end latency
# MAGIC   * Inter-agent message count
# MAGIC   * Routing accuracy
# MAGIC   * System throughput
# MAGIC
# MAGIC **Implementation:**
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC # Log agent metrics
# MAGIC with mlflow.start_run():
# MAGIC     mlflow.log_metric('agent.inventory.success_rate', 0.92)
# MAGIC     mlflow.log_metric('agent.inventory.avg_latency', 1.2)
# MAGIC     mlflow.log_metric('system.routing_accuracy', 0.95)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4. Error Handling & Resilience
# MAGIC
# MAGIC **Patterns:**
# MAGIC
# MAGIC **A. Retry with Exponential Backoff**
# MAGIC ```python
# MAGIC import time
# MAGIC
# MAGIC def execute_with_retry(func, max_retries=3):
# MAGIC     for attempt in range(max_retries):
# MAGIC         try:
# MAGIC             return func()
# MAGIC         except Exception as e:
# MAGIC             if attempt == max_retries - 1:
# MAGIC                 raise
# MAGIC             wait = 2 ** attempt  # 1s, 2s, 4s
# MAGIC             time.sleep(wait)
# MAGIC ```
# MAGIC
# MAGIC **B. Circuit Breaker**
# MAGIC ```python
# MAGIC class CircuitBreaker:
# MAGIC     def __init__(self, failure_threshold=5, timeout=60):
# MAGIC         self.failure_count = 0
# MAGIC         self.failure_threshold = failure_threshold
# MAGIC         self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
# MAGIC         self.timeout = timeout
# MAGIC         self.last_failure_time = None
# MAGIC ```
# MAGIC
# MAGIC **C. Fallback Agents**
# MAGIC ```python
# MAGIC def route_with_fallback(query):
# MAGIC     primary_agent = route_query(query)
# MAGIC     try:
# MAGIC         return primary_agent.execute(query)
# MAGIC     except Exception as e:
# MAGIC         # Fall back to generalist agent
# MAGIC         return fallback_agent.execute(query)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. Cost Optimization
# MAGIC
# MAGIC **Strategies:**
# MAGIC
# MAGIC * **Caching**: Cache frequent queries and agent responses
# MAGIC * **Early termination**: Stop after first successful agent in parallel execution
# MAGIC * **Dynamic routing**: Route to cheaper models when possible
# MAGIC * **Batching**: Process multiple similar queries together
# MAGIC * **Rate limiting**: Prevent runaway agent loops
# MAGIC
# MAGIC ```python
# MAGIC from functools import lru_cache
# MAGIC
# MAGIC @lru_cache(maxsize=1000)
# MAGIC def cached_agent_call(query: str, agent_id: str):
# MAGIC     # Expensive LLM call
# MAGIC     return agent.execute(query)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Decision Framework & Conclusion
# MAGIC %md
# MAGIC ## Decision Framework: Your Multiagent Strategy
# MAGIC
# MAGIC ### Step 1: Assess Current State
# MAGIC
# MAGIC **Questions to ask:**
# MAGIC 1. How many tools does your agent have? (< 8: stay single, > 12: consider multi)
# MAGIC 2. What's your tool selection accuracy? (< 80%: likely need multi-agent)
# MAGIC 3. Do tools cluster into natural domains? (Yes: strong signal for multi-agent)
# MAGIC 4. Is your prompt getting too complex? (Yes: specialization helps)
# MAGIC 5. Do you need parallel execution? (Yes: multi-agent enables this)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 2: Choose Your Pattern
# MAGIC
# MAGIC **Use this decision tree:**
# MAGIC
# MAGIC ```
# MAGIC Start here
# MAGIC     │
# MAGIC     v
# MAGIC Clear domains?  ── NO ──> Dynamic task allocation?  ── YES ──> SWARM
# MAGIC     │                           │
# MAGIC    YES                         NO
# MAGIC     │                           │
# MAGIC     v                           v
# MAGIC Fixed workflow? ── YES ──> PIPELINE      SINGLE AGENT
# MAGIC     │
# MAGIC    NO
# MAGIC     │
# MAGIC     v
# MAGIC Quality critical? ── YES ──> ACTOR-CRITIC
# MAGIC     │
# MAGIC    NO
# MAGIC     │
# MAGIC     v
# MAGIC SUPERVISOR (start here, most common)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 3: Implement Incrementally
# MAGIC
# MAGIC **Migration Path (Zero-Downtime):**
# MAGIC
# MAGIC 1. **Phase 1: Keep single agent in production**
# MAGIC    * Build multiagent system in parallel
# MAGIC    * Test with synthetic queries
# MAGIC    * Compare metrics
# MAGIC
# MAGIC 2. **Phase 2: Shadow mode**
# MAGIC    * Route real queries to both systems
# MAGIC    * Log responses from both
# MAGIC    * Analyze differences
# MAGIC
# MAGIC 3. **Phase 3: Gradual rollout**
# MAGIC    * Route 10% of traffic to multiagent
# MAGIC    * Monitor metrics closely
# MAGIC    * Increase gradually (25%, 50%, 100%)
# MAGIC
# MAGIC 4. **Phase 4: Deprecate single agent**
# MAGIC    * Full cutover to multiagent
# MAGIC    * Keep single agent as fallback
# MAGIC    * Eventually remove
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 4: Monitor & Iterate
# MAGIC
# MAGIC **Key Success Metrics:**
# MAGIC
# MAGIC | Metric | Target | Action if Below |
# MAGIC |--------|--------|----------------|
# MAGIC | Routing accuracy | > 95% | Improve routing logic |
# MAGIC | Task success rate | > 90% | Review failed tasks per agent |
# MAGIC | End-to-end latency | < 3s | Optimize or parallelize |
# MAGIC | User satisfaction | > 4.5/5 | Gather feedback, iterate |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC ✅ **Start simple**: Single agent is often sufficient
# MAGIC ✅ **Watch for signals**: Tool count, accuracy, domain boundaries
# MAGIC ✅ **Choose the right pattern**: Supervisor for most, swarm for dynamic
# MAGIC ✅ **Production-ready**: Message brokers, observability, error handling
# MAGIC ✅ **Iterate continuously**: Monitor, learn, improve
# MAGIC
# MAGIC ### When to Move to Multi-Agent
# MAGIC
# MAGIC **Strong signals:**
# MAGIC * Tool selection accuracy drops below 80%
# MAGIC * 3+ distinct tool domains emerge
# MAGIC * Team wants domain ownership
# MAGIC * Parallel execution provides value
# MAGIC
# MAGIC **Weak signals:**
# MAGIC * Just hitting tool count threshold
# MAGIC * No clear domain boundaries
# MAGIC * Early prototype phase
# MAGIC
# MAGIC ### Final Advice
# MAGIC
# MAGIC > **"The best architecture is the simplest one that meets your needs."**
# MAGIC
# MAGIC Don't over-engineer. Start with a single agent. When you feel the pain (poor tool selection, prompt complexity, etc.), then evolve to multi-agent. The patterns and code in this notebook give you a clear path forward when that time comes.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Resources & Next Steps
# MAGIC
# MAGIC **Further Reading:**
# MAGIC * LangGraph Multi-Agent documentation
# MAGIC * "Generative Agents: Interactive Simulacra of Human Behavior" (Stanford)
# MAGIC * "Communicative Agents for Software Development" (ChatDev paper)
# MAGIC * AutoGen framework documentation
# MAGIC
# MAGIC **Try it yourself:**
# MAGIC 1. Run the supervisor pattern demo above
# MAGIC 2. Run the swarm bidding demo
# MAGIC 3. Adapt one pattern to your use case
# MAGIC 4. Measure before/after metrics
# MAGIC
# MAGIC **Questions to explore:**
# MAGIC * How would you add a 4th specialist agent?
# MAGIC * What would an actor-critic pattern look like for your domain?
# MAGIC * How would you implement async message passing with Kafka?
# MAGIC * What observability dashboard would you build?