# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introduction: Validation and Measurement
# MAGIC %md
# MAGIC # Validation and Measurement for Agentic Systems
# MAGIC
# MAGIC ## Why Measurement Matters
# MAGIC
# MAGIC Building agentic systems is easier than ever, but **measuring their quality remains challenging**. Without rigorous evaluation:
# MAGIC * You can't tell if changes are improvements
# MAGIC * Subtle regressions go unnoticed
# MAGIC * Production failures surprise you
# MAGIC * Trust erodes over time
# MAGIC
# MAGIC **Key insight**: Measurement is the keystone of reliable agent development.
# MAGIC
# MAGIC ## What We'll Cover
# MAGIC
# MAGIC 1. **Measurement Fundamentals**: Defining metrics, building evaluation sets
# MAGIC 2. **Component Testing**: Validating tools, planning, memory, learning
# MAGIC 3. **End-to-End Evaluation**: Testing complete workflows
# MAGIC 4. **Quality Metrics**: Consistency, coherence, accuracy
# MAGIC 5. **Deployment Readiness**: Quality gates and monitoring
# MAGIC
# MAGIC ## Running Example: E-Commerce Customer Support
# MAGIC
# MAGIC Throughout this notebook, we'll build and test a customer support agent handling:
# MAGIC * **Simple refunds**: Customer reports damaged item, agent issues refund
# MAGIC * **Multi-item orders**: Only refund the damaged item, not entire order
# MAGIC * **Cancellations**: Handle timing (can't cancel delivered orders)
# MAGIC * **Address changes**: Update shipping address for pending orders
# MAGIC
# MAGIC Let's start with the basics of measurement.

# COMMAND ----------

# DBTITLE 1,Measurement Fundamentals
# MAGIC %md
# MAGIC ## Measurement Fundamentals
# MAGIC
# MAGIC ### The Challenge with AI Agents
# MAGIC
# MAGIC Unlike traditional software:
# MAGIC * **Non-deterministic**: Same input can produce different outputs
# MAGIC * **Context-dependent**: Quality varies by scenario
# MAGIC * **Multi-faceted**: Must evaluate correctness, coherence, and user experience
# MAGIC * **Evolving**: Models and behaviors change over time
# MAGIC
# MAGIC ### Key Principles
# MAGIC
# MAGIC 1. **Define clear objectives**: What does success look like?
# MAGIC 2. **Choose appropriate metrics**: Quantify what matters
# MAGIC 3. **Build diverse test sets**: Cover common cases and edge cases
# MAGIC 4. **Automate evaluation**: Integrate into CI/CD
# MAGIC 5. **Monitor continuously**: Track metrics over time
# MAGIC
# MAGIC ### Types of Metrics
# MAGIC
# MAGIC **Functional Metrics** (Did it work?)
# MAGIC * Task success rate
# MAGIC * Tool selection accuracy
# MAGIC * Parameter correctness
# MAGIC
# MAGIC **Quality Metrics** (How well did it work?)
# MAGIC * Response coherence
# MAGIC * Consistency across turns
# MAGIC * Hallucination rate
# MAGIC
# MAGIC **Performance Metrics** (How efficiently?)
# MAGIC * Latency
# MAGIC * Cost per interaction
# MAGIC * Token usage
# MAGIC
# MAGIC **User Experience Metrics**
# MAGIC * User satisfaction scores
# MAGIC * Task completion time
# MAGIC * Escalation rate

# COMMAND ----------

# DBTITLE 1,Example: Customer Support Agent Setup
# Setup: Simple Customer Support Agent
# This agent handles refunds, cancellations, and address changes

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class Order:
    """Represents a customer order."""
    order_id: str
    status: str  # 'Pending', 'Shipped', 'Delivered'
    total: float
    items: List[Dict[str, Any]]
    shipping_address: Optional[str] = None
    delivered_at: Optional[str] = None

class CustomerSupportTools:
    """Tools available to the support agent."""
    
    def __init__(self):
        self.actions_log = []  # Track all actions taken
    
    def issue_refund(self, order_id: str, amount: float, reason: str = "") -> Dict:
        """Issue a refund for an order or part of an order."""
        action = {
            'tool': 'issue_refund',
            'params': {'order_id': order_id, 'amount': amount, 'reason': reason}
        }
        self.actions_log.append(action)
        return {'success': True, 'refund_id': f'REF-{order_id[:6]}'}
    
    def cancel_order(self, order_id: str, reason: str = "") -> Dict:
        """Cancel a pending order."""
        action = {
            'tool': 'cancel_order',
            'params': {'order_id': order_id, 'reason': reason}
        }
        self.actions_log.append(action)
        return {'success': True, 'cancellation_id': f'CAN-{order_id[:6]}'}
    
    def update_shipping_address(self, order_id: str, new_address: str) -> Dict:
        """Update shipping address for a pending order."""
        action = {
            'tool': 'update_shipping_address',
            'params': {'order_id': order_id, 'new_address': new_address}
        }
        self.actions_log.append(action)
        return {'success': True, 'updated_order_id': order_id}
    
    def get_order_details(self, order_id: str) -> Dict:
        """Retrieve order details."""
        action = {
            'tool': 'get_order_details',
            'params': {'order_id': order_id}
        }
        self.actions_log.append(action)
        return {'success': True, 'order_id': order_id}

# Example order for testing
example_order = Order(
    order_id="A89268",
    status="Delivered",
    total=39.99,
    items=[
        {"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99},
        {"sku": "TSHIRT-S", "name": "T-Shirt (Small)", "qty": 1, "unit_price": 20.00}
    ],
    delivered_at="2025-05-15"
)

print("✅ Customer support agent tools initialized")
print(f"\nExample order: {example_order.order_id}")
print(f"  Status: {example_order.status}")
print(f"  Items: {len(example_order.items)}")
print(f"  Total: ${example_order.total}")

# COMMAND ----------

# DBTITLE 1,Building Evaluation Datasets
# MAGIC %md
# MAGIC ## Building Evaluation Datasets
# MAGIC
# MAGIC ### Anatomy of a Test Case
# MAGIC
# MAGIC Each evaluation example should include:
# MAGIC
# MAGIC 1. **Input State**: Order data, conversation history
# MAGIC 2. **Expected Behavior**: Which tools to call, with what parameters
# MAGIC 3. **Expected Output**: Key phrases that should appear in response
# MAGIC
# MAGIC ### Example Structure
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "test_id": "refund_damaged_item_001",
# MAGIC   "order": {...},
# MAGIC   "conversation": [
# MAGIC     {"role": "customer", "content": "My mug arrived cracked..."},
# MAGIC     {"role": "assistant", "content": "I'm sorry to hear that..."}
# MAGIC   ],
# MAGIC   "expected": {
# MAGIC     "tool_calls": [
# MAGIC       {"tool": "issue_refund", "params": {"order_id": "A89268", "amount": 19.99}}
# MAGIC     ],
# MAGIC     "response_should_contain": ["refund", "processed", "business days"]
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ### Strategies for Building Test Sets
# MAGIC
# MAGIC **1. Start with Happy Paths**
# MAGIC * Most common user requests
# MAGIC * Clear, unambiguous scenarios
# MAGIC * Expected to work perfectly
# MAGIC
# MAGIC **2. Add Edge Cases**
# MAGIC * Multi-item orders (refund one item)
# MAGIC * Timing issues (can't cancel delivered order)
# MAGIC * Ambiguous requests
# MAGIC
# MAGIC **3. Include Adversarial Cases**
# MAGIC * Malformed inputs
# MAGIC * Conflicting information
# MAGIC * Out-of-scope requests
# MAGIC
# MAGIC **4. Cover Regression Cases**
# MAGIC * Previously fixed bugs
# MAGIC * Known failure modes
# MAGIC * Performance-sensitive scenarios

# COMMAND ----------

# DBTITLE 1,Example: Building Test Cases
# Build a collection of test cases for our customer support agent

test_cases = [
    {
        "test_id": "refund_001_simple",
        "description": "Customer reports damaged mug, requests refund",
        "order": {
            "order_id": "A89268",
            "status": "Delivered",
            "total": 19.99,
            "items": [{"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}]
        },
        "conversation": [
            {"role": "customer", "content": "My coffee mug arrived cracked. Can I get a refund?"},
        ],
        "expected": {
            "tool_calls": [
                {"tool": "issue_refund", "params": {"order_id": "A89268", "amount": 19.99}}
            ],
            "response_should_contain": ["refund", "processed"]
        }
    },
    {
        "test_id": "refund_002_multiitem",
        "description": "Refund only damaged item from multi-item order",
        "order": {
            "order_id": "A89268",
            "status": "Delivered",
            "total": 39.99,
            "items": [
                {"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99},
                {"sku": "TSHIRT-S", "name": "T-Shirt (Small)", "qty": 1, "unit_price": 20.00}
            ]
        },
        "conversation": [
            {"role": "customer", "content": "The mug in my order arrived cracked. Can I get a refund for just the mug?"},
        ],
        "expected": {
            "tool_calls": [
                {"tool": "issue_refund", "params": {"order_id": "A89268", "amount": 19.99}}  # Not 39.99!
            ],
            "response_should_contain": ["refund", "mug", "19.99"]
        }
    },
    {
        "test_id": "cancel_003_delivered",
        "description": "Cannot cancel delivered order",
        "order": {
            "order_id": "A89268",
            "status": "Delivered",
            "total": 19.99,
            "items": [{"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}],
            "delivered_at": "2025-05-15"
        },
        "conversation": [
            {"role": "customer", "content": "I want to cancel my order A89268"},
        ],
        "expected": {
            "tool_calls": [],  # Should NOT call cancel_order
            "response_should_contain": ["already delivered", "cannot cancel", "refund"]
        }
    },
    {
        "test_id": "address_004_pending",
        "description": "Update address for pending order",
        "order": {
            "order_id": "A12345",
            "status": "Pending",
            "total": 19.99,
            "items": [{"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}],
            "shipping_address": "123 Old St"
        },
        "conversation": [
            {"role": "customer", "content": "Can I change my shipping address to 456 New Ave?"},
        ],
        "expected": {
            "tool_calls": [
                {"tool": "update_shipping_address", "params": {"order_id": "A12345", "new_address": "456 New Ave"}}
            ],
            "response_should_contain": ["updated", "456 New Ave"]
        }
    }
]

print(f"✅ Created {len(test_cases)} test cases")
print("\nTest Coverage:")
for tc in test_cases:
    print(f"  • {tc['test_id']}: {tc['description']}")

# COMMAND ----------

# DBTITLE 1,Component Testing: Tools
# MAGIC %md
# MAGIC ## Component Testing: Validating Tools
# MAGIC
# MAGIC Tools are the building blocks of agent actions. Each tool must be tested independently to ensure:
# MAGIC * Correct outputs for valid inputs
# MAGIC * Graceful error handling for invalid inputs
# MAGIC * Expected side effects (logging, state changes)
# MAGIC * Performance within acceptable bounds
# MAGIC
# MAGIC ### Tool Testing Checklist
# MAGIC
# MAGIC ✅ **Happy path**: Tool works as expected with valid inputs
# MAGIC ✅ **Edge cases**: Handles boundary conditions (empty strings, null values, extremes)
# MAGIC ✅ **Error cases**: Returns appropriate errors for invalid inputs
# MAGIC ✅ **Idempotency**: Multiple calls with same input produce consistent results (when applicable)
# MAGIC ✅ **Performance**: Completes within acceptable time limits

# COMMAND ----------

# DBTITLE 1,Example: Tool Unit Tests
# Unit tests for customer support tools

def test_issue_refund_tool():
    """Test the issue_refund tool with various inputs."""
    tools = CustomerSupportTools()
    
    # Test 1: Valid refund
    result = tools.issue_refund(order_id="A89268", amount=19.99, reason="Damaged item")
    assert result['success'] == True
    assert 'refund_id' in result
    print("✅ Test 1 passed: Valid refund")
    
    # Test 2: Check action was logged
    assert len(tools.actions_log) == 1
    assert tools.actions_log[0]['tool'] == 'issue_refund'
    assert tools.actions_log[0]['params']['amount'] == 19.99
    print("✅ Test 2 passed: Action logged correctly")
    
    # Test 3: Partial refund
    result = tools.issue_refund(order_id="A89268", amount=10.00, reason="Partial damage")
    assert result['success'] == True
    print("✅ Test 3 passed: Partial refund")
    
    # Test 4: Edge case - zero amount (should still process)
    result = tools.issue_refund(order_id="A89268", amount=0.0, reason="Goodwill gesture")
    assert result['success'] == True
    print("✅ Test 4 passed: Zero-amount refund (edge case)")
    
    print(f"\n✅ All tool tests passed! Total actions logged: {len(tools.actions_log)}")

test_issue_refund_tool()

# COMMAND ----------

# DBTITLE 1,Component Testing: Planning & Routing
# MAGIC %md
# MAGIC ## Component Testing: Planning & Routing
# MAGIC
# MAGIC The planning module decides:
# MAGIC * Which tool(s) to call
# MAGIC * In what order
# MAGIC * With what parameters
# MAGIC
# MAGIC Planning errors are among the most critical failures in agentic systems.
# MAGIC
# MAGIC ### Planning Test Metrics
# MAGIC
# MAGIC **Tool Recall**: Did the agent call all necessary tools?
# MAGIC ```
# MAGIC recall = tools_called_correctly / tools_that_should_be_called
# MAGIC ```
# MAGIC
# MAGIC **Tool Precision**: Did the agent avoid calling unnecessary tools?
# MAGIC ```
# MAGIC precision = tools_called_correctly / total_tools_called
# MAGIC ```
# MAGIC
# MAGIC **Parameter Accuracy**: Were tool parameters correct?
# MAGIC ```
# MAGIC param_accuracy = correct_parameters / total_parameters
# MAGIC ```
# MAGIC
# MAGIC ### Common Planning Failures
# MAGIC
# MAGIC 1. **Wrong tool**: Calls `cancel_order` instead of `issue_refund`
# MAGIC 2. **Wrong parameters**: Refunds entire order instead of single item
# MAGIC 3. **Missing tool**: Forgets to call required tool
# MAGIC 4. **Extra tool**: Calls unnecessary tool (e.g., `get_order_details` multiple times)

# COMMAND ----------

# DBTITLE 1,Planning Metrics Implementation
# Metrics for evaluating planning quality

def compute_tool_metrics(predicted_tools: List[str], expected_tools: List[Dict]) -> Dict[str, float]:
    """
    Compute tool recall and precision.
    
    Args:
        predicted_tools: List of tool names that were called
        expected_tools: List of expected tool calls with params
    
    Returns:
        Dictionary with recall and precision scores
    """
    if not expected_tools:
        return {"tool_recall": 1.0, "tool_precision": 1.0}
    
    expected_names = [t['tool'] for t in expected_tools]
    
    pred_set = set(predicted_tools)
    exp_set = set(expected_names)
    
    # True positives: tools we expected and got
    true_positives = len(pred_set & exp_set)
    
    # Recall: fraction of expected tools we actually called
    recall = true_positives / len(exp_set) if exp_set else 1.0
    
    # Precision: fraction of called tools that were expected
    precision = true_positives / len(pred_set) if pred_set else 0.0
    
    return {
        "tool_recall": recall,
        "tool_precision": precision
    }

def compute_param_accuracy(predicted_calls: List[Dict], expected_calls: List[Dict]) -> float:
    """
    Compute parameter accuracy: fraction of tool calls with correct parameters.
    
    Args:
        predicted_calls: List of {"tool": name, "params": {...}}
        expected_calls: List of {"tool": name, "params": {...}}
    
    Returns:
        Accuracy score from 0.0 to 1.0
    """
    if not expected_calls:
        return 1.0
    
    matched = 0
    for expected in expected_calls:
        for predicted in predicted_calls:
            if (predicted['tool'] == expected['tool'] and 
                predicted['params'] == expected['params']):
                matched += 1
                break
    
    return matched / len(expected_calls)

# Test the metrics
print("Example 1: Perfect planning")
pred = ['issue_refund']
exp = [{'tool': 'issue_refund', 'params': {'order_id': 'A89268', 'amount': 19.99}}]
pred_calls = [{'tool': 'issue_refund', 'params': {'order_id': 'A89268', 'amount': 19.99}}]

metrics = compute_tool_metrics(pred, exp)
metrics['param_accuracy'] = compute_param_accuracy(pred_calls, exp)

print(f"  Tool Recall: {metrics['tool_recall']:.2f}")
print(f"  Tool Precision: {metrics['tool_precision']:.2f}")
print(f"  Param Accuracy: {metrics['param_accuracy']:.2f}")

print("\nExample 2: Wrong amount (parameter error)")
pred_calls_wrong = [{'tool': 'issue_refund', 'params': {'order_id': 'A89268', 'amount': 39.99}}]  # Wrong!
metrics2 = compute_tool_metrics(pred, exp)
metrics2['param_accuracy'] = compute_param_accuracy(pred_calls_wrong, exp)

print(f"  Tool Recall: {metrics2['tool_recall']:.2f}")
print(f"  Tool Precision: {metrics2['tool_precision']:.2f}")
print(f"  Param Accuracy: {metrics2['param_accuracy']:.2f}  ❌ Parameter mismatch!")

print("\nExample 3: Missing required tool")
pred_missing = []  # Didn't call anything
metrics3 = compute_tool_metrics(pred_missing, exp)
metrics3['param_accuracy'] = compute_param_accuracy([], exp)

print(f"  Tool Recall: {metrics3['tool_recall']:.2f}  ❌ Forgot to call tool!")
print(f"  Tool Precision: {metrics3['tool_precision']:.2f}")
print(f"  Param Accuracy: {metrics3['param_accuracy']:.2f}")

# COMMAND ----------

# DBTITLE 1,End-to-End Evaluation
# MAGIC %md
# MAGIC ## End-to-End Evaluation
# MAGIC
# MAGIC Component tests validate individual pieces. End-to-end tests validate the **complete system** in realistic scenarios.
# MAGIC
# MAGIC ### What E2E Tests Validate
# MAGIC
# MAGIC 1. **Complete workflows**: From user request to final response
# MAGIC 2. **Tool integration**: Multiple tools working together
# MAGIC 3. **Context handling**: Agent maintains state across turns
# MAGIC 4. **Error recovery**: Agent handles failures gracefully
# MAGIC
# MAGIC ### E2E Test Structure
# MAGIC
# MAGIC ```python
# MAGIC def evaluate_agent(test_case):
# MAGIC     # 1. Initialize with test data
# MAGIC     order = test_case['order']
# MAGIC     messages = test_case['conversation']
# MAGIC     
# MAGIC     # 2. Run the agent
# MAGIC     result = agent.process(order, messages)
# MAGIC     
# MAGIC     # 3. Extract agent's actions and response
# MAGIC     tools_called = extract_tool_calls(result)
# MAGIC     final_message = extract_response(result)
# MAGIC     
# MAGIC     # 4. Compute metrics
# MAGIC     metrics = {
# MAGIC         'tool_recall': compute_tool_recall(tools_called, expected),
# MAGIC         'tool_precision': compute_tool_precision(tools_called, expected),
# MAGIC         'param_accuracy': compute_param_accuracy(tools_called, expected),
# MAGIC         'response_quality': check_response_phrases(final_message, expected)
# MAGIC     }
# MAGIC     
# MAGIC     # 5. Compute overall success
# MAGIC     metrics['task_success'] = all([
# MAGIC         metrics['tool_recall'] == 1.0,
# MAGIC         metrics['param_accuracy'] == 1.0,
# MAGIC         metrics['response_quality'] >= 0.8
# MAGIC     ])
# MAGIC     
# MAGIC     return metrics
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,E2E Evaluation Framework
# End-to-end evaluation framework

class AgentSimulator:
    """Simulates an agent for testing purposes."""
    
    def __init__(self):
        self.tools = CustomerSupportTools()
    
    def process(self, order: Dict, conversation: List[Dict]) -> Dict:
        """
        Simulates agent processing.
        In a real system, this would call your LLM-based agent.
        
        For demo purposes, we'll implement rule-based logic.
        """
        last_message = conversation[-1]['content'].lower()
        
        result = {
            'tool_calls': [],
            'response': ''
        }
        
        # Simple rule-based logic for demo
        if 'cracked' in last_message or 'damaged' in last_message:
            # Issue refund for damaged item
            if len(order['items']) > 1 and 'mug' in last_message:
                # Multi-item order, refund only the mug
                mug_item = [item for item in order['items'] if 'mug' in item['name'].lower()][0]
                self.tools.issue_refund(order['order_id'], mug_item['unit_price'], "Damaged item")
                result['tool_calls'] = self.tools.actions_log[-1:]
                result['response'] = f"I've processed a refund of ${mug_item['unit_price']} for the damaged mug. You should see it in 3-5 business days."
            else:
                # Single item or unclear - refund full amount
                self.tools.issue_refund(order['order_id'], order['total'], "Damaged item")
                result['tool_calls'] = self.tools.actions_log[-1:]
                result['response'] = f"I've processed a full refund of ${order['total']}. You should see it in 3-5 business days."
        
        elif 'cancel' in last_message:
            if order['status'] == 'Delivered':
                # Cannot cancel delivered order
                result['response'] = "This order has already been delivered, so we cannot cancel it. However, I can help you with a refund if there's an issue."
            else:
                # Cancel pending order
                self.tools.cancel_order(order['order_id'], "Customer request")
                result['tool_calls'] = self.tools.actions_log[-1:]
                result['response'] = f"I've cancelled order {order['order_id']}. You won't be charged."
        
        elif 'address' in last_message or 'shipping' in last_message:
            if order['status'] in ['Pending', 'Processing']:
                # Extract new address (simplified)
                import re
                address_match = re.search(r'\d+\s+\w+\s+\w+', last_message)
                if address_match:
                    new_address = address_match.group(0)
                    self.tools.update_shipping_address(order['order_id'], new_address)
                    result['tool_calls'] = self.tools.actions_log[-1:]
                    result['response'] = f"I've updated your shipping address to {new_address}."
            else:
                result['response'] = "This order has already shipped, so we cannot change the address."
        
        return result

def evaluate_single_case(test_case: Dict, agent: AgentSimulator) -> Dict[str, float]:
    """Evaluate agent on a single test case."""
    
    # Run the agent
    result = agent.process(
        order=test_case['order'],
        conversation=test_case['conversation']
    )
    
    # Extract predictions
    pred_tools = [tc['tool'] for tc in result['tool_calls']]
    pred_calls = result['tool_calls']
    final_response = result['response']
    
    # Extract expected
    expected = test_case['expected']
    
    # Compute metrics
    tool_metrics_result = compute_tool_metrics(pred_tools, expected['tool_calls'])
    param_acc = compute_param_accuracy(pred_calls, expected['tool_calls'])
    
    # Check response contains expected phrases
    response_phrases = expected.get('response_should_contain', [])
    phrases_found = sum(1 for phrase in response_phrases if phrase.lower() in final_response.lower())
    phrase_recall = phrases_found / len(response_phrases) if response_phrases else 1.0
    
    # Overall task success
    task_success = (
        tool_metrics_result['tool_recall'] == 1.0 and
        tool_metrics_result['tool_precision'] >= 1.0 and
        param_acc == 1.0 and
        phrase_recall >= 0.5
    )
    
    return {
        'test_id': test_case['test_id'],
        'tool_recall': tool_metrics_result['tool_recall'],
        'tool_precision': tool_metrics_result['tool_precision'],
        'param_accuracy': param_acc,
        'phrase_recall': phrase_recall,
        'task_success': 1.0 if task_success else 0.0,
        'response': final_response
    }

print("✅ E2E evaluation framework ready")

# COMMAND ----------

# DBTITLE 1,Run E2E Evaluation on Test Suite
# Run end-to-end evaluation on our test suite

agent = AgentSimulator()
results = []

print("="*70)
print("END-TO-END EVALUATION RESULTS")
print("="*70)

for test_case in test_cases:
    result = evaluate_single_case(test_case, agent)
    results.append(result)
    
    # Print result
    print(f"\n{test_case['test_id']}: {test_case['description']}")
    print(f"  Tool Recall:    {result['tool_recall']:.2f}")
    print(f"  Tool Precision: {result['tool_precision']:.2f}")
    print(f"  Param Accuracy: {result['param_accuracy']:.2f}")
    print(f"  Phrase Recall:  {result['phrase_recall']:.2f}")
    print(f"  Task Success:   {'✅ PASS' if result['task_success'] == 1.0 else '❌ FAIL'}")
    print(f"  Response: \"{result['response'][:80]}...\"")

# Aggregate metrics
print("\n" + "="*70)
print("AGGREGATE METRICS")
print("="*70)

avg_tool_recall = sum(r['tool_recall'] for r in results) / len(results)
avg_tool_precision = sum(r['tool_precision'] for r in results) / len(results)
avg_param_accuracy = sum(r['param_accuracy'] for r in results) / len(results)
avg_phrase_recall = sum(r['phrase_recall'] for r in results) / len(results)
success_rate = sum(r['task_success'] for r in results) / len(results)

print(f"\nAverage Tool Recall:    {avg_tool_recall:.2%}")
print(f"Average Tool Precision: {avg_tool_precision:.2%}")
print(f"Average Param Accuracy: {avg_param_accuracy:.2%}")
print(f"Average Phrase Recall:  {avg_phrase_recall:.2%}")
print(f"\n🎯 Overall Success Rate: {success_rate:.2%}")

if success_rate >= 0.95:
    print("\n✅ EXCELLENT: Agent is production-ready!")
elif success_rate >= 0.80:
    print("\n⚠️  GOOD: Agent performs well but has room for improvement")
else:
    print("\n❌ NEEDS WORK: Agent requires significant improvements before deployment")

# COMMAND ----------

# DBTITLE 1,Quality Metrics: Consistency & Coherence
# MAGIC %md
# MAGIC ## Quality Metrics: Consistency & Coherence
# MAGIC
# MAGIC ### Consistency
# MAGIC
# MAGIC Consistency testing ensures the agent:
# MAGIC * Produces aligned outputs for similar inputs
# MAGIC * Doesn't contradict itself across turns
# MAGIC * Maintains context throughout conversations
# MAGIC
# MAGIC **Challenge**: LLMs are probabilistic, so outputs vary. We must test for *acceptable* variation, not exact matches.
# MAGIC
# MAGIC ### Coherence
# MAGIC
# MAGIC Coherence testing ensures:
# MAGIC * Logical flow in multi-turn conversations
# MAGIC * Appropriate reference to prior context
# MAGIC * No contradictions with earlier statements
# MAGIC * Goal-directed behavior
# MAGIC
# MAGIC ### Testing Approach
# MAGIC
# MAGIC **For Consistency:**
# MAGIC 1. Run the same input multiple times
# MAGIC 2. Check that tool calls remain the same
# MAGIC 3. Allow variation in phrasing, but not in actions
# MAGIC
# MAGIC **For Coherence:**
# MAGIC 1. Simulate multi-turn conversations
# MAGIC 2. Verify agent remembers prior context
# MAGIC 3. Check that responses build logically
# MAGIC 4. Detect contradictions

# COMMAND ----------

# DBTITLE 1,Consistency Testing Example
# Test consistency: Same input should produce same actions

def test_consistency(test_case: Dict, agent: AgentSimulator, num_runs: int = 5) -> Dict:
    """Run the same test case multiple times and check consistency."""
    
    results = []
    for i in range(num_runs):
        result = agent.process(
            order=test_case['order'],
            conversation=test_case['conversation']
        )
        results.append(result)
    
    # Check tool consistency
    tool_calls_list = [[tc['tool'] for tc in r['tool_calls']] for r in results]
    all_same = all(tc == tool_calls_list[0] for tc in tool_calls_list)
    
    # Check parameter consistency
    params_list = [r['tool_calls'] for r in results]
    params_same = all(p == params_list[0] for p in params_list)
    
    return {
        'test_id': test_case['test_id'],
        'num_runs': num_runs,
        'tool_calls_consistent': all_same,
        'parameters_consistent': params_same,
        'tool_calls': tool_calls_list[0] if tool_calls_list else [],
        'variation': len(set(str(tc) for tc in tool_calls_list))
    }

print("Testing consistency on multi-item refund case...\n")

# Test the multi-item refund case
refund_case = test_cases[1]  # Multi-item order
consistency_result = test_consistency(refund_case, agent, num_runs=5)

print(f"Test: {consistency_result['test_id']}")
print(f"Runs: {consistency_result['num_runs']}")
print(f"Tool calls consistent: {'✅' if consistency_result['tool_calls_consistent'] else '❌'}")
print(f"Parameters consistent: {'✅' if consistency_result['parameters_consistent'] else '❌'}")
print(f"Variation count: {consistency_result['variation']} unique outputs")
print(f"Tool calls: {consistency_result['tool_calls']}")

if consistency_result['tool_calls_consistent'] and consistency_result['parameters_consistent']:
    print("\n✅ PASS: Agent is consistent across runs")
else:
    print("\n❌ FAIL: Agent shows inconsistent behavior - investigate!")

# COMMAND ----------

# DBTITLE 1,Handling Unexpected Inputs
# MAGIC %md
# MAGIC ## Handling Unexpected Inputs
# MAGIC
# MAGIC Real-world agents must gracefully handle:
# MAGIC * **Malformed inputs**: Typos, incomplete information, invalid IDs
# MAGIC * **Ambiguous requests**: Multiple interpretations possible
# MAGIC * **Out-of-scope requests**: Tasks the agent cannot handle
# MAGIC * **Adversarial inputs**: Attempts to break the system
# MAGIC
# MAGIC ### Graceful Degradation
# MAGIC
# MAGIC When faced with unexpected inputs, the agent should:
# MAGIC 1. **Recognize** the issue
# MAGIC 2. **Avoid harmful actions** (don't guess)
# MAGIC 3. **Communicate clearly** to the user
# MAGIC 4. **Offer alternatives** or ask for clarification
# MAGIC
# MAGIC ### Testing Strategy
# MAGIC
# MAGIC **Adversarial Test Cases**
# MAGIC * Malformed order IDs
# MAGIC * Conflicting information in conversation
# MAGIC * Requests that violate business rules
# MAGIC * Injection attempts
# MAGIC * Very long or very short inputs
# MAGIC * Special characters and edge-case formatting

# COMMAND ----------

# DBTITLE 1,Adversarial Test Cases
# Build adversarial test cases

adversarial_cases = [
    {
        "test_id": "adv_001_malformed_id",
        "description": "Malformed order ID",
        "order": {
            "order_id": "INVALID-123-XYZ",  # Invalid format
            "status": "Delivered",
            "total": 19.99,
            "items": [{"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}]
        },
        "conversation": [
            {"role": "customer", "content": "Refund my order"},
        ],
        "expected_behavior": "Should NOT call tools with invalid ID, should ask for clarification"
    },
    {
        "test_id": "adv_002_conflicting_info",
        "description": "Conflicting information",
        "order": {
            "order_id": "A89268",
            "status": "Delivered",
            "total": 19.99,
            "items": [{"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}]
        },
        "conversation": [
            {"role": "customer", "content": "I want to cancel my order but also get a refund"},
        ],
        "expected_behavior": "Should clarify: cancel (if pending) OR refund (if delivered), not both"
    },
    {
        "test_id": "adv_003_out_of_scope",
        "description": "Out of scope request",
        "order": {
            "order_id": "A89268",
            "status": "Delivered",
            "total": 19.99,
            "items": [{"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}]
        },
        "conversation": [
            {"role": "customer", "content": "What's the weather like today?"},
        ],
        "expected_behavior": "Should decline politely and offer to help with order-related issues"
    },
    {
        "test_id": "adv_004_empty_message",
        "description": "Empty customer message",
        "order": {
            "order_id": "A89268",
            "status": "Delivered",
            "total": 19.99,
            "items": [{"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}]
        },
        "conversation": [
            {"role": "customer", "content": ""},  # Empty!
        ],
        "expected_behavior": "Should prompt for information, not crash or call tools"
    }
]

print("✅ Created adversarial test cases")
print(f"\nAdversarial Coverage ({len(adversarial_cases)} cases):")
for case in adversarial_cases:
    print(f"  • {case['test_id']}: {case['description']}")
    print(f"    Expected: {case['expected_behavior']}")

# COMMAND ----------

# DBTITLE 1,Deployment Readiness & Quality Gates
# MAGIC %md
# MAGIC ## Deployment Readiness & Quality Gates
# MAGIC
# MAGIC ### What is a Quality Gate?
# MAGIC
# MAGIC A quality gate is an **automated checkpoint** that prevents deployment unless all criteria are met.
# MAGIC
# MAGIC ### Typical Deployment Criteria
# MAGIC
# MAGIC #### Functional Requirements
# MAGIC * ✅ **Success rate** ≥ 95% on evaluation set
# MAGIC * ✅ **Tool recall** ≥ 98% (rarely miss required actions)
# MAGIC * ✅ **Parameter accuracy** ≥ 99% (rarely use wrong parameters)
# MAGIC
# MAGIC #### Quality Requirements
# MAGIC * ✅ **Consistency** ≥ 95% (consistent tool selection across runs)
# MAGIC * ✅ **Hallucination rate** < 2% (rarely fabricate information)
# MAGIC * ✅ **Graceful degradation** on adversarial inputs
# MAGIC
# MAGIC #### Performance Requirements
# MAGIC * ✅ **P95 latency** < 3 seconds
# MAGIC * ✅ **Cost per interaction** within budget
# MAGIC * ✅ **Throughput** ≥ target QPS
# MAGIC
# MAGIC #### Safety Requirements
# MAGIC * ✅ No critical or high-severity bugs
# MAGIC * ✅ Passes security review
# MAGIC * ✅ Appropriate error handling
# MAGIC * ✅ Logging and monitoring in place
# MAGIC
# MAGIC ### Deployment Process
# MAGIC
# MAGIC 1. **Offline Evaluation**: Run full test suite
# MAGIC 2. **Quality Gates**: Check all criteria
# MAGIC 3. **Staging Deployment**: Deploy to staging environment
# MAGIC 4. **Canary Testing**: Route small % of traffic to new version
# MAGIC 5. **Monitor**: Watch for regressions
# MAGIC 6. **Gradual Rollout**: Increase traffic gradually
# MAGIC 7. **Rollback Plan**: Be ready to revert if issues arise

# COMMAND ----------

# DBTITLE 1,Quality Gate Implementation
# Implement automated quality gates

class QualityGate:
    """Automated quality gate for deployment approval."""
    
    def __init__(self, criteria: Dict[str, float]):
        self.criteria = criteria
    
    def evaluate(self, metrics: Dict[str, float]) -> Dict:
        """Check if metrics meet deployment criteria."""
        
        checks = {}
        all_passed = True
        
        for metric_name, threshold in self.criteria.items():
            actual_value = metrics.get(metric_name, 0.0)
            passed = actual_value >= threshold
            
            checks[metric_name] = {
                'threshold': threshold,
                'actual': actual_value,
                'passed': passed,
                'margin': actual_value - threshold
            }
            
            if not passed:
                all_passed = False
        
        return {
            'passed': all_passed,
            'checks': checks,
            'timestamp': '2026-07-26T00:00:00Z'
        }
    
    def generate_report(self, evaluation_result: Dict) -> str:
        """Generate human-readable quality gate report."""
        
        report = []
        report.append("=" * 70)
        report.append("QUALITY GATE EVALUATION")
        report.append("=" * 70)
        
        for metric, check in evaluation_result['checks'].items():
            status = "✅ PASS" if check['passed'] else "❌ FAIL"
            margin = f"+{check['margin']:.2%}" if check['margin'] >= 0 else f"{check['margin']:.2%}"
            
            report.append(f"\n{metric}:")
            report.append(f"  Required: {check['threshold']:.2%}")
            report.append(f"  Actual:   {check['actual']:.2%} ({margin})")
            report.append(f"  Status:   {status}")
        
        report.append("\n" + "=" * 70)
        if evaluation_result['passed']:
            report.append("✅ ALL CHECKS PASSED - APPROVED FOR DEPLOYMENT")
        else:
            report.append("❌ QUALITY GATE FAILED - DEPLOYMENT BLOCKED")
        report.append("=" * 70)
        
        return "\n".join(report)

# Define deployment criteria
deployment_criteria = {
    'tool_recall': 0.98,        # 98% of required tools must be called
    'tool_precision': 0.95,     # 95% of called tools must be correct
    'param_accuracy': 0.99,     # 99% of parameters must be correct
    'task_success': 0.95,       # 95% overall success rate
}

gate = QualityGate(deployment_criteria)

# Use our earlier aggregate metrics
metrics_to_check = {
    'tool_recall': avg_tool_recall,
    'tool_precision': avg_tool_precision,
    'param_accuracy': avg_param_accuracy,
    'task_success': success_rate,
}

gate_result = gate.evaluate(metrics_to_check)
print(gate.generate_report(gate_result))

if not gate_result['passed']:
    print("\n⚠️  ACTION REQUIRED:")
    print("  1. Review failing test cases")
    print("  2. Improve agent implementation")
    print("  3. Re-run evaluation")
    print("  4. Only deploy after all gates pass")

# COMMAND ----------

# DBTITLE 1,Best Practices & Key Takeaways
# MAGIC %md
# MAGIC ## Best Practices for Measurement & Validation
# MAGIC
# MAGIC ### 1. Build Evaluation Into the Development Cycle
# MAGIC
# MAGIC * ✅ **Automate everything**: Run tests on every commit
# MAGIC * ✅ **Track metrics over time**: Detect regressions early
# MAGIC * ✅ **Expand test sets continuously**: Add new scenarios as you discover them
# MAGIC * ✅ **Version your evaluation sets**: Track what was tested when
# MAGIC
# MAGIC ### 2. Use Multiple Evaluation Strategies
# MAGIC
# MAGIC * ✅ **Unit tests**: Validate individual components
# MAGIC * ✅ **Integration tests**: Validate complete workflows
# MAGIC * ✅ **Consistency tests**: Check for stable behavior
# MAGIC * ✅ **Adversarial tests**: Probe for failure modes
# MAGIC * ✅ **Human evaluation**: Sample and review regularly
# MAGIC
# MAGIC ### 3. Choose the Right Metrics
# MAGIC
# MAGIC * ✅ **Task-specific**: Align with business objectives
# MAGIC * ✅ **Comprehensive**: Cover functionality, quality, and performance
# MAGIC * ✅ **Actionable**: Help you understand what to fix
# MAGIC * ✅ **Realistic**: Reflect real-world conditions
# MAGIC
# MAGIC ### 4. Set Appropriate Quality Gates
# MAGIC
# MAGIC * ✅ **Based on data**: Use real evaluation results to set thresholds
# MAGIC * ✅ **Balanced**: Don't be too strict or too lenient
# MAGIC * ✅ **Enforced**: Block deployment if gates fail
# MAGIC * ✅ **Reviewed regularly**: Adjust as system evolves
# MAGIC
# MAGIC ### 5. Monitor in Production
# MAGIC
# MAGIC * ✅ **Real-time metrics**: Track success rate, latency, errors
# MAGIC * ✅ **User feedback**: Collect explicit ratings and implicit signals
# MAGIC * ✅ **A/B testing**: Compare versions scientifically
# MAGIC * ✅ **Alerting**: Get notified of regressions immediately
# MAGIC
# MAGIC ### 6. Iterate Based on Failures
# MAGIC
# MAGIC * ✅ **Root cause analysis**: Understand why tests fail
# MAGIC * ✅ **Add test cases**: Turn failures into regression tests
# MAGIC * ✅ **Refine metrics**: Improve measurement as you learn
# MAGIC * ✅ **Update gates**: Keep criteria aligned with reality
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC 📊 **Measurement is the foundation** of reliable agent development
# MAGIC * Without rigorous measurement, you're flying blind
# MAGIC * Invest in evaluation infrastructure early
# MAGIC * Treat test sets as first-class artifacts
# MAGIC
# MAGIC ⚙️ **Test at multiple levels**
# MAGIC * Unit tests validate components
# MAGIC * Integration tests validate workflows
# MAGIC * Consistency tests validate stability
# MAGIC * Adversarial tests validate robustness
# MAGIC
# MAGIC 🚦 **Quality gates prevent bad deployments**
# MAGIC * Define clear, measurable criteria
# MAGIC * Automate enforcement
# MAGIC * Never deploy if gates fail
# MAGIC * Monitor continuously in production
# MAGIC
# MAGIC 🔄 **Iterate continuously**
# MAGIC * Expand test coverage over time
# MAGIC * Refine metrics based on learnings
# MAGIC * Adjust gates as system matures
# MAGIC * Use production feedback to improve
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What's Next?
# MAGIC
# MAGIC With robust measurement and validation in place, you can:
# MAGIC 1. **Confidently deploy** agents to production
# MAGIC 2. **Detect regressions** before users do
# MAGIC 3. **Optimize systematically** using data
# MAGIC 4. **Build trust** with stakeholders
# MAGIC 5. **Scale safely** as complexity grows
# MAGIC
# MAGIC Remember: **Measurement enables improvement. What gets measured gets managed.**

# COMMAND ----------

# DBTITLE 1,Summary: Complete Evaluation Pipeline
# Summary: Complete evaluation pipeline in one place

def run_complete_evaluation(agent: AgentSimulator, 
                           test_suite: List[Dict],
                           quality_gate: QualityGate) -> Dict:
    """
    Complete evaluation pipeline:
    1. Run all tests
    2. Compute aggregate metrics
    3. Check quality gates
    4. Generate report
    """
    
    print("Starting complete evaluation pipeline...\n")
    
    # 1. Run all tests
    print("Phase 1: Running test suite...")
    results = []
    for test_case in test_suite:
        result = evaluate_single_case(test_case, agent)
        results.append(result)
    print(f"✅ Completed {len(results)} test cases\n")
    
    # 2. Compute aggregate metrics
    print("Phase 2: Computing aggregate metrics...")
    metrics = {
        'tool_recall': sum(r['tool_recall'] for r in results) / len(results),
        'tool_precision': sum(r['tool_precision'] for r in results) / len(results),
        'param_accuracy': sum(r['param_accuracy'] for r in results) / len(results),
        'phrase_recall': sum(r['phrase_recall'] for r in results) / len(results),
        'task_success': sum(r['task_success'] for r in results) / len(results),
    }
    
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2%}")
    print()
    
    # 3. Check quality gates
    print("Phase 3: Checking quality gates...")
    gate_result = quality_gate.evaluate(metrics)
    print(quality_gate.generate_report(gate_result))
    
    return {
        'test_results': results,
        'aggregate_metrics': metrics,
        'quality_gate': gate_result,
        'deployment_approved': gate_result['passed']
    }

# Run the complete pipeline
print("="*70)
print("COMPLETE EVALUATION PIPELINE")
print("="*70 + "\n")

final_result = run_complete_evaluation(
    agent=agent,
    test_suite=test_cases,
    quality_gate=gate
)

print("\n" + "="*70)
if final_result['deployment_approved']:
    print("✅ DEPLOYMENT APPROVED")
    print("\nNext steps:")
    print("  1. Deploy to staging environment")
    print("  2. Run canary test (5% traffic)")
    print("  3. Monitor for 24 hours")
    print("  4. Gradually roll out to 100%")
else:
    print("❌ DEPLOYMENT BLOCKED")
    print("\nNext steps:")
    print("  1. Review failing test cases")
    print("  2. Identify root causes")
    print("  3. Improve agent implementation")
    print("  4. Re-run evaluation pipeline")
print("="*70)