# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introduction: Monitoring Agentic Systems
# MAGIC %md
# MAGIC # Monitoring in Production
# MAGIC
# MAGIC ## Why Monitoring Matters for Agents
# MAGIC
# MAGIC Shipping agentic systems is only halfway. The real challenge begins in production, where agents face:
# MAGIC * **Unpredictable inputs**: Users ask questions you never tested
# MAGIC * **Dynamic environments**: APIs change, data drifts, models update
# MAGIC * **Probabilistic behavior**: Same input can produce different outputs
# MAGIC * **Emergent failures**: Subtle issues that don't crash but degrade quality
# MAGIC
# MAGIC **Traditional monitoring is insufficient.** Infrastructure metrics (CPU, latency, errors) tell you *if* the system is running, but not *whether it's working well*.
# MAGIC
# MAGIC ## What Makes Agent Monitoring Different
# MAGIC
# MAGIC **Semantic failures**: Agent produces fluent but incorrect output
# MAGIC * Tool calls succeed but cascade errors
# MAGIC * Planning is partially correct but misses the goal
# MAGIC * No error logs, but user abandons task
# MAGIC
# MAGIC **These failures are invisible to traditional monitoring.**
# MAGIC
# MAGIC ## Monitoring as a Learning System
# MAGIC
# MAGIC Production monitoring enables:
# MAGIC * **Early detection**: Catch regressions before users complain
# MAGIC * **Root cause analysis**: Understand why failures happen
# MAGIC * **Continuous improvement**: Turn failures into test cases
# MAGIC * **Safe experimentation**: Deploy changes confidently
# MAGIC
# MAGIC ## What We'll Cover
# MAGIC
# MAGIC 1. **Metrics taxonomy**: What to measure and why
# MAGIC 2. **Instrumentation**: How to collect meaningful signals
# MAGIC 3. **Distribution shifts**: Detecting when your world changes
# MAGIC 4. **Monitoring patterns**: Shadow mode, canaries, self-healing
# MAGIC 5. **Alerting**: When and how to notify teams
# MAGIC 6. **Cross-functional ownership**: Who owns what metrics
# MAGIC
# MAGIC Let's build a production-grade monitoring system for our customer support agent.

# COMMAND ----------

# DBTITLE 1,Taxonomy of Metrics
# MAGIC %md
# MAGIC ## Taxonomy of Metrics
# MAGIC
# MAGIC ### Infrastructure Metrics
# MAGIC
# MAGIC **System Health**
# MAGIC * CPU/memory usage
# MAGIC * Request latency (P50, P95, P99)
# MAGIC * Uptime/availability
# MAGIC * Error rates
# MAGIC
# MAGIC ### Workflow Metrics
# MAGIC
# MAGIC **Task Performance**
# MAGIC * Task success rate: How often does the agent complete its goal?
# MAGIC * Token usage: Input/output tokens per request
# MAGIC * Tool call success/failure rate
# MAGIC * Retry frequency: How often do retries happen?
# MAGIC * Fallback frequency: How often do fallback paths trigger?
# MAGIC
# MAGIC ### Output Quality Metrics
# MAGIC
# MAGIC **Semantic Correctness**
# MAGIC * Hallucination rate: Fabricated or incorrect information
# MAGIC * Response coherence: Does the output make sense?
# MAGIC * Tool accuracy: Right tool with right parameters?
# MAGIC * Embedding drift: Are inputs/outputs shifting distribution?
# MAGIC
# MAGIC ### User Experience Metrics
# MAGIC
# MAGIC **User Satisfaction**
# MAGIC * Task abandonment rate: Users give up mid-conversation
# MAGIC * Requery/rephrasing rate: Users have to rephrase to be understood
# MAGIC * Explicit ratings: Thumbs up/down, star ratings
# MAGIC * Task completion time
# MAGIC
# MAGIC ### Example: What to Alert On
# MAGIC
# MAGIC ⚠️ **Critical alerts** (page immediately):
# MAGIC * Task success rate < 90%
# MAGIC * Hallucination rate > 5%
# MAGIC * P99 latency > 10 seconds
# MAGIC
# MAGIC ⚠️ **Warning alerts** (investigate within hours):
# MAGIC * Tool failure rate increasing > 20%
# MAGIC * Token usage spike > 50%
# MAGIC * User abandonment > 15%
# MAGIC
# MAGIC 📈 **Trend monitoring** (review weekly):
# MAGIC * Embedding drift metrics
# MAGIC * Feature usage patterns
# MAGIC * Cost per conversation

# COMMAND ----------

# DBTITLE 1,Setup: Instrumentation Basics
# Setup: Instrumentation framework for monitoring agents

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import json

@dataclass
class Span:
    """Represents a trace span (inspired by OpenTelemetry)."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "in_progress"  # in_progress, success, error
    
    def finish(self, status: str = "success"):
        self.end_time = time.time()
        self.status = status
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

@dataclass
class Trace:
    """Represents a full request trace containing multiple spans."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: Span):
        self.spans.append(span)
    
    def get_total_duration_ms(self) -> float:
        if not self.spans:
            return 0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans if s.end_time)
        return (end - start) * 1000

class MetricsCollector:
    """Simple metrics collector for agent monitoring."""
    
    def __init__(self):
        self.metrics = []
        self.traces = []
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        self.metrics.append({
            'timestamp': datetime.utcnow().isoformat(),
            'name': name,
            'value': value,
            'labels': labels or {}
        })
    
    def start_span(self, name: str, attributes: Dict = None) -> Span:
        """Start a new span for tracing."""
        span = Span(
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )
        return span
    
    def add_trace(self, trace: Trace):
        """Store a completed trace."""
        self.traces.append(trace)
    
    def get_metric_summary(self, metric_name: str) -> Dict:
        """Get summary statistics for a metric."""
        values = [m['value'] for m in self.metrics if m['name'] == metric_name]
        if not values:
            return {}
        
        values.sort()
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'p50': values[len(values) // 2],
            'p95': values[int(len(values) * 0.95)] if len(values) > 20 else values[-1],
            'p99': values[int(len(values) * 0.99)] if len(values) > 100 else values[-1],
        }

# Initialize global collector
collector = MetricsCollector()

print("✅ Instrumentation framework initialized")
print("\nAvailable:")
print("  • Span: For tracing agent operations")
print("  • Trace: For grouping related spans")
print("  • MetricsCollector: For recording metrics")

# COMMAND ----------

# DBTITLE 1,Example: Instrumented Agent
# Instrumented version of our customer support agent

import uuid

class InstrumentedAgent:
    """Customer support agent with monitoring instrumentation."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def process_request(self, order: Dict, message: str) -> Dict:
        """Process a customer request with full instrumentation."""
        
        # Create a trace for this request
        trace_id = str(uuid.uuid4())
        trace = Trace(trace_id=trace_id, metadata={'order_id': order.get('order_id')})
        
        # Overall request span
        request_span = self.collector.start_span(
            "process_request",
            attributes={'trace_id': trace_id, 'order_id': order.get('order_id')}
        )
        
        try:
            # Intent classification span
            intent_span = self.collector.start_span(
                "classify_intent",
                attributes={'message_length': len(message)}
            )
            intent = self._classify_intent(message)
            intent_span.attributes['intent'] = intent
            intent_span.finish()
            trace.add_span(intent_span)
            
            # Planning span
            plan_span = self.collector.start_span(
                "generate_plan",
                attributes={'intent': intent}
            )
            plan = self._generate_plan(intent, order)
            plan_span.attributes['tools_planned'] = len(plan.get('tools', []))
            plan_span.finish()
            trace.add_span(plan_span)
            
            # Tool execution span
            if plan.get('tools'):
                tool_span = self.collector.start_span(
                    "execute_tools",
                    attributes={'num_tools': len(plan['tools'])}
                )
                result = self._execute_tools(plan['tools'])
                tool_span.attributes['success'] = result.get('success', False)
                tool_span.finish('success' if result.get('success') else 'error')
                trace.add_span(tool_span)
            
            # Response generation span
            response_span = self.collector.start_span(
                "generate_response",
                attributes={'intent': intent}
            )
            response = self._generate_response(intent, plan)
            response_span.attributes['response_length'] = len(response)
            response_span.finish()
            trace.add_span(response_span)
            
            # Record metrics
            self.collector.record_metric('task_success', 1.0, {'intent': intent})
            self.collector.record_metric('request_latency_ms', request_span.duration_ms)
            
            request_span.finish('success')
            trace.add_span(request_span)
            self.collector.add_trace(trace)
            
            return {
                'success': True,
                'trace_id': trace_id,
                'intent': intent,
                'response': response
            }
            
        except Exception as e:
            # Record failure
            self.collector.record_metric('task_success', 0.0, {'intent': 'unknown', 'error': str(e)})
            request_span.attributes['error'] = str(e)
            request_span.finish('error')
            trace.add_span(request_span)
            self.collector.add_trace(trace)
            
            return {
                'success': False,
                'trace_id': trace_id,
                'error': str(e)
            }
    
    def _classify_intent(self, message: str) -> str:
        """Simple intent classification."""
        message_lower = message.lower()
        if 'refund' in message_lower or 'damaged' in message_lower:
            return 'refund'
        elif 'cancel' in message_lower:
            return 'cancel'
        elif 'address' in message_lower or 'shipping' in message_lower:
            return 'update_address'
        return 'unknown'
    
    def _generate_plan(self, intent: str, order: Dict) -> Dict:
        """Generate execution plan based on intent."""
        if intent == 'refund':
            return {
                'tools': ['issue_refund'],
                'params': {'order_id': order.get('order_id'), 'amount': order.get('total')}
            }
        elif intent == 'cancel':
            if order.get('status') == 'Delivered':
                return {'tools': [], 'message': 'cannot_cancel_delivered'}
            return {'tools': ['cancel_order'], 'params': {'order_id': order.get('order_id')}}
        elif intent == 'update_address':
            return {'tools': ['update_shipping_address'], 'params': {'order_id': order.get('order_id')}}
        return {'tools': []}
    
    def _execute_tools(self, tools: List[str]) -> Dict:
        """Execute tools (simulated)."""
        # Simulate tool execution
        time.sleep(0.01)  # Small delay
        return {'success': True}
    
    def _generate_response(self, intent: str, plan: Dict) -> str:
        """Generate customer-facing response."""
        if intent == 'refund':
            return "I've processed your refund. You should see it in 3-5 business days."
        elif intent == 'cancel':
            if not plan.get('tools'):
                return "This order has already been delivered, so we cannot cancel it."
            return "I've cancelled your order. You won't be charged."
        elif intent == 'update_address':
            return "I've updated your shipping address."
        return "I'm sorry, I didn't understand your request. Can you please clarify?"

# Test the instrumented agent
agent = InstrumentedAgent(collector)

test_order = {
    'order_id': 'A89268',
    'status': 'Delivered',
    'total': 19.99
}

result = agent.process_request(test_order, "My mug arrived damaged, can I get a refund?")

print("✅ Instrumented agent test complete")
print(f"\nResult:")
print(f"  Success: {result['success']}")
print(f"  Trace ID: {result['trace_id']}")
print(f"  Intent: {result['intent']}")
print(f"  Response: {result['response'][:60]}...")

# COMMAND ----------

# DBTITLE 1,Distribution Shift Detection
# MAGIC %md
# MAGIC ## Distribution Shift Detection
# MAGIC
# MAGIC ### The Silent Killer: Drift
# MAGIC
# MAGIC One of the subtlest failures in production agents is **distribution shift**:
# MAGIC * User language changes over time
# MAGIC * New product terminology emerges
# MAGIC * API responses evolve
# MAGIC * Model updates change behavior
# MAGIC
# MAGIC **These shifts don't crash your system—they degrade it slowly.**
# MAGIC
# MAGIC ### Statistical Tests for Drift
# MAGIC
# MAGIC We'll implement three common drift detection methods:
# MAGIC
# MAGIC **1. Kolmogorov-Smirnov (KS) Test**
# MAGIC * Compares two probability distributions
# MAGIC * Good for continuous features (e.g., query lengths, latencies)
# MAGIC * Threshold: KS statistic > 0.1 indicates drift
# MAGIC
# MAGIC **2. Kullback-Leibler (KL) Divergence**
# MAGIC * Measures how one distribution diverges from another
# MAGIC * Good for token distributions, word frequencies
# MAGIC * Threshold: KL > 0.5 suggests significant drift
# MAGIC
# MAGIC **3. Population Stability Index (PSI)**
# MAGIC * Detects shifts in categorical distributions
# MAGIC * Good for tool usage patterns, intent categories
# MAGIC * < 0.1: stable, 0.1-0.25: minor drift, > 0.25: major drift
# MAGIC
# MAGIC ### When to Use Each
# MAGIC
# MAGIC | Test | Best For | Example Use Case |
# MAGIC |------|----------|------------------|
# MAGIC | KS   | Continuous distributions | Query length drift, latency changes |
# MAGIC | KL   | Probability distributions | Token frequency shifts, language changes |
# MAGIC | PSI  | Categorical/binned data | Tool usage patterns, intent distribution |

# COMMAND ----------

# DBTITLE 1,KS Test: Detecting Continuous Distribution Shifts
# Kolmogorov-Smirnov Test for drift detection

import numpy as np
from scipy import stats

def detect_drift_ks(historical: np.ndarray, current: np.ndarray, threshold: float = 0.1) -> Dict:
    """
    Detect distribution drift using KS test.
    
    Args:
        historical: Historical data (baseline)
        current: Current data (new observations)
        threshold: KS statistic threshold for drift detection
    
    Returns:
        Dictionary with drift status and statistics
    """
    ks_stat, p_value = stats.ks_2samp(historical, current)
    
    drift_detected = ks_stat > threshold
    
    return {
        'drift_detected': drift_detected,
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'threshold': threshold,
        'interpretation': 'Significant drift detected' if drift_detected else 'No significant drift'
    }

# Example: Query length drift
print("="*70)
print("KS TEST: Query Length Drift Detection")
print("="*70)

# Historical baseline: typical query lengths
historical_query_lengths = np.array([10, 12, 15, 18, 20, 22, 14, 16, 19, 21, 13, 17])

# Current data: users are writing much longer queries
current_query_lengths = np.array([25, 28, 30, 32, 35, 27, 29, 31, 33, 26, 34, 36])

result = detect_drift_ks(historical_query_lengths, current_query_lengths)

print(f"\nHistorical mean: {historical_query_lengths.mean():.1f} characters")
print(f"Current mean: {current_query_lengths.mean():.1f} characters")
print(f"\nKS Statistic: {result['ks_statistic']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Threshold: {result['threshold']}")
print(f"\n{'🔴' if result['drift_detected'] else '🟢'} {result['interpretation']}")

if result['drift_detected']:
    print("\n⚠️  ACTION REQUIRED:")
    print("  • Users are writing significantly longer queries")
    print("  • May need to adjust prompt truncation")
    print("  • Check if token costs are increasing")
    print("  • Investigate if this is a product change or organic trend")

# COMMAND ----------

# DBTITLE 1,KL Divergence: Token Distribution Drift
# Kullback-Leibler Divergence for concept drift

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        p: Historical distribution (baseline)
        q: Current distribution (new observations)
        epsilon: Small constant to avoid log(0)
    
    Returns:
        KL divergence value
    """
    # Add epsilon to avoid division by zero
    p = p + epsilon
    q = q + epsilon
    
    # Normalize to probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute KL divergence: sum(p * log(p/q))
    return np.sum(p * np.log(p / q))

def detect_drift_kl(historical: np.ndarray, current: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Detect concept drift using KL divergence.
    
    Args:
        historical: Historical distribution
        current: Current distribution
        threshold: KL threshold for drift detection
    
    Returns:
        Dictionary with drift status and KL value
    """
    kl = kl_divergence(historical, current)
    drift_detected = kl > threshold
    
    return {
        'drift_detected': drift_detected,
        'kl_divergence': kl,
        'threshold': threshold,
        'interpretation': 'Significant concept drift' if drift_detected else 'No significant drift'
    }

# Example: Token/word frequency drift
print("\n" + "="*70)
print("KL DIVERGENCE: Token Frequency Drift")
print("="*70)

# Historical: typical word distribution in customer queries
# ["refund", "cancel", "address", "tracking", "other"]
historical_tokens = np.array([0.40, 0.25, 0.15, 0.10, 0.10])

# Current: sudden spike in "address" changes, drop in refunds
current_tokens = np.array([0.20, 0.25, 0.40, 0.10, 0.05])

result_kl = detect_drift_kl(historical_tokens, current_tokens)

print("\nToken Distribution:")
print("  Category    | Historical | Current | Change")
print("  ------------|------------|---------|--------")
labels = ["refund", "cancel", "address", "tracking", "other"]
for i, label in enumerate(labels):
    change = (current_tokens[i] - historical_tokens[i]) * 100
    arrow = "↑" if change > 0 else "↓" if change < 0 else "="
    print(f"  {label:11s} | {historical_tokens[i]:.2f}       | {current_tokens[i]:.2f}    | {arrow} {abs(change):.0f}%")

print(f"\nKL Divergence: {result_kl['kl_divergence']:.4f}")
print(f"Threshold: {result_kl['threshold']}")
print(f"\n{'🔴' if result_kl['drift_detected'] else '🟢'} {result_kl['interpretation']}")

if result_kl['drift_detected']:
    print("\n⚠️  ACTION REQUIRED:")
    print("  • User intent distribution has shifted significantly")
    print("  • 'address' queries increased by 25 percentage points")
    print("  • May indicate a product issue (e.g., address update UI broken)")
    print("  • Review recent product changes")

# COMMAND ----------

# DBTITLE 1,PSI: Categorical Distribution Drift
# Population Stability Index for categorical drift

def psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Population Stability Index.
    
    Args:
        expected: Historical distribution (baseline)
        actual: Current distribution (new observations)
        epsilon: Small constant to avoid division by zero
    
    Returns:
        PSI value
    """
    # Convert to percentages
    expected_percents = expected / np.sum(expected)
    actual_percents = actual / np.sum(actual)
    
    # Add epsilon to avoid division by zero
    expected_percents = expected_percents + epsilon
    actual_percents = actual_percents + epsilon
    
    # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    
    return np.sum(psi_values)

def detect_drift_psi(historical: np.ndarray, current: np.ndarray) -> Dict:
    """
    Detect drift using PSI with standard thresholds.
    
    PSI Interpretation:
    - < 0.1: No significant change
    - 0.1 - 0.25: Minor drift (monitor)
    - > 0.25: Major drift (action required)
    """
    psi_value = psi(historical, current)
    
    if psi_value < 0.1:
        severity = 'none'
        interpretation = 'No significant change (stable)'
    elif psi_value < 0.25:
        severity = 'minor'
        interpretation = 'Minor drift detected (monitor)'
    else:
        severity = 'major'
        interpretation = 'Major drift detected (action required)'
    
    return {
        'psi': psi_value,
        'severity': severity,
        'interpretation': interpretation,
        'drift_detected': severity in ['minor', 'major']
    }

# Example: Tool usage pattern drift
print("\n" + "="*70)
print("PSI: Tool Usage Pattern Drift")
print("="*70)

# Historical tool usage counts
# ["issue_refund", "cancel_order", "update_address", "get_tracking"]
historical_tools = np.array([50, 30, 15, 5])

# Current: sudden shift in tool usage
current_tools = np.array([20, 50, 25, 5])

result_psi = detect_drift_psi(historical_tools, current_tools)

print("\nTool Usage:")
print("  Tool              | Historical | Current | % of Total")
print("  ------------------|------------|---------|------------")
tool_labels = ["issue_refund", "cancel_order", "update_address", "get_tracking"]
for i, label in enumerate(tool_labels):
    hist_pct = (historical_tools[i] / historical_tools.sum()) * 100
    curr_pct = (current_tools[i] / current_tools.sum()) * 100
    print(f"  {label:17s} | {historical_tools[i]:3.0f} ({hist_pct:4.1f}%) | {current_tools[i]:3.0f} ({curr_pct:4.1f}%) |")

print(f"\nPSI Value: {result_psi['psi']:.4f}")
print(f"Severity: {result_psi['severity'].upper()}")
print(f"\n{'🔴' if result_psi['severity'] == 'major' else '🟡' if result_psi['severity'] == 'minor' else '🟢'} {result_psi['interpretation']}")

if result_psi['drift_detected']:
    print("\n⚠️  ACTION REQUIRED:")
    if result_psi['severity'] == 'major':
        print("  • MAJOR SHIFT in tool usage patterns")
        print("  • 'cancel_order' increased from 30% to 50%")
        print("  • 'issue_refund' dropped from 50% to 20%")
        print("  • Possible causes:")
        print("    - Product quality issue leading to more cancellations")
        print("    - Shipping delays causing users to cancel")
        print("    - Changes in customer behavior")
        print("  • Immediate investigation recommended")
    else:
        print("  • Minor shift detected - continue monitoring")
        print("  • Review again in 24-48 hours")

# COMMAND ----------

# DBTITLE 1,Monitoring Patterns
# MAGIC %md
# MAGIC ## Monitoring Patterns for Safe Deployments
# MAGIC
# MAGIC ### The Challenge
# MAGIC
# MAGIC Agents are probabilistic. How do you deploy changes confidently?
# MAGIC * Can't test every possible input
# MAGIC * Performance varies by user and context
# MAGIC * Failures are often subtle, not catastrophic
# MAGIC
# MAGIC **Solution**: Monitoring-aware deployment patterns that create safety nets.
# MAGIC
# MAGIC ### Pattern 1: Shadow Mode
# MAGIC
# MAGIC **Concept**: New agent version runs in parallel with production, processing same inputs, but outputs aren't served to users.
# MAGIC
# MAGIC **Benefits**:
# MAGIC * Zero user risk
# MAGIC * Real-world evaluation
# MAGIC * Direct comparison with baseline
# MAGIC
# MAGIC **Use when**:
# MAGIC * Testing major changes (new model, different planning logic)
# MAGIC * Need extensive data before go-live
# MAGIC * Want to validate metrics before exposing to users
# MAGIC
# MAGIC ### Pattern 2: Canary Deployment
# MAGIC
# MAGIC **Concept**: New version serves small % of traffic (e.g., 5%), majority stays on baseline.
# MAGIC
# MAGIC **Benefits**:
# MAGIC * Real user feedback
# MAGIC * Limited blast radius
# MAGIC * Gradual rollout
# MAGIC
# MAGIC **Use when**:
# MAGIC * Shadow mode shows promise
# MAGIC * Ready for real user exposure
# MAGIC * Can monitor and rollback quickly
# MAGIC
# MAGIC ### Pattern 3: A/B Testing
# MAGIC
# MAGIC **Concept**: Split traffic between versions, compare metrics statistically.
# MAGIC
# MAGIC **Benefits**:
# MAGIC * Rigorous comparison
# MAGIC * Quantify improvement
# MAGIC * Data-driven decisions
# MAGIC
# MAGIC **Use when**:
# MAGIC * Optimizing existing features
# MAGIC * Need statistical confidence
# MAGIC * Have sufficient traffic
# MAGIC
# MAGIC ### Pattern 4: Self-Healing Agents
# MAGIC
# MAGIC **Concept**: Agent monitors its own telemetry and triggers fallback behaviors.
# MAGIC
# MAGIC **Benefits**:
# MAGIC * Automatic recovery
# MAGIC * Reduced manual intervention
# MAGIC * Better user experience
# MAGIC
# MAGIC **Use when**:
# MAGIC * Have reliable fallback strategies
# MAGIC * Can detect failures in real-time
# MAGIC * Want to reduce MTTR (Mean Time To Recovery)

# COMMAND ----------

# DBTITLE 1,Pattern: Shadow Mode Implementation
# Shadow Mode: Running two agent versions side-by-side

class ShadowModeController:
    """Controller for running shadow mode experiments."""
    
    def __init__(self, production_agent, shadow_agent, collector: MetricsCollector):
        self.production_agent = production_agent
        self.shadow_agent = shadow_agent
        self.collector = collector
        self.comparisons = []
    
    def process_request(self, order: Dict, message: str) -> Dict:
        """
        Process request through both agents, serve production result.
        Log comparison metrics.
        """
        
        # Run production agent
        prod_result = self.production_agent.process_request(order, message)
        
        # Run shadow agent (in parallel, but don't serve result)
        shadow_result = self.shadow_agent.process_request(order, message)
        
        # Compare and log
        comparison = self._compare_results(prod_result, shadow_result)
        self.comparisons.append(comparison)
        
        # Record comparison metrics
        self.collector.record_metric(
            'shadow_latency_diff_ms',
            comparison['latency_diff_ms'],
            {'shadow_faster': str(comparison['shadow_faster'])}
        )
        
        self.collector.record_metric(
            'shadow_intent_match',
            1.0 if comparison['intent_match'] else 0.0
        )
        
        # Always return production result
        return prod_result
    
    def _compare_results(self, prod: Dict, shadow: Dict) -> Dict:
        """Compare production and shadow results."""
        # Get traces for latency comparison
        prod_trace = [t for t in self.collector.traces if t.trace_id == prod.get('trace_id')]
        shadow_trace = [t for t in self.collector.traces if t.trace_id == shadow.get('trace_id')]
        
        prod_latency = prod_trace[0].get_total_duration_ms() if prod_trace else 0
        shadow_latency = shadow_trace[0].get_total_duration_ms() if shadow_trace else 0
        
        return {
            'prod_intent': prod.get('intent'),
            'shadow_intent': shadow.get('intent'),
            'intent_match': prod.get('intent') == shadow.get('intent'),
            'prod_latency_ms': prod_latency,
            'shadow_latency_ms': shadow_latency,
            'latency_diff_ms': shadow_latency - prod_latency,
            'shadow_faster': shadow_latency < prod_latency,
            'prod_success': prod.get('success'),
            'shadow_success': shadow.get('success'),
        }
    
    def get_summary(self) -> Dict:
        """Get summary of shadow mode comparison."""
        if not self.comparisons:
            return {}
        
        intent_matches = sum(1 for c in self.comparisons if c['intent_match'])
        shadow_faster_count = sum(1 for c in self.comparisons if c['shadow_faster'])
        
        latency_diffs = [c['latency_diff_ms'] for c in self.comparisons]
        
        return {
            'total_requests': len(self.comparisons),
            'intent_match_rate': intent_matches / len(self.comparisons),
            'shadow_faster_rate': shadow_faster_count / len(self.comparisons),
            'avg_latency_diff_ms': sum(latency_diffs) / len(latency_diffs),
            'prod_success_rate': sum(1 for c in self.comparisons if c['prod_success']) / len(self.comparisons),
            'shadow_success_rate': sum(1 for c in self.comparisons if c['shadow_success']) / len(self.comparisons)
        }

# Simulate shadow mode
print("="*70)
print("SHADOW MODE: Comparing Production vs New Agent")
print("="*70)

# Create two agents with separate collectors
prod_collector = MetricsCollector()
shadow_collector = MetricsCollector()

prod_agent = InstrumentedAgent(prod_collector)
shadow_agent = InstrumentedAgent(shadow_collector)  # In reality, this would be the new version

controller = ShadowModeController(prod_agent, shadow_agent, collector)

# Simulate traffic
test_messages = [
    "My mug is damaged",
    "Cancel my order please",
    "Update my shipping address",
    "Where is my package?"
]

for msg in test_messages:
    controller.process_request(test_order, msg)

summary = controller.get_summary()

print(f"\nShadow Mode Results ({summary['total_requests']} requests):")
print(f"  Intent Match Rate:    {summary['intent_match_rate']:.1%}")
print(f"  Production Success:   {summary['prod_success_rate']:.1%}")
print(f"  Shadow Success:       {summary['shadow_success_rate']:.1%}")
print(f"  Shadow Faster:        {summary['shadow_faster_rate']:.1%} of requests")
print(f"  Avg Latency Diff:     {summary['avg_latency_diff_ms']:.1f}ms")

print("\n✅ Decision: Shadow agent matches production behavior")
print("   Ready to proceed to canary deployment (5% traffic)")

# COMMAND ----------

# DBTITLE 1,Pattern: Canary Deployment
# Canary Deployment: Gradual rollout with monitoring

import random

class CanaryController:
    """Controller for canary deployments."""
    
    def __init__(self, baseline_agent, canary_agent, canary_percentage: float, collector: MetricsCollector):
        self.baseline_agent = baseline_agent
        self.canary_agent = canary_agent
        self.canary_percentage = canary_percentage
        self.collector = collector
        self.baseline_results = []
        self.canary_results = []
    
    def process_request(self, order: Dict, message: str) -> Dict:
        """
        Route request to baseline or canary based on percentage.
        """
        # Randomly select canary based on percentage
        use_canary = random.random() < self.canary_percentage
        
        if use_canary:
            result = self.canary_agent.process_request(order, message)
            result['version'] = 'canary'
            self.canary_results.append(result)
            self.collector.record_metric('canary_request', 1.0)
        else:
            result = self.baseline_agent.process_request(order, message)
            result['version'] = 'baseline'
            self.baseline_results.append(result)
            self.collector.record_metric('baseline_request', 1.0)
        
        # Record success rate by version
        self.collector.record_metric(
            'request_success',
            1.0 if result.get('success') else 0.0,
            {'version': result['version']}
        )
        
        return result
    
    def get_comparison(self) -> Dict:
        """Compare canary vs baseline performance."""
        if not self.canary_results or not self.baseline_results:
            return {'error': 'Insufficient data'}
        
        baseline_success = sum(1 for r in self.baseline_results if r.get('success')) / len(self.baseline_results)
        canary_success = sum(1 for r in self.canary_results if r.get('success')) / len(self.canary_results)
        
        return {
            'baseline_requests': len(self.baseline_results),
            'canary_requests': len(self.canary_results),
            'baseline_success_rate': baseline_success,
            'canary_success_rate': canary_success,
            'success_rate_diff': canary_success - baseline_success,
            'canary_healthy': canary_success >= baseline_success * 0.95  # Within 5%
        }
    
    def should_promote(self) -> bool:
        """Decide if canary should be promoted to 100%."""
        comparison = self.get_comparison()
        
        if 'error' in comparison:
            return False
        
        # Promote if:
        # 1. Canary has enough traffic (at least 20 requests)
        # 2. Success rate is within 5% of baseline
        # 3. Preferably better than baseline
        
        return (
            comparison['canary_requests'] >= 20 and
            comparison['canary_healthy'] and
            comparison['success_rate_diff'] >= -0.05
        )
    
    def should_rollback(self) -> bool:
        """Decide if canary should be rolled back."""
        comparison = self.get_comparison()
        
        if 'error' in comparison:
            return False
        
        # Rollback if:
        # 1. Success rate drops more than 10%
        # 2. Canary has at least 10 requests (to avoid false positives)
        
        return (
            comparison['canary_requests'] >= 10 and
            comparison['success_rate_diff'] < -0.10
        )

# Simulate canary deployment
print("\n" + "="*70)
print("CANARY DEPLOYMENT: 10% Traffic to New Version")
print("="*70)

baseline_collector = MetricsCollector()
canary_collector = MetricsCollector()

baseline = InstrumentedAgent(baseline_collector)
canary = InstrumentedAgent(canary_collector)

canary_controller = CanaryController(
    baseline_agent=baseline,
    canary_agent=canary,
    canary_percentage=0.10,  # 10% canary
    collector=collector
)

# Simulate 50 requests
print("\nProcessing 50 requests with 10% canary traffic...")
for i in range(50):
    msg = random.choice([
        "Damaged product, need refund",
        "Cancel this order",
        "Change shipping address",
        "Track my package"
    ])
    canary_controller.process_request(test_order, msg)

comparison = canary_controller.get_comparison()

print(f"\nCanary Deployment Results:")
print(f"  Baseline Requests:    {comparison['baseline_requests']}")
print(f"  Canary Requests:      {comparison['canary_requests']}")
print(f"  Baseline Success:     {comparison['baseline_success_rate']:.1%}")
print(f"  Canary Success:       {comparison['canary_success_rate']:.1%}")
print(f"  Difference:           {comparison['success_rate_diff']:+.1%}")
print(f"  Canary Healthy:       {'✅ Yes' if comparison['canary_healthy'] else '❌ No'}")

# Decision
if canary_controller.should_rollback():
    print("\n🔴 DECISION: ROLLBACK CANARY")
    print("   Canary performance is significantly worse than baseline")
    print("   Rolling back to 100% baseline traffic")
elif canary_controller.should_promote():
    print("\n🟢 DECISION: PROMOTE CANARY")
    print("   Canary performing well, promoting to 25% traffic")
    print("   Will continue gradual rollout: 25% → 50% → 100%")
else:
    print("\n🟡 DECISION: CONTINUE MONITORING")
    print("   Need more data before making promotion/rollback decision")
    print("   Continue with 10% canary traffic")

# COMMAND ----------

# DBTITLE 1,Pattern: Self-Healing Agents
# Self-Healing Agent: Automatic fallback on degraded performance

class SelfHealingAgent:
    """Agent that monitors itself and triggers fallbacks."""
    
    def __init__(self, primary_agent, fallback_agent, collector: MetricsCollector):
        self.primary_agent = primary_agent
        self.fallback_agent = fallback_agent
        self.collector = collector
        self.recent_failures = []
        self.max_failures = 3  # Trigger fallback after 3 failures in window
        self.failure_window = 10  # Consider last 10 requests
    
    def process_request(self, order: Dict, message: str) -> Dict:
        """
        Process request with automatic fallback on high failure rate.
        """
        
        # Check if we should use fallback
        use_fallback = self._should_use_fallback()
        
        if use_fallback:
            self.collector.record_metric('fallback_triggered', 1.0)
            result = self.fallback_agent.process_request(order, message)
            result['used_fallback'] = True
        else:
            try:
                result = self.primary_agent.process_request(order, message)
                result['used_fallback'] = False
            except Exception as e:
                # Primary agent failed, use fallback
                self.collector.record_metric('primary_failure', 1.0)
                result = self.fallback_agent.process_request(order, message)
                result['used_fallback'] = True
                result['fallback_reason'] = 'exception'
        
        # Track success/failure
        self.recent_failures.append(not result.get('success', True))
        if len(self.recent_failures) > self.failure_window:
            self.recent_failures.pop(0)
        
        return result
    
    def _should_use_fallback(self) -> bool:
        """Decide if we should use fallback based on recent failures."""
        if len(self.recent_failures) < self.failure_window:
            return False
        
        recent_failure_count = sum(self.recent_failures[-self.failure_window:])
        failure_rate = recent_failure_count / self.failure_window
        
        # Use fallback if failure rate > 30%
        return failure_rate > 0.30
    
    def get_health_status(self) -> Dict:
        """Get current health status."""
        if not self.recent_failures:
            return {'status': 'unknown', 'failure_rate': 0.0}
        
        failure_count = sum(self.recent_failures)
        failure_rate = failure_count / len(self.recent_failures)
        
        if failure_rate < 0.10:
            status = 'healthy'
        elif failure_rate < 0.30:
            status = 'degraded'
        else:
            status = 'fallback_active'
        
        return {
            'status': status,
            'failure_rate': failure_rate,
            'recent_requests': len(self.recent_failures),
            'recent_failures': failure_count
        }

# Simulate self-healing behavior
print("\n" + "="*70)
print("SELF-HEALING AGENT: Automatic Fallback on Degradation")
print("="*70)

class FailingAgent:
    """Agent that simulates intermittent failures."""
    def __init__(self, collector, failure_rate=0.4):
        self.collector = collector
        self.failure_rate = failure_rate
    
    def process_request(self, order: Dict, message: str) -> Dict:
        # Simulate failure
        fails = random.random() < self.failure_rate
        trace_id = str(uuid.uuid4())
        
        if fails:
            return {
                'success': False,
                'trace_id': trace_id,
                'intent': 'unknown',
                'error': 'Primary agent degraded'
            }
        else:
            return {
                'success': True,
                'trace_id': trace_id,
                'intent': 'refund',
                'response': 'Processed successfully'
            }

class ReliableFallback:
    """Simpler, more reliable fallback agent."""
    def __init__(self, collector):
        self.collector = collector
    
    def process_request(self, order: Dict, message: str) -> Dict:
        trace_id = str(uuid.uuid4())
        return {
            'success': True,
            'trace_id': trace_id,
            'intent': 'fallback',
            'response': 'Processed by fallback agent (simpler logic)'
        }

primary_failing = FailingAgent(collector, failure_rate=0.40)
fallback = ReliableFallback(collector)

self_healing = SelfHealingAgent(primary_failing, fallback, collector)

# Process requests and watch self-healing activate
print("\nProcessing 20 requests with degraded primary agent (40% failure rate)...\n")

for i in range(20):
    result = self_healing.process_request(test_order, "Test message")
    health = self_healing.get_health_status()
    
    status_emoji = {
        'healthy': '🟢',
        'degraded': '🟡',
        'fallback_active': '🔴',
        'unknown': '⚪'
    }.get(health['status'], '⚪')
    
    fallback_marker = ' [FALLBACK]' if result.get('used_fallback') else ''
    success_marker = '✅' if result.get('success') else '❌'
    
    if (i + 1) % 5 == 0:  # Print every 5 requests
        print(f"Request {i+1:2d}: {success_marker} {status_emoji} {health['status']:15s} "
              f"(failure rate: {health['failure_rate']:.0%}){fallback_marker}")

final_health = self_healing.get_health_status()
print(f"\nFinal Health Status: {final_health['status'].upper()}")
print(f"  Failure Rate:      {final_health['failure_rate']:.1%}")
print(f"  Recent Failures:   {final_health['recent_failures']}/{final_health['recent_requests']}")

print("\n✅ Self-healing agent automatically switched to fallback")
print("   when primary agent failure rate exceeded 30%")
print("   This prevented cascade failures and maintained service availability")

# COMMAND ----------

# DBTITLE 1,Alerting Strategy
# MAGIC %md
# MAGIC ## Alerting Strategy
# MAGIC
# MAGIC ### The Problem with Alert Fatigue
# MAGIC
# MAGIC **Too many alerts = ignored alerts**
# MAGIC
# MAGIC Bad alerting:
# MAGIC * Pages for every error
# MAGIC * No context on severity
# MAGIC * No clear action
# MAGIC * Alerts that can't be acted upon
# MAGIC
# MAGIC ### Principles of Good Alerting
# MAGIC
# MAGIC **1. Alert on symptoms, not causes**
# MAGIC * ❌ Bad: "Model API returned 429"
# MAGIC * ✅ Good: "Task success rate dropped below 85%"
# MAGIC
# MAGIC **2. Include context**
# MAGIC * What metric breached?
# MAGIC * Current vs baseline value
# MAGIC * Link to dashboard/runbook
# MAGIC * Affected user segment
# MAGIC
# MAGIC **3. Make alerts actionable**
# MAGIC * What should the on-call engineer do?
# MAGIC * Is there an automatic mitigation?
# MAGIC * What's the business impact?
# MAGIC
# MAGIC **4. Use severity levels appropriately**
# MAGIC
# MAGIC | Severity | When to Use | Response Time |
# MAGIC |----------|-------------|---------------|
# MAGIC | P0 (Critical) | System down, data loss, security breach | Page immediately |
# MAGIC | P1 (High) | Task success < 85%, hallucination spike | Page during business hours |
# MAGIC | P2 (Medium) | Performance degraded, minor drift | Investigate within 24h |
# MAGIC | P3 (Low) | Trend alerts, capacity planning | Review weekly |
# MAGIC
# MAGIC ### Alert Thresholds
# MAGIC
# MAGIC **Task Success Rate**
# MAGIC * P1: < 90% (immediate attention)
# MAGIC * P2: < 95% (investigate soon)
# MAGIC
# MAGIC **Hallucination Rate**
# MAGIC * P1: > 5% (high risk)
# MAGIC * P2: > 2% (elevated risk)
# MAGIC
# MAGIC **Latency**
# MAGIC * P1: P99 > 10s (user experience degraded)
# MAGIC * P2: P95 > 5s (performance issue)
# MAGIC
# MAGIC **Cost/Token Usage**
# MAGIC * P2: Daily cost increase > 50%
# MAGIC * P3: Weekly trend increase > 20%
# MAGIC
# MAGIC **Distribution Drift**
# MAGIC * P2: PSI > 0.25 (major drift)
# MAGIC * P3: PSI > 0.10 (minor drift)

# COMMAND ----------

# DBTITLE 1,Alerting Framework
# Alerting framework with severity levels and context

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class AlertSeverity(Enum):
    P0_CRITICAL = 0
    P1_HIGH = 1
    P2_MEDIUM = 2
    P3_LOW = 3

@dataclass
class Alert:
    """Represents a monitoring alert."""
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    message: str
    context: Dict[str, Any]
    runbook_url: Optional[str] = None
    suggested_action: Optional[str] = None
    
    def format(self) -> str:
        """Format alert for notification."""
        severity_emoji = {
            AlertSeverity.P0_CRITICAL: '🔴',
            AlertSeverity.P1_HIGH: '🟠',
            AlertSeverity.P2_MEDIUM: '🟡',
            AlertSeverity.P3_LOW: '🔵'
        }
        
        lines = [
            f"{severity_emoji[self.severity]} {self.severity.name}: {self.message}",
            f"",
            f"Metric: {self.metric_name}",
            f"Current Value: {self.current_value}",
            f"Threshold: {self.threshold}",
            f""
        ]
        
        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if self.suggested_action:
            lines.append(f"Suggested Action: {self.suggested_action}")
        
        if self.runbook_url:
            lines.append(f"Runbook: {self.runbook_url}")
        
        return "\n".join(lines)

class AlertManager:
    """Manages alert evaluation and notification."""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = []
    
    def add_rule(self, rule: Dict):
        """Add an alert rule."""
        self.alert_rules.append(rule)
    
    def evaluate_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate metrics against alert rules."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            metric_name = rule['metric']
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            threshold = rule['threshold']
            comparison = rule.get('comparison', 'less_than')
            
            # Check if alert should trigger
            should_alert = False
            if comparison == 'less_than':
                should_alert = current_value < threshold
            elif comparison == 'greater_than':
                should_alert = current_value > threshold
            
            if should_alert:
                alert = Alert(
                    severity=rule['severity'],
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    message=rule['message'],
                    context=rule.get('context', {}),
                    runbook_url=rule.get('runbook_url'),
                    suggested_action=rule.get('suggested_action')
                )
                triggered_alerts.append(alert)
                self.alerts.append(alert)
        
        return triggered_alerts

# Example: Set up alert rules
print("\n" + "="*70)
print("ALERTING FRAMEWORK: Monitoring with Context")
print("="*70)

alert_mgr = AlertManager()

# Define alert rules
alert_mgr.add_rule({
    'metric': 'task_success_rate',
    'threshold': 0.90,
    'comparison': 'less_than',
    'severity': AlertSeverity.P1_HIGH,
    'message': 'Task success rate dropped below 90%',
    'context': {'impact': 'User experience degraded'},
    'suggested_action': 'Check recent deployments, review error logs, consider rollback',
    'runbook_url': 'https://docs.example.com/runbooks/low-success-rate'
})

alert_mgr.add_rule({
    'metric': 'hallucination_rate',
    'threshold': 0.05,
    'comparison': 'greater_than',
    'severity': AlertSeverity.P1_HIGH,
    'message': 'Hallucination rate exceeded 5%',
    'context': {'impact': 'Agent producing incorrect information'},
    'suggested_action': 'Review recent prompt changes, check grounding retrieval quality',
    'runbook_url': 'https://docs.example.com/runbooks/high-hallucination'
})

alert_mgr.add_rule({
    'metric': 'p99_latency_ms',
    'threshold': 10000,
    'comparison': 'greater_than',
    'severity': AlertSeverity.P2_MEDIUM,
    'message': 'P99 latency exceeded 10 seconds',
    'context': {'impact': 'Slow responses affecting user experience'},
    'suggested_action': 'Check model endpoint health, review tool execution times',
    'runbook_url': 'https://docs.example.com/runbooks/high-latency'
})

# Simulate metrics
current_metrics = {
    'task_success_rate': 0.87,  # Below threshold!
    'hallucination_rate': 0.03,
    'p99_latency_ms': 8500,
}

print("\nCurrent Metrics:")
for name, value in current_metrics.items():
    print(f"  {name}: {value}")

# Evaluate and trigger alerts
triggered = alert_mgr.evaluate_metrics(current_metrics)

print(f"\n{len(triggered)} Alert(s) Triggered:\n")
for alert in triggered:
    print(alert.format())
    print("\n" + "-"*70 + "\n")

if not triggered:
    print("✅ All metrics within normal ranges")

# COMMAND ----------

# DBTITLE 1,User Feedback as Observability
# MAGIC %md
# MAGIC ## User Feedback as Observability
# MAGIC
# MAGIC ### Why User Feedback Matters
# MAGIC
# MAGIC Telemetry tells you *what* happened. Users tell you *how it felt*.
# MAGIC
# MAGIC **Metrics can't capture:**
# MAGIC * Tone and appropriateness
# MAGIC * Whether the task was actually completed
# MAGIC * User satisfaction with the interaction
# MAGIC * Nuanced failure modes
# MAGIC
# MAGIC ### Types of User Feedback
# MAGIC
# MAGIC **1. Explicit Feedback**
# MAGIC * 👍 Thumbs up/down
# MAGIC * ⭐ Star ratings (1-5)
# MAGIC * 💬 Written comments
# MAGIC * 🚨 "Report an issue" button
# MAGIC
# MAGIC **2. Implicit Feedback**
# MAGIC * Task abandonment (user gives up mid-conversation)
# MAGIC * Re-queries (user rephrases same question)
# MAGIC * Escalation to human support
# MAGIC * Follow-up interactions
# MAGIC
# MAGIC **3. Structured Feedback**
# MAGIC * Post-interaction surveys
# MAGIC * A/B test comparisons
# MAGIC * Targeted feedback requests
# MAGIC
# MAGIC ### Integrating Feedback into Monitoring
# MAGIC
# MAGIC **Close the loop:**
# MAGIC 1. Collect feedback with trace IDs
# MAGIC 2. Join feedback to telemetry
# MAGIC 3. Identify patterns in negative feedback
# MAGIC 4. Convert feedback into test cases
# MAGIC 5. Measure improvement over time
# MAGIC
# MAGIC ### Example: Feedback-Driven Improvement
# MAGIC
# MAGIC **Week 1**: User thumbs-down rate = 15%
# MAGIC
# MAGIC **Investigation**: Join thumbs-down to traces
# MAGIC * 60% are on "multi-item refund" scenarios
# MAGIC * Agent refunding entire order instead of single item
# MAGIC
# MAGIC **Fix**: Improve item extraction logic
# MAGIC
# MAGIC **Week 2**: User thumbs-down rate = 8%
# MAGIC
# MAGIC **Result**: Feedback validated the fix

# COMMAND ----------

# DBTITLE 1,User Feedback Integration
# Integrating user feedback with telemetry

@dataclass
class UserFeedback:
    """Represents user feedback on an interaction."""
    trace_id: str
    feedback_type: str  # 'thumbs_up', 'thumbs_down', 'rating', 'comment'
    value: Any  # Boolean for thumbs, int for rating, str for comment
    timestamp: str
    user_id: Optional[str] = None
    
class FeedbackCollector:
    """Collects and analyzes user feedback."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.feedback = []
        self.metrics_collector = metrics_collector
    
    def record_feedback(self, feedback: UserFeedback):
        """Record user feedback."""
        self.feedback.append(feedback)
        
        # Convert to metrics
        if feedback.feedback_type == 'thumbs_up':
            self.metrics_collector.record_metric('user_satisfaction', 1.0, {'type': 'thumbs'})
        elif feedback.feedback_type == 'thumbs_down':
            self.metrics_collector.record_metric('user_satisfaction', 0.0, {'type': 'thumbs'})
        elif feedback.feedback_type == 'rating':
            # Normalize 1-5 rating to 0-1
            normalized = (feedback.value - 1) / 4.0
            self.metrics_collector.record_metric('user_satisfaction', normalized, {'type': 'rating'})
    
    def get_satisfaction_rate(self) -> float:
        """Calculate overall satisfaction rate."""
        thumbs = [f for f in self.feedback if f.feedback_type in ['thumbs_up', 'thumbs_down']]
        if not thumbs:
            return 0.0
        
        positive = sum(1 for f in thumbs if f.feedback_type == 'thumbs_up')
        return positive / len(thumbs)
    
    def analyze_negative_feedback(self, traces: List[Trace]) -> Dict:
        """Analyze traces associated with negative feedback."""
        # Get negative feedback trace IDs
        negative_trace_ids = {
            f.trace_id for f in self.feedback 
            if f.feedback_type == 'thumbs_down' or (f.feedback_type == 'rating' and f.value <= 2)
        }
        
        if not negative_trace_ids:
            return {'negative_count': 0}
        
        # Find traces with negative feedback
        negative_traces = [t for t in traces if t.trace_id in negative_trace_ids]
        
        # Analyze patterns
        intent_distribution = {}
        for trace in negative_traces:
            intent = trace.metadata.get('intent', 'unknown')
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
        
        return {
            'negative_count': len(negative_trace_ids),
            'intent_distribution': intent_distribution,
            'most_problematic_intent': max(intent_distribution.items(), key=lambda x: x[1])[0] if intent_distribution else None
        }

# Simulate user feedback collection
print("\n" + "="*70)
print("USER FEEDBACK: Closing the Observability Loop")
print("="*70)

feedback_collector = FeedbackCollector(collector)

# Simulate some interactions with feedback
print("\nSimulating 20 interactions with user feedback...\n")

for i in range(20):
    # Generate interaction
    order = {'order_id': f'ORD-{i}', 'status': 'Delivered', 'total': 19.99}
    message = random.choice([
        "Refund my damaged item",
        "Cancel this order",
        "Update shipping address"
    ])
    
    agent_result = agent.process_request(order, message)
    
    # Simulate user feedback (70% positive, 30% negative)
    if random.random() < 0.70:
        feedback = UserFeedback(
            trace_id=agent_result['trace_id'],
            feedback_type='thumbs_up',
            value=True,
            timestamp=datetime.utcnow().isoformat()
        )
    else:
        feedback = UserFeedback(
            trace_id=agent_result['trace_id'],
            feedback_type='thumbs_down',
            value=False,
            timestamp=datetime.utcnow().isoformat()
        )
    
    # Store intent in trace metadata for analysis
    matching_traces = [t for t in collector.traces if t.trace_id == agent_result['trace_id']]
    if matching_traces:
        matching_traces[0].metadata['intent'] = agent_result.get('intent')
    
    feedback_collector.record_feedback(feedback)

satisfaction_rate = feedback_collector.get_satisfaction_rate()
analysis = feedback_collector.analyze_negative_feedback(collector.traces)

print(f"Feedback Summary:")
print(f"  Total Feedback:        {len(feedback_collector.feedback)}")
print(f"  Satisfaction Rate:     {satisfaction_rate:.1%}")
print(f"  Negative Feedback:     {analysis['negative_count']}")

if analysis.get('intent_distribution'):
    print(f"\nNegative Feedback by Intent:")
    for intent, count in sorted(analysis['intent_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / analysis['negative_count']) * 100
        print(f"  {intent:15s}: {count:2d} ({pct:.0f}%)")
    
    print(f"\n🚨 Most Problematic Intent: {analysis['most_problematic_intent']}")
    print(f"   → Review traces for '{analysis['most_problematic_intent']}' intent")
    print(f"   → Add test cases covering these scenarios")
    print(f"   → Improve agent logic for this intent")

print("\n✅ Feedback integrated with telemetry for continuous improvement")

# COMMAND ----------

# DBTITLE 1,Cross-Functional Ownership
# MAGIC %md
# MAGIC ## Cross-Functional Ownership
# MAGIC
# MAGIC ### The Challenge
# MAGIC
# MAGIC Agent systems span multiple domains:
# MAGIC * **ML Engineers**: Model performance, embeddings, tool selection
# MAGIC * **Product Managers**: User experience, feature adoption
# MAGIC * **SREs**: Uptime, latency, infrastructure costs
# MAGIC * **Data Scientists**: Data quality, drift detection
# MAGIC * **Support Teams**: User satisfaction, escalation rates
# MAGIC
# MAGIC **Without clear ownership, metrics are ignored or duplicated.**
# MAGIC
# MAGIC ### RACI Matrix for Metrics
# MAGIC
# MAGIC **R**esponsible | **A**ccountable | **C**onsulted | **I**nformed
# MAGIC
# MAGIC | Metric | Responsible | Accountable | Consulted | Informed |
# MAGIC |--------|-------------|-------------|-----------|----------|
# MAGIC | Task success rate | ML Engineer | Product Manager | All | All |
# MAGIC | Hallucination rate | ML Engineer | ML Engineer | Product, Support | All |
# MAGIC | User satisfaction | Product Manager | Product Manager | ML, Support | All |
# MAGIC | P99 latency | SRE | SRE | ML, Product | All |
# MAGIC | Tool failure rate | ML Engineer | ML Engineer | SRE | All |
# MAGIC | Cost per interaction | ML Engineer | Engineering Manager | Finance | All |
# MAGIC | Distribution drift | Data Scientist | ML Engineer | All | All |
# MAGIC | Escalation rate | Support Team | Product Manager | ML | All |
# MAGIC
# MAGIC ### Metric Ownership Best Practices
# MAGIC
# MAGIC **1. Assign one Responsible owner per metric**
# MAGIC * They monitor the metric daily
# MAGIC * They respond to alerts
# MAGIC * They drive improvement
# MAGIC
# MAGIC **2. Define clear SLOs (Service Level Objectives)**
# MAGIC * Task success rate > 95%
# MAGIC * P99 latency < 5 seconds
# MAGIC * Hallucination rate < 2%
# MAGIC
# MAGIC **3. Review metrics cross-functionally**
# MAGIC * Weekly: Review trends and anomalies
# MAGIC * Monthly: Assess SLO compliance
# MAGIC * Quarterly: Update SLOs and priorities
# MAGIC
# MAGIC **4. Tie metrics to business outcomes**
# MAGIC * User satisfaction → retention
# MAGIC * Task success rate → support ticket deflection
# MAGIC * Latency → task completion rate
# MAGIC
# MAGIC ### Example: Metric-Driven Incident Response
# MAGIC
# MAGIC **Alert**: Task success rate dropped from 95% to 85%
# MAGIC
# MAGIC **Step 1**: ML Engineer (Responsible) investigates
# MAGIC * Reviews traces and error patterns
# MAGIC * Identifies that a recent model update broke multi-item handling
# MAGIC
# MAGIC **Step 2**: Product Manager (Accountable) assesses impact
# MAGIC * 10% of users affected
# MAGIC * High-value customer segment
# MAGIC * Business impact: ~$50K in support costs if not fixed
# MAGIC
# MAGIC **Step 3**: Cross-functional decision
# MAGIC * Rollback model update immediately
# MAGIC * Create regression test for multi-item scenarios
# MAGIC * Schedule hotfix for tomorrow
# MAGIC
# MAGIC **Step 4**: Follow-up
# MAGIC * SRE: Ensure rollback is smooth
# MAGIC * Support: Notify affected users
# MAGIC * Data Scientist: Add multi-item cases to validation set
# MAGIC
# MAGIC **Result**: Task success back to 95% within 2 hours

# COMMAND ----------

# DBTITLE 1,Monitoring Dashboard Summary
# Summary: What to put on your monitoring dashboard

class MonitoringDashboard:
    """Blueprint for a production monitoring dashboard."""
    
    @staticmethod
    def get_dashboard_layout() -> Dict:
        """Returns the structure of a comprehensive monitoring dashboard."""
        return {
            "sections": [
                {
                    "name": "Health Overview",
                    "metrics": [
                        {"name": "Task Success Rate", "threshold": "95%", "alert": "P1 if < 90%"},
                        {"name": "P99 Latency", "threshold": "5s", "alert": "P2 if > 10s"},
                        {"name": "Uptime", "threshold": "99.9%", "alert": "P0 if < 99%"},
                        {"name": "Error Rate", "threshold": "< 1%", "alert": "P1 if > 5%"}
                    ]
                },
                {
                    "name": "Quality Metrics",
                    "metrics": [
                        {"name": "Hallucination Rate", "threshold": "< 2%", "alert": "P1 if > 5%"},
                        {"name": "Tool Accuracy", "threshold": "> 95%", "alert": "P2 if < 90%"},
                        {"name": "Response Coherence", "threshold": "> 90%", "alert": "P2 if < 85%"},
                        {"name": "User Satisfaction", "threshold": "> 85%", "alert": "P2 if < 80%"}
                    ]
                },
                {
                    "name": "Distribution Monitoring",
                    "metrics": [
                        {"name": "Query Length KS Test", "threshold": "< 0.1", "alert": "P3 if > 0.1"},
                        {"name": "Token Distribution KL", "threshold": "< 0.5", "alert": "P2 if > 0.5"},
                        {"name": "Tool Usage PSI", "threshold": "< 0.25", "alert": "P2 if > 0.25"},
                        {"name": "Intent Distribution", "threshold": "stable", "alert": "P3 if shifting"}
                    ]
                },
                {
                    "name": "Resource Utilization",
                    "metrics": [
                        {"name": "Token Usage (daily)", "threshold": "baseline +/- 20%", "alert": "P3 if > +50%"},
                        {"name": "Cost per Interaction", "threshold": "< $0.10", "alert": "P2 if > $0.20"},
                        {"name": "Tool Call Rate", "threshold": "stable", "alert": "P3 if spike"},
                        {"name": "Retry Rate", "threshold": "< 5%", "alert": "P2 if > 10%"}
                    ]
                },
                {
                    "name": "User Experience",
                    "metrics": [
                        {"name": "Task Completion Time", "threshold": "< 30s", "alert": "P2 if > 60s"},
                        {"name": "Abandonment Rate", "threshold": "< 10%", "alert": "P2 if > 15%"},
                        {"name": "Escalation Rate", "threshold": "< 5%", "alert": "P2 if > 10%"},
                        {"name": "Requery Rate", "threshold": "< 15%", "alert": "P3 if > 20%"}
                    ]
                }
            ],
            "visualizations": [
                "Time series: Task success rate over time",
                "Heatmap: Intent distribution by hour",
                "Distribution: Latency histogram (P50, P95, P99)",
                "Pie chart: Tool usage breakdown",
                "Line chart: Cost trends (daily/weekly)",
                "Scatter: User satisfaction vs latency"
            ],
            "alerts": [
                "P0: System down, data loss",
                "P1: Task success < 90%, hallucination > 5%",
                "P2: Performance degraded, drift detected",
                "P3: Trend alerts, capacity planning"
            ]
        }

print("\n" + "="*70)
print("MONITORING DASHBOARD: Complete Blueprint")
print("="*70)

dashboard = MonitoringDashboard.get_dashboard_layout()

for section in dashboard['sections']:
    print(f"\n📋 {section['name']}")
    print("  " + "-" * 65)
    for metric in section['metrics']:
        print(f"  • {metric['name']:25s} | Threshold: {metric['threshold']:15s} | {metric['alert']}")

print(f"\n📈 Recommended Visualizations:")
for i, viz in enumerate(dashboard['visualizations'], 1):
    print(f"  {i}. {viz}")

print(f"\n🚨 Alert Severity Levels:")
for alert in dashboard['alerts']:
    print(f"  • {alert}")

print("\n" + "="*70)
print("✅ Complete monitoring system ready for production!")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Summary and Best Practices
# MAGIC %md
# MAGIC ## Summary: Production Monitoring Best Practices
# MAGIC
# MAGIC ### ✅ Key Takeaways
# MAGIC
# MAGIC **1. Monitor the right things**
# MAGIC * Infrastructure metrics (latency, uptime) are necessary but not sufficient
# MAGIC * Add semantic metrics (task success, hallucination rate, tool accuracy)
# MAGIC * Track user experience (satisfaction, abandonment, escalation)
# MAGIC
# MAGIC **2. Detect drift early**
# MAGIC * Use statistical tests (KS, KL, PSI) to catch distribution shifts
# MAGIC * Monitor input distributions (query length, intent mix)
# MAGIC * Track output distributions (token usage, tool usage patterns)
# MAGIC
# MAGIC **3. Deploy safely**
# MAGIC * Start with shadow mode (zero user risk)
# MAGIC * Graduate to canary deployments (limited blast radius)
# MAGIC * Implement self-healing (automatic fallback on degradation)
# MAGIC
# MAGIC **4. Alert intelligently**
# MAGIC * Alert on symptoms, not causes
# MAGIC * Provide context and suggested actions
# MAGIC * Use severity levels appropriately
# MAGIC * Avoid alert fatigue
# MAGIC
# MAGIC **5. Close the feedback loop**
# MAGIC * Collect user feedback with trace IDs
# MAGIC * Join feedback to telemetry
# MAGIC * Convert feedback into test cases
# MAGIC * Measure improvement over time
# MAGIC
# MAGIC **6. Define ownership**
# MAGIC * Use RACI matrix for metrics
# MAGIC * Assign one Responsible owner per metric
# MAGIC * Review cross-functionally
# MAGIC * Tie metrics to business outcomes
# MAGIC
# MAGIC ### 🛠️ Tools Mentioned
# MAGIC
# MAGIC **Open Source Monitoring Stack**:
# MAGIC * **OpenTelemetry**: Instrumentation (traces, metrics, logs)
# MAGIC * **Loki**: Log aggregation
# MAGIC * **Tempo**: Distributed tracing
# MAGIC * **Grafana**: Dashboards and visualization
# MAGIC * **Prometheus**: Metrics storage and alerting
# MAGIC
# MAGIC **Specialized Agent Monitoring**:
# MAGIC * **LangSmith**: LangChain tracing
# MAGIC * **Phoenix**: Arize AI observability
# MAGIC * **Weights & Biases**: Experiment tracking
# MAGIC * **MLflow**: Model tracking and registry
# MAGIC
# MAGIC ### 📚 Next Steps
# MAGIC
# MAGIC 1. **Instrument your agent**: Add tracing to all components
# MAGIC 2. **Define your metrics**: Pick 5-10 key metrics to start
# MAGIC 3. **Set up dashboards**: Visualize metrics in real-time
# MAGIC 4. **Configure alerts**: Start with P1 alerts only
# MAGIC 5. **Deploy gradually**: Use shadow mode → canary → full rollout
# MAGIC 6. **Collect feedback**: Add thumbs up/down to your UI
# MAGIC 7. **Review regularly**: Weekly metrics review with team
# MAGIC
# MAGIC ### 🔗 Integration with Validation
# MAGIC
# MAGIC This monitoring notebook completes the cycle:
# MAGIC
# MAGIC **[Validation and Measurement]** notebook:
# MAGIC * Offline evaluation
# MAGIC * Test set construction
# MAGIC * Component testing
# MAGIC * Quality gates
# MAGIC
# MAGIC **[Monitoring in Production]** (this notebook):
# MAGIC * Real-time telemetry
# MAGIC * Distribution drift detection
# MAGIC * User feedback
# MAGIC * Safe deployment patterns
# MAGIC
# MAGIC **Together**: Build agents you can trust in production.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Ready to Monitor Your Agent?
# MAGIC
# MAGIC 🚀 **Start by running the cells above** to:
# MAGIC * Understand instrumentation basics
# MAGIC * Implement drift detection
# MAGIC * Try shadow mode and canary patterns
# MAGIC * Set up alerting framework
# MAGIC * Integrate user feedback
# MAGIC
# MAGIC 💬 **Questions or want to discuss monitoring strategies?** Reach out to your team!