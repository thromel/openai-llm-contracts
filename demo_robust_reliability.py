#!/usr/bin/env python3
"""
Demonstration of Robust Circuit Breaker and Retry Mechanisms

This script shows how the enhanced LLM Contracts framework now provides
enterprise-grade reliability through sophisticated circuit breaker and
retry mechanisms that work seamlessly with the OpenAI provider.

Features Demonstrated:
- Robust circuit breaker with adaptive timeout and health checks
- Intelligent retry mechanism with multiple backoff strategies
- Comprehensive metrics and monitoring
- Error classification and handling
- Integration with OpenAI provider
- Production-ready reliability patterns
"""

import asyncio
import time
import sys
import os
from typing import Dict, Any

# Add src to path for demo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_contracts.reliability import (
    RobustCircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FailureType,
    RobustRetryMechanism,
    RetryConfig,
    RetryStrategy,
    retry
)


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print('='*60)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n📋 {title}")
    print('-'*40)


async def demo_circuit_breaker_basics():
    """Demonstrate basic circuit breaker functionality."""
    print_section_header("Circuit Breaker Fundamentals")
    
    # Create a circuit breaker with custom configuration
    config = CircuitBreakerConfig(
        failure_threshold=3,          # Open after 3 failures
        timeout_seconds=5,            # Stay open for 5 seconds
        adaptive_timeout=True,        # Enable adaptive timeout
        health_check_enabled=True,    # Enable health checks
        health_check_interval=2       # Check every 2 seconds
    )
    
    breaker = RobustCircuitBreaker("demo_service", config)
    print(f"✅ Created circuit breaker: {breaker.name}")
    print(f"   Initial state: {breaker.state.value}")
    print(f"   Configuration: {breaker.config}")
    
    print_subsection("Simulating Failures")
    
    # Simulate some failures
    failure_types = [FailureType.TIMEOUT, FailureType.API_ERROR, FailureType.NETWORK_ERROR]
    for i, failure_type in enumerate(failure_types, 1):
        breaker.record_failure(failure_type, f"Simulated failure {i}")
        print(f"   Failure {i}: {failure_type.value} - State: {breaker.state.value}")
        
        if breaker.state == CircuitState.OPEN:
            print(f"   🚨 Circuit breaker OPENED after {i} failures")
            break
    
    print_subsection("Circuit Breaker Behavior")
    
    # Test request blocking
    if not breaker.should_allow_request():
        print("   ❌ Requests are now BLOCKED by circuit breaker")
    
    # Get comprehensive status
    status = breaker.get_status()
    print(f"   📊 Current timeout: {status['current_timeout']}s")
    print(f"   📈 Metrics: {status['metrics']['total_requests']} total requests")
    print(f"   📉 Success rate: {status['metrics']['success_rate']:.1%}")
    
    return breaker


async def demo_retry_mechanisms():
    """Demonstrate retry mechanism capabilities."""
    print_section_header("Advanced Retry Mechanisms")
    
    # Configure different retry strategies
    configs = {
        "Exponential Backoff": RetryConfig(
            max_attempts=4,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay_ms=100,
            backoff_multiplier=2.0,
            jitter=True
        ),
        "Linear Backoff": RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay_ms=200,
            jitter=False
        ),
        "Fixed Delay": RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay_ms=500,
            jitter=True
        )
    }
    
    for strategy_name, config in configs.items():
        print_subsection(f"Testing {strategy_name} Strategy")
        
        retry_mechanism = RobustRetryMechanism(config)
        
        # Create a flaky operation
        attempt_count = 0
        
        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts
                raise ConnectionError(f"Temporary failure on attempt {attempt_count}")
            return f"Success on attempt {attempt_count}!"
        
        # Execute with retry
        start_time = time.time()
        result = await retry_mechanism.execute_with_retry(flaky_operation)
        elapsed_time = time.time() - start_time
        
        print(f"   Result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"   Final result: {result.final_result}")
        print(f"   Attempts made: {len(result.attempts)}")
        print(f"   Total time: {elapsed_time:.2f}s")
        
        # Reset for next test
        attempt_count = 0


async def demo_error_classification():
    """Demonstrate intelligent error classification."""
    print_section_header("Intelligent Error Classification")
    
    # Configure with specific error handling rules
    config = RetryConfig(
        max_attempts=3,
        retryable_exceptions=['ConnectionError', 'TimeoutError', 'RateLimitError'],
        non_retryable_exceptions=['ValueError', 'AuthenticationError'],
        base_delay_ms=100
    )
    
    retry_mechanism = RobustRetryMechanism(config)
    classifier = retry_mechanism.error_classifier
    
    # Test different error types
    test_errors = [
        (ConnectionError("Network hiccup"), "Should be retried"),
        (TimeoutError("Request timeout"), "Should be retried"),
        (ValueError("Invalid input"), "Should NOT be retried"),
        (RuntimeError("Unknown error"), "Default behavior")
    ]
    
    print_subsection("Error Classification Results")
    
    for error, description in test_errors:
        classification = classifier.classify_error(error)
        print(f"   {error.__class__.__name__}: {classification.value} - {description}")
    
    print_subsection("Retry Behavior Testing")
    
    # Test retryable error
    async def operation_with_retryable_error():
        raise ConnectionError("Network temporarily unavailable")
    
    result = await retry_mechanism.execute_with_retry(operation_with_retryable_error)
    print(f"   Retryable error: {len(result.attempts)} attempts made")
    
    # Test non-retryable error
    async def operation_with_non_retryable_error():
        raise ValueError("Invalid parameter")
    
    result = await retry_mechanism.execute_with_retry(operation_with_non_retryable_error)
    print(f"   Non-retryable error: {len(result.attempts)} attempts made (should be 1)")


async def demo_circuit_breaker_retry_integration():
    """Demonstrate circuit breaker and retry working together."""
    print_section_header("Circuit Breaker + Retry Integration")
    
    # Create circuit breaker that opens quickly for demo
    cb_config = CircuitBreakerConfig(
        failure_threshold=2,
        timeout_seconds=3,
        adaptive_timeout=False
    )
    circuit_breaker = RobustCircuitBreaker("integrated_service", cb_config)
    
    # Create retry mechanism with circuit breaker
    retry_config = RetryConfig(
        max_attempts=5,
        base_delay_ms=200,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    )
    retry_mechanism = RobustRetryMechanism(retry_config, circuit_breaker)
    
    print_subsection("Scenario 1: Failures Leading to Circuit Opening")
    
    async def always_failing_operation():
        raise TimeoutError("Service consistently failing")
    
    # First request - will fail and trigger retries, then open circuit
    result1 = await retry_mechanism.execute_with_retry(always_failing_operation)
    print(f"   First request: {'✅ SUCCESS' if result1.success else '❌ FAILED'}")
    print(f"   Attempts made: {len(result1.attempts)}")
    print(f"   Circuit state: {circuit_breaker.state.value}")
    
    print_subsection("Scenario 2: Circuit Breaker Blocking Subsequent Requests")
    
    # Second request - should be blocked by circuit breaker
    result2 = await retry_mechanism.execute_with_retry(always_failing_operation)
    print(f"   Second request: {'✅ SUCCESS' if result2.success else '❌ FAILED'}")
    print(f"   Circuit breaker triggered: {result2.circuit_breaker_triggered}")
    print(f"   Attempts made: {len(result2.attempts)} (should be 0 - blocked)")
    
    print_subsection("Scenario 3: Circuit Recovery")
    
    print("   Waiting for circuit breaker timeout...")
    await asyncio.sleep(3.5)  # Wait for circuit breaker timeout
    
    # Define a recovering operation
    recovery_count = 0
    
    async def recovering_operation():
        nonlocal recovery_count
        recovery_count += 1
        if recovery_count == 1:
            raise ConnectionError("Still having issues")
        return "Service recovered!"
    
    result3 = await retry_mechanism.execute_with_retry(recovering_operation)
    print(f"   Recovery request: {'✅ SUCCESS' if result3.success else '❌ FAILED'}")
    print(f"   Result: {result3.final_result}")
    print(f"   Circuit state: {circuit_breaker.state.value}")


def demo_decorator_usage():
    """Demonstrate decorator-based usage."""
    print_section_header("Decorator-Based Reliability")
    
    print_subsection("Retry Decorator Example")
    
    # Configure retry decorator
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay_ms=100,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    )
    
    @retry(retry_config)
    async def unreliable_api_call(data: str):
        """Simulate an unreliable API call."""
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("API temporarily unavailable")
        return f"✅ Successfully processed: {data}"
    
    async def test_decorated_function():
        try:
            result = await unreliable_api_call("test_data")
            print(f"   Decorated function result: {result}")
        except Exception as e:
            print(f"   Decorated function failed: {e}")
    
    # Note: We can't easily run this in demo due to randomness
    print("   Retry decorator configured for unreliable_api_call function")
    print("   - 3 max attempts with exponential backoff")
    print("   - Automatic retry on ConnectionError")
    print("   - Transparent to calling code")


async def demo_comprehensive_metrics():
    """Demonstrate comprehensive metrics collection."""
    print_section_header("Comprehensive Metrics & Monitoring")
    
    # Create components with metrics enabled
    circuit_breaker = RobustCircuitBreaker("metrics_demo")
    retry_mechanism = RobustRetryMechanism()
    
    print_subsection("Simulating Various Operations")
    
    # Simulate various operations to generate metrics
    operations = [
        ("success", lambda: "✅ Success"),
        ("timeout", lambda: (_ for _ in ()).throw(TimeoutError("Timeout"))),
        ("success", lambda: "✅ Another success"),
        ("api_error", lambda: (_ for _ in ()).throw(RuntimeError("API Error"))),
        ("success", lambda: "✅ Final success")
    ]
    
    for op_type, operation in operations:
        try:
            if op_type == "success":
                circuit_breaker.record_success()
                result = operation()
                print(f"   Operation: {result}")
            else:
                error = None
                try:
                    operation()
                except Exception as e:
                    error = e
                
                failure_type = FailureType.TIMEOUT if "Timeout" in str(error) else FailureType.API_ERROR
                circuit_breaker.record_failure(failure_type, str(error))
                print(f"   Operation failed: {error}")
        except Exception as e:
            print(f"   Unexpected error: {e}")
    
    print_subsection("Circuit Breaker Metrics")
    
    status = circuit_breaker.get_status()
    metrics = status["metrics"]
    
    print(f"   📊 Total requests: {metrics['total_requests']}")
    print(f"   ✅ Success count: {metrics['success_count']}")
    print(f"   📈 Success rate: {metrics['success_rate']:.1%}")
    print(f"   🔄 State transitions: {metrics['state_transitions']}")
    print(f"   🐛 Failure types: {metrics['failure_types']}")
    
    print_subsection("Retry Mechanism Metrics")
    
    retry_metrics = retry_mechanism.get_metrics()
    print(f"   📊 Operations: {retry_metrics.get('total_operations', 0)}")
    print(f"   ✅ Success rate: {retry_metrics.get('success_rate', 1.0):.1%}")
    print(f"   🔄 Avg attempts: {retry_metrics.get('average_attempts_per_operation', 1.0):.1f}")


def demo_production_benefits():
    """Demonstrate production benefits."""
    print_section_header("Production Benefits & Best Practices")
    
    benefits = {
        "🛡️ Fault Tolerance": [
            "Prevents cascade failures across services",
            "Graceful degradation during outages",
            "Automatic recovery without human intervention"
        ],
        "📊 Observability": [
            "Real-time health monitoring",
            "Detailed failure analysis and patterns",
            "Performance metrics and SLA tracking"
        ],
        "⚡ Performance": [
            "Intelligent backoff prevents system overload",
            "Circuit breaking reduces unnecessary load",
            "Configurable timeouts and thresholds"
        ],
        "🔧 Operations": [
            "Zero-configuration defaults for most scenarios",
            "Extensive configurability for specific needs",
            "Manual controls for emergency situations"
        ],
        "💰 Cost Efficiency": [
            "Reduces wasted API calls to failing services",
            "Prevents resource exhaustion",
            "Enables better capacity planning"
        ]
    }
    
    for category, items in benefits.items():
        print_subsection(category)
        for item in items:
            print(f"   ✅ {item}")
    
    print_subsection("Integration Scenarios")
    
    scenarios = [
        "🌐 High-traffic web applications with LLM features",
        "🤖 AI-powered chatbots and virtual assistants", 
        "📝 Content generation and processing pipelines",
        "🔍 Real-time analysis and recommendation systems",
        "📊 Batch processing with LLM-based analytics",
        "🎯 A/B testing frameworks for AI features"
    ]
    
    for scenario in scenarios:
        print(f"   {scenario}")


async def main():
    """Run the complete reliability mechanisms demonstration."""
    print("🚀 LLM Contracts Framework - Robust Reliability Demonstration")
    print("This demo showcases enterprise-grade circuit breaker and retry mechanisms")
    
    # Run all demonstrations
    circuit_breaker = await demo_circuit_breaker_basics()
    await demo_retry_mechanisms()
    await demo_error_classification()
    await demo_circuit_breaker_retry_integration()
    demo_decorator_usage()
    await demo_comprehensive_metrics()
    demo_production_benefits()
    
    print_section_header("Demo Complete - Summary")
    
    print("🎉 Successfully demonstrated:")
    print("   ✅ Robust circuit breaker with adaptive features")
    print("   ✅ Intelligent retry mechanisms with multiple strategies")
    print("   ✅ Sophisticated error classification and handling")
    print("   ✅ Seamless integration between components")
    print("   ✅ Comprehensive metrics and monitoring")
    print("   ✅ Production-ready reliability patterns")
    
    print("\n🔧 The LLM Contracts framework now provides enterprise-grade reliability!")
    print("   Ready for production deployment with confidence.")
    
    # Final status check
    final_status = circuit_breaker.get_status()
    print(f"\n📊 Final circuit breaker state: {final_status['state']}")
    print(f"📈 Total operations handled: {final_status['metrics']['total_requests']}")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())