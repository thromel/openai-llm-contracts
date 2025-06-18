#!/usr/bin/env python3
"""
Standalone test for reliability mechanisms.
This avoids import issues by testing the reliability components directly.
"""

import asyncio
import time
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_contracts.reliability.circuit_breaker import (
    RobustCircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FailureType
)

from llm_contracts.reliability.retry_mechanism import (
    RobustRetryMechanism,
    RetryConfig,
    RetryStrategy,
    ErrorClassification
)


def test_circuit_breaker_basic_functionality():
    """Test basic circuit breaker functionality."""
    print("🧪 Testing Circuit Breaker Basic Functionality")
    
    # Test creation
    breaker = RobustCircuitBreaker("test_breaker")
    assert breaker.name == "test_breaker"
    assert breaker.state == CircuitState.CLOSED
    print("✅ Circuit breaker creation works")
    
    # Test failure recording
    config = CircuitBreakerConfig(failure_threshold=2)
    breaker = RobustCircuitBreaker("test_breaker", config)
    
    # Should start closed
    assert breaker.should_allow_request() == True
    
    # Record first failure
    breaker.record_failure(FailureType.TIMEOUT, "Test timeout")
    assert breaker.state == CircuitState.CLOSED
    
    # Record second failure - should open
    breaker.record_failure(FailureType.API_ERROR, "Test API error")
    assert breaker.state == CircuitState.OPEN
    assert breaker.should_allow_request() == False
    print("✅ Circuit breaker opens after threshold failures")
    
    # Test manual control
    breaker.force_close()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.should_allow_request() == True
    print("✅ Manual circuit breaker control works")


async def test_retry_mechanism_basic_functionality():
    """Test basic retry mechanism functionality."""
    print("🧪 Testing Retry Mechanism Basic Functionality")
    
    # Test successful operation (no retry needed)
    retry_mechanism = RobustRetryMechanism()
    
    async def successful_operation():
        return "success"
    
    result = await retry_mechanism.execute_with_retry(successful_operation)
    assert result.success == True
    assert result.final_result == "success"
    assert len(result.attempts) == 1
    print("✅ Successful operations don't trigger retries")
    
    # Test retry on failure
    config = RetryConfig(max_attempts=3, base_delay_ms=10)
    retry_mechanism = RobustRetryMechanism(config)
    
    call_count = 0
    
    async def failing_then_succeeding_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    call_count = 0  # Reset
    result = await retry_mechanism.execute_with_retry(failing_then_succeeding_operation)
    assert result.success == True
    assert result.final_result == "success"
    assert len(result.attempts) == 3
    assert call_count == 3
    print("✅ Retry mechanism works on failures")
    
    # Test max attempts exceeded
    config = RetryConfig(max_attempts=2, base_delay_ms=10)
    retry_mechanism = RobustRetryMechanism(config)
    
    async def always_failing_operation():
        raise TimeoutError("Always fails")
    
    result = await retry_mechanism.execute_with_retry(always_failing_operation)
    assert result.success == False
    assert isinstance(result.final_error, TimeoutError)
    assert len(result.attempts) == 2
    print("✅ Retry respects max attempts limit")


def test_sync_retry_mechanism():
    """Test synchronous retry functionality."""
    print("🧪 Testing Synchronous Retry Mechanism")
    
    config = RetryConfig(max_attempts=3, base_delay_ms=10)
    retry_mechanism = RobustRetryMechanism(config)
    
    call_count = 0
    
    def failing_then_succeeding_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    result = retry_mechanism.execute_with_retry_sync(failing_then_succeeding_operation)
    assert result.success == True
    assert result.final_result == "success"
    assert len(result.attempts) == 3
    print("✅ Synchronous retry mechanism works")


async def test_circuit_breaker_integration_with_retry():
    """Test circuit breaker integration with retry mechanism."""
    print("🧪 Testing Circuit Breaker Integration with Retry")
    
    # Create circuit breaker that opens quickly
    cb_config = CircuitBreakerConfig(failure_threshold=1)
    circuit_breaker = RobustCircuitBreaker("test", cb_config)
    
    # Create retry mechanism with circuit breaker
    retry_config = RetryConfig(max_attempts=5, base_delay_ms=10)
    retry_mechanism = RobustRetryMechanism(retry_config, circuit_breaker)
    
    async def always_failing_operation():
        raise ConnectionError("Always fails")
    
    # First operation should fail and open circuit breaker
    result = await retry_mechanism.execute_with_retry(always_failing_operation)
    assert result.success == False
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Second operation should be blocked by circuit breaker
    result2 = await retry_mechanism.execute_with_retry(always_failing_operation)
    assert result2.success == False
    assert result2.circuit_breaker_triggered == True
    print("✅ Circuit breaker blocks retry attempts when open")


def test_error_classification():
    """Test error classification for retry decisions."""
    print("🧪 Testing Error Classification")
    
    config = RetryConfig(
        retryable_exceptions=['ConnectionError', 'TimeoutError'],
        non_retryable_exceptions=['ValueError', 'AuthenticationError']
    )
    retry_mechanism = RobustRetryMechanism(config)
    classifier = retry_mechanism.error_classifier
    
    # Test retryable errors
    assert classifier.classify_error(ConnectionError()) == ErrorClassification.RETRYABLE
    assert classifier.classify_error(TimeoutError()) == ErrorClassification.RETRYABLE
    
    # Test non-retryable errors
    assert classifier.classify_error(ValueError()) == ErrorClassification.NON_RETRYABLE
    print("✅ Error classification works correctly")


def test_backoff_calculations():
    """Test different backoff calculation strategies."""
    print("🧪 Testing Backoff Calculations")
    
    from llm_contracts.reliability.retry_mechanism import (
        ExponentialBackoffCalculator,
        LinearBackoffCalculator,
        FixedDelayCalculator
    )
    
    config = RetryConfig(base_delay_ms=1000, backoff_multiplier=2.0, jitter=False)
    
    # Test exponential backoff
    exp_calc = ExponentialBackoffCalculator()
    assert exp_calc.calculate_delay(1, 1000, config) == 1000
    assert exp_calc.calculate_delay(2, 1000, config) == 2000
    assert exp_calc.calculate_delay(3, 1000, config) == 4000
    
    # Test linear backoff
    lin_calc = LinearBackoffCalculator()
    assert lin_calc.calculate_delay(1, 1000, config) == 1000
    assert lin_calc.calculate_delay(2, 1000, config) == 2000
    assert lin_calc.calculate_delay(3, 1000, config) == 3000
    
    # Test fixed delay
    fixed_calc = FixedDelayCalculator()
    assert fixed_calc.calculate_delay(1, 1000, config) == 1000
    assert fixed_calc.calculate_delay(2, 1000, config) == 1000
    assert fixed_calc.calculate_delay(3, 1000, config) == 1000
    
    print("✅ All backoff calculations work correctly")


def test_metrics_collection():
    """Test metrics collection functionality."""
    print("🧪 Testing Metrics Collection")
    
    # Test circuit breaker metrics
    breaker = RobustCircuitBreaker("test_breaker")
    breaker.record_success()
    breaker.record_failure(FailureType.TIMEOUT, "Test timeout")
    breaker.record_success()
    
    status = breaker.get_status()
    metrics = status["metrics"]
    assert metrics["total_requests"] == 3
    assert metrics["success_count"] == 2
    print("✅ Circuit breaker metrics collection works")
    
    # Test retry metrics
    retry_mechanism = RobustRetryMechanism()
    metrics = retry_mechanism.get_metrics()
    assert isinstance(metrics, dict)
    print("✅ Retry mechanism metrics collection works")


async def test_performance_under_load():
    """Test performance under concurrent load."""
    print("🧪 Testing Performance Under Load")
    
    retry_mechanism = RobustRetryMechanism()
    
    async def test_operation(operation_id: int):
        # Simulate some operations failing
        if operation_id % 3 == 0:
            raise ConnectionError(f"Failure in operation {operation_id}")
        await asyncio.sleep(0.001)  # Small delay to simulate work
        return f"Success {operation_id}"
    
    # Run multiple operations concurrently
    start_time = time.time()
    tasks = [
        retry_mechanism.execute_with_retry(test_operation, i)
        for i in range(20)
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time
    
    # Count successes and failures
    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)
    
    print(f"   Processed 20 operations in {elapsed_time:.2f}s")
    print(f"   Successes: {successes}, Failures: {failures}")
    assert successes > 0
    assert failures > 0
    assert successes + failures == 20
    assert elapsed_time < 5.0  # Should complete reasonably quickly
    print("✅ Performance under load is acceptable")


async def run_all_tests():
    """Run all reliability mechanism tests."""
    print("🚀 Starting Comprehensive Reliability Mechanism Tests")
    print("=" * 60)
    
    try:
        # Synchronous tests
        test_circuit_breaker_basic_functionality()
        test_sync_retry_mechanism()
        test_error_classification()
        test_backoff_calculations()
        test_metrics_collection()
        
        # Asynchronous tests
        await test_retry_mechanism_basic_functionality()
        await test_circuit_breaker_integration_with_retry()
        await test_performance_under_load()
        
        print("\n🎉 All tests passed!")
        print("✅ Circuit breaker and retry mechanisms are working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n📊 Test Summary:")
        print("  - Circuit breaker functionality: ✅")
        print("  - Retry mechanism functionality: ✅")
        print("  - Integration between components: ✅")
        print("  - Error classification: ✅")
        print("  - Backoff calculations: ✅")
        print("  - Metrics collection: ✅")
        print("  - Performance under load: ✅")
        print("\n🔧 Reliability mechanisms are robust and ready for production!")
        exit(0)
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
        exit(1)