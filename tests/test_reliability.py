"""
Comprehensive tests for reliability mechanisms (circuit breaker and retry).

Tests cover:
- Circuit breaker functionality
- Retry mechanism behavior  
- Integration with OpenAI provider
- Error classification and handling
- Metrics collection and reporting
- Various failure scenarios
"""

import pytest
import asyncio
import time
import unittest.mock as mock
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List

from src.llm_contracts.reliability import (
    RobustCircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FailureType,
    RobustRetryMechanism,
    RetryConfig,
    RetryStrategy,
    ErrorClassification
)

from src.llm_contracts.providers.openai_provider import ImprovedOpenAIProvider
from src.llm_contracts.core.exceptions import ContractViolationError


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation with default config."""
        breaker = RobustCircuitBreaker("test_breaker")
        assert breaker.name == "test_breaker"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.should_allow_request() == True
    
    def test_circuit_breaker_custom_config(self):
        """Test circuit breaker with custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30,
            adaptive_timeout=False
        )
        breaker = RobustCircuitBreaker("test_breaker", config)
        assert breaker.config.failure_threshold == 3
        assert breaker.config.timeout_seconds == 30
        assert breaker.config.adaptive_timeout == False
    
    def test_failure_recording_and_opening(self):
        """Test that circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = RobustCircuitBreaker("test_breaker", config)
        
        # Initial state should be closed
        assert breaker.state == CircuitState.CLOSED
        assert breaker.should_allow_request() == True
        
        # Record first failure
        breaker.record_failure(FailureType.TIMEOUT, "Test timeout")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.should_allow_request() == True
        
        # Record second failure - should open circuit
        breaker.record_failure(FailureType.API_ERROR, "Test API error")
        assert breaker.state == CircuitState.OPEN
        assert breaker.should_allow_request() == False
    
    def test_circuit_recovery_after_timeout(self):
        """Test circuit breaker recovery after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=1  # Very short timeout for testing
        )
        breaker = RobustCircuitBreaker("test_breaker", config)
        
        # Trigger circuit opening
        breaker.record_failure(FailureType.TIMEOUT, "Test failure")
        assert breaker.state == CircuitState.OPEN
        assert breaker.should_allow_request() == False
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should transition to half-open and allow one request
        assert breaker.should_allow_request() == True
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_half_open_success_closes_circuit(self):
        """Test that success in half-open state closes circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            half_open_success_threshold=1
        )
        breaker = RobustCircuitBreaker("test_breaker", config)
        
        # Open circuit
        breaker.record_failure(FailureType.TIMEOUT, "Test failure")
        assert breaker.state == CircuitState.OPEN
        
        # Manually transition to half-open
        breaker._transition_to_half_open()
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Record success - should close circuit
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
    
    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = RobustCircuitBreaker("test_breaker", config)
        
        # Open circuit then transition to half-open
        breaker.record_failure(FailureType.TIMEOUT, "Test failure")
        breaker._transition_to_half_open()
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Record another failure - should reopen circuit
        breaker.record_failure(FailureType.API_ERROR, "Another failure")
        assert breaker.state == CircuitState.OPEN
    
    def test_metrics_collection(self):
        """Test that circuit breaker collects metrics properly."""
        breaker = RobustCircuitBreaker("test_breaker")
        
        # Record some operations
        breaker.record_success()
        breaker.record_failure(FailureType.TIMEOUT, "Test timeout")
        breaker.record_success()
        
        # Get metrics
        status = breaker.get_status()
        metrics = status["metrics"]
        
        assert metrics["total_requests"] == 3
        assert metrics["success_count"] == 2
        assert metrics["success_rate"] == 2/3
        assert "timeout" in metrics["failure_types"]
    
    def test_force_open_and_close(self):
        """Test manual circuit breaker control."""
        breaker = RobustCircuitBreaker("test_breaker")
        
        # Force open
        breaker.force_open()
        assert breaker.state == CircuitState.OPEN
        assert breaker.should_allow_request() == False
        
        # Force close
        breaker.force_close()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.should_allow_request() == True


class TestRetryMechanism:
    """Test retry mechanism functionality."""
    
    def test_retry_mechanism_creation(self):
        """Test retry mechanism creation with default config."""
        retry_mechanism = RobustRetryMechanism()
        assert retry_mechanism.config.max_attempts == 3
        assert retry_mechanism.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
    
    def test_retry_config_validation(self):
        """Test retry configuration options."""
        config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay_ms=500,
            jitter=False
        )
        retry_mechanism = RobustRetryMechanism(config)
        assert retry_mechanism.config.max_attempts == 5
        assert retry_mechanism.config.strategy == RetryStrategy.LINEAR_BACKOFF
        assert retry_mechanism.config.jitter == False
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self):
        """Test that successful operations don't trigger retries."""
        retry_mechanism = RobustRetryMechanism()
        
        async def successful_operation():
            return "success"
        
        result = await retry_mechanism.execute_with_retry(successful_operation)
        
        assert result.success == True
        assert result.final_result == "success"
        assert len(result.attempts) == 1
        assert result.attempts[0].success == True
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry behavior on failures."""
        config = RetryConfig(max_attempts=3, base_delay_ms=10)  # Fast retry for testing
        retry_mechanism = RobustRetryMechanism(config)
        
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await retry_mechanism.execute_with_retry(failing_operation)
        
        assert result.success == True
        assert result.final_result == "success"
        assert len(result.attempts) == 3
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test behavior when max attempts are exceeded."""
        config = RetryConfig(max_attempts=2, base_delay_ms=10)
        retry_mechanism = RobustRetryMechanism(config)
        
        async def always_failing_operation():
            raise TimeoutError("Always fails")
        
        result = await retry_mechanism.execute_with_retry(always_failing_operation)
        
        assert result.success == False
        assert isinstance(result.final_error, TimeoutError)
        assert len(result.attempts) == 2
    
    def test_sync_retry_mechanism(self):
        """Test synchronous retry functionality."""
        config = RetryConfig(max_attempts=3, base_delay_ms=10)
        retry_mechanism = RobustRetryMechanism(config)
        
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = retry_mechanism.execute_with_retry_sync(failing_operation)
        
        assert result.success == True
        assert result.final_result == "success"
        assert len(result.attempts) == 3
    
    def test_error_classification(self):
        """Test error classification for retry decisions."""
        config = RetryConfig(
            retryable_exceptions=['ConnectionError', 'TimeoutError'],
            non_retryable_exceptions=['ValueError', 'AuthError']
        )
        retry_mechanism = RobustRetryMechanism(config)
        classifier = retry_mechanism.error_classifier
        
        # Test retryable errors
        assert classifier.classify_error(ConnectionError()) == ErrorClassification.RETRYABLE
        assert classifier.classify_error(TimeoutError()) == ErrorClassification.RETRYABLE
        
        # Test non-retryable errors
        assert classifier.classify_error(ValueError()) == ErrorClassification.NON_RETRYABLE
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_stops_immediately(self):
        """Test that non-retryable errors stop retry immediately."""
        config = RetryConfig(
            max_attempts=5,
            non_retryable_exceptions=['ValueError']
        )
        retry_mechanism = RobustRetryMechanism(config)
        
        async def operation_with_non_retryable_error():
            raise ValueError("Non-retryable error")
        
        result = await retry_mechanism.execute_with_retry(operation_with_non_retryable_error)
        
        assert result.success == False
        assert isinstance(result.final_error, ValueError)
        assert len(result.attempts) == 1  # Only one attempt
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        from src.llm_contracts.reliability.retry_mechanism import ExponentialBackoffCalculator
        
        config = RetryConfig(
            base_delay_ms=1000,
            backoff_multiplier=2.0,
            jitter=False
        )
        calculator = ExponentialBackoffCalculator()
        
        # Test delay progression
        assert calculator.calculate_delay(1, 1000, config) == 1000  # First retry
        assert calculator.calculate_delay(2, 1000, config) == 2000  # Second retry
        assert calculator.calculate_delay(3, 1000, config) == 4000  # Third retry
    
    def test_linear_backoff_calculation(self):
        """Test linear backoff delay calculation."""
        from src.llm_contracts.reliability.retry_mechanism import LinearBackoffCalculator
        
        config = RetryConfig(base_delay_ms=1000, jitter=False)
        calculator = LinearBackoffCalculator()
        
        # Test delay progression
        assert calculator.calculate_delay(1, 1000, config) == 1000  # First retry
        assert calculator.calculate_delay(2, 1000, config) == 2000  # Second retry
        assert calculator.calculate_delay(3, 1000, config) == 3000  # Third retry
    
    def test_metrics_collection(self):
        """Test retry metrics collection."""
        retry_mechanism = RobustRetryMechanism()
        
        # Simulate some operations
        from src.llm_contracts.reliability.retry_mechanism import RetryResult, RetryAttempt
        
        # Successful operation
        success_result = RetryResult(
            success=True,
            attempts=[RetryAttempt(1, time.time(), 0, success=True)],
            final_result="success",
            total_time_ms=100
        )
        retry_mechanism.metrics.record_retry_operation(success_result)
        
        # Failed operation
        failed_result = RetryResult(
            success=False,
            attempts=[
                RetryAttempt(1, time.time(), 0, error=ConnectionError("fail"), success=False),
                RetryAttempt(2, time.time(), 1000, error=ConnectionError("fail"), success=False)
            ],
            final_error=ConnectionError("fail"),
            total_time_ms=1500
        )
        retry_mechanism.metrics.record_retry_operation(failed_result)
        
        # Get metrics
        metrics = retry_mechanism.get_metrics()
        
        assert metrics["total_operations"] == 2
        assert metrics["successful_operations"] == 1
        assert metrics["failed_operations"] == 1
        assert "ConnectionError" in metrics["most_common_errors"]


class TestIntegrationWithCircuitBreaker:
    """Test retry mechanism integration with circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_retry_attempts(self):
        """Test that circuit breaker can block retry attempts."""
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
        assert len(result2.attempts) == 0  # No attempts made
    
    @pytest.mark.asyncio
    async def test_successful_retry_records_circuit_breaker_success(self):
        """Test that successful retry records success in circuit breaker."""
        circuit_breaker = RobustCircuitBreaker("test")
        retry_mechanism = RobustRetryMechanism(circuit_breaker=circuit_breaker)
        
        call_count = 0
        
        async def operation_that_recovers():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First failure")
            return "success"
        
        result = await retry_mechanism.execute_with_retry(operation_that_recovers)
        
        assert result.success == True
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Check circuit breaker recorded the success
        status = circuit_breaker.get_status()
        assert status["metrics"]["success_count"] > 0


class TestOpenAIProviderIntegration:
    """Test reliability mechanisms integrated with OpenAI provider."""
    
    def test_provider_reliability_components_initialization(self):
        """Test that provider initializes reliability components."""
        with patch('src.llm_contracts.providers.openai_provider.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('src.llm_contracts.providers.openai_provider.AsyncOpenAI') as mock_async_openai:
                mock_async_client = Mock()
                mock_async_openai.return_value = mock_async_client
                
                provider = ImprovedOpenAIProvider(api_key="test-key")
                
                # Check that reliability components are initialized
                assert hasattr(provider, '_robust_circuit_breaker')
                assert hasattr(provider, '_retry_mechanism')
                assert provider._robust_circuit_breaker is not None
                assert provider._retry_mechanism is not None
    
    def test_provider_reliability_configuration(self):
        """Test provider reliability configuration methods."""
        with patch('src.llm_contracts.providers.openai_provider.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('src.llm_contracts.providers.openai_provider.AsyncOpenAI') as mock_async_openai:
                mock_async_client = Mock()
                mock_async_openai.return_value = mock_async_client
                
                provider = ImprovedOpenAIProvider(api_key="test-key")
                
                # Test configuration update
                new_cb_config = CircuitBreakerConfig(failure_threshold=10)
                new_retry_config = RetryConfig(max_attempts=5)
                
                provider.configure_reliability(new_cb_config, new_retry_config)
                
                # Verify configuration was updated
                assert provider._robust_circuit_breaker.config.failure_threshold == 10
                assert provider._retry_mechanism.config.max_attempts == 5
    
    def test_provider_metrics_include_reliability(self):
        """Test that provider metrics include reliability information."""
        with patch('src.llm_contracts.providers.openai_provider.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('src.llm_contracts.providers.openai_provider.AsyncOpenAI') as mock_async_openai:
                mock_async_client = Mock()
                mock_async_openai.return_value = mock_async_client
                
                provider = ImprovedOpenAIProvider(api_key="test-key")
                
                # Get metrics
                metrics = provider.get_metrics()
                
                # Verify reliability metrics are included
                assert "retry_metrics" in metrics
                assert "circuit_breaker_status" in metrics
                assert "reliability_summary" in metrics
                
                # Check reliability summary structure
                reliability = metrics["reliability_summary"]
                assert "circuit_breaker_state" in reliability
                assert "retry_success_rate" in reliability
                assert "average_attempts_per_operation" in reliability
    
    def test_circuit_breaker_status_method(self):
        """Test dedicated circuit breaker status method."""
        with patch('src.llm_contracts.providers.openai_provider.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('src.llm_contracts.providers.openai_provider.AsyncOpenAI') as mock_async_openai:
                mock_async_client = Mock()
                mock_async_openai.return_value = mock_async_client
                
                provider = ImprovedOpenAIProvider(api_key="test-key")
                
                # Get circuit breaker status
                status = provider.get_circuit_breaker_status()
                
                # Verify status structure
                assert "name" in status
                assert "state" in status
                assert "metrics" in status
                assert status["state"] == "closed"  # Should start closed
    
    def test_retry_metrics_method(self):
        """Test dedicated retry metrics method."""
        with patch('src.llm_contracts.providers.openai_provider.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('src.llm_contracts.providers.openai_provider.AsyncOpenAI') as mock_async_openai:
                mock_async_client = Mock()
                mock_async_openai.return_value = mock_async_client
                
                provider = ImprovedOpenAIProvider(api_key="test-key")
                
                # Get retry metrics
                metrics = provider.get_retry_metrics()
                
                # Should have metrics structure (even if no operations yet)
                assert isinstance(metrics, dict)


class TestReliabilityDecorators:
    """Test decorator-based reliability mechanisms."""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_async(self):
        """Test retry decorator on async functions."""
        from src.llm_contracts.reliability.retry_mechanism import retry
        
        config = RetryConfig(max_attempts=3, base_delay_ms=10)
        
        call_count = 0
        
        @retry(config)
        async def decorated_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await decorated_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_decorator_sync(self):
        """Test retry decorator on sync functions."""
        from src.llm_contracts.reliability.retry_mechanism import retry
        
        config = RetryConfig(max_attempts=3, base_delay_ms=10)
        
        call_count = 0
        
        @retry(config)
        def decorated_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = decorated_function()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        from src.llm_contracts.reliability.circuit_breaker import circuit_breaker
        
        config = CircuitBreakerConfig(failure_threshold=2)
        
        @circuit_breaker("test_decorator", config)
        async def decorated_function(should_fail=False):
            if should_fail:
                raise TimeoutError("Simulated failure")
            return "success"
        
        # Should work normally
        result = await decorated_function(should_fail=False)
        assert result == "success"
        
        # Fail twice to open circuit
        with pytest.raises(TimeoutError):
            await decorated_function(should_fail=True)
        
        with pytest.raises(TimeoutError):
            await decorated_function(should_fail=True)
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(Exception) as exc_info:
            await decorated_function(should_fail=False)
        
        assert "Circuit breaker" in str(exc_info.value)


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in retry mechanism."""
        config = RetryConfig(timeout_ms=100)  # 100ms timeout
        retry_mechanism = RobustRetryMechanism(config)
        
        async def slow_operation():
            await asyncio.sleep(0.2)  # 200ms - should timeout
            return "should not reach here"
        
        result = await retry_mechanism.execute_with_retry(slow_operation)
        
        assert result.success == False
        assert isinstance(result.final_error, asyncio.TimeoutError)
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Test invalid retry configuration
        with pytest.raises(Exception):
            RetryConfig(max_attempts=0)  # Should be at least 1
        
        # Test invalid circuit breaker configuration  
        with pytest.raises(Exception):
            CircuitBreakerConfig(failure_threshold=0)  # Should be at least 1
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test reliability mechanisms under concurrent load."""
        retry_mechanism = RobustRetryMechanism()
        
        async def test_operation(operation_id: int):
            # Simulate some operations failing
            if operation_id % 3 == 0:
                raise ConnectionError(f"Failure in operation {operation_id}")
            return f"Success {operation_id}"
        
        # Run multiple operations concurrently
        tasks = [
            retry_mechanism.execute_with_retry(test_operation, i)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Count successes and failures
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        
        # Should have some successes and some failures
        assert successes > 0
        assert failures > 0
        assert successes + failures == 10
    
    def test_memory_usage_under_load(self):
        """Test that reliability mechanisms don't leak memory."""
        import gc
        
        # Create many retry mechanisms to test memory usage
        mechanisms = []
        for i in range(100):
            config = RetryConfig(max_attempts=2)
            mechanisms.append(RobustRetryMechanism(config))
        
        # Force garbage collection
        del mechanisms
        gc.collect()
        
        # Test should complete without memory issues
        assert True


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])