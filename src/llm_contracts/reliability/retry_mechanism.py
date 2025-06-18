"""
Advanced Retry Mechanism for LLM API Contracts

This module provides a comprehensive retry mechanism designed to work with
the circuit breaker for maximum reliability in LLM API interactions.

Features:
- Multiple retry strategies (exponential backoff, linear, custom)
- Error classification and selective retry logic
- Integration with circuit breaker
- Comprehensive metrics and monitoring
- Jitter to prevent thundering herd
- Custom retry conditions and failure handling
"""

import time
import asyncio
import logging
import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Type
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Different retry strategies available."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    CUSTOM = "custom"


class ErrorClassification(Enum):
    """Classification of errors for retry decisions."""
    RETRYABLE = "retryable"          # Should retry (timeouts, rate limits, temporary failures)
    NON_RETRYABLE = "non_retryable"  # Should not retry (auth errors, invalid input)
    CIRCUIT_BREAKING = "circuit_breaking"  # Should trigger circuit breaker


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay_ms: float = 1000.0
    max_delay_ms: float = 60000.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    timeout_ms: Optional[float] = None
    
    # Error classification
    retryable_exceptions: List[str] = field(default_factory=lambda: [
        'TimeoutError', 'ConnectionError', 'RateLimitError', 'ServiceUnavailableError'
    ])
    non_retryable_exceptions: List[str] = field(default_factory=lambda: [
        'AuthenticationError', 'ValidationError', 'PermissionError'
    ])
    
    # Advanced features
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 502, 503, 504])
    custom_retry_condition: Optional[Callable[[Exception], bool]] = None


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""
    attempt_number: int
    timestamp: float
    delay_ms: float
    error: Optional[Exception] = None
    success: bool = False
    response_time_ms: Optional[float] = None


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    attempts: List[RetryAttempt]
    final_result: Any = None
    final_error: Optional[Exception] = None
    total_time_ms: float = 0
    circuit_breaker_triggered: bool = False


class BackoffCalculator(ABC):
    """Abstract base class for backoff calculation strategies."""
    
    @abstractmethod
    def calculate_delay(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        """Calculate delay for given attempt number."""
        pass


class ExponentialBackoffCalculator(BackoffCalculator):
    """Exponential backoff with optional jitter."""
    
    def calculate_delay(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        delay = base_delay * (config.backoff_multiplier ** (attempt - 1))
        delay = min(delay, config.max_delay_ms)
        
        if config.jitter:
            # Add jitter to prevent thundering herd
            jitter = delay * config.jitter_factor * (random.random() * 2 - 1)
            delay = max(0, delay + jitter)
        
        return delay


class LinearBackoffCalculator(BackoffCalculator):
    """Linear backoff with optional jitter."""
    
    def calculate_delay(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        delay = base_delay * attempt
        delay = min(delay, config.max_delay_ms)
        
        if config.jitter:
            jitter = delay * config.jitter_factor * (random.random() * 2 - 1)
            delay = max(0, delay + jitter)
        
        return delay


class FixedDelayCalculator(BackoffCalculator):
    """Fixed delay with optional jitter."""
    
    def calculate_delay(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        delay = base_delay
        
        if config.jitter:
            jitter = delay * config.jitter_factor * (random.random() * 2 - 1)
            delay = max(0, delay + jitter)
        
        return delay


class ErrorClassifier:
    """Classifies errors for retry decisions."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def classify_error(self, error: Exception) -> ErrorClassification:
        """Classify an error to determine retry behavior."""
        error_name = error.__class__.__name__
        
        # Check custom retry condition first
        if self.config.custom_retry_condition:
            try:
                if self.config.custom_retry_condition(error):
                    return ErrorClassification.RETRYABLE
                else:
                    return ErrorClassification.NON_RETRYABLE
            except Exception as e:
                logger.warning(f"Custom retry condition failed: {e}")
        
        # Check explicitly non-retryable errors
        if error_name in self.config.non_retryable_exceptions:
            return ErrorClassification.NON_RETRYABLE
        
        # Check explicitly retryable errors
        if error_name in self.config.retryable_exceptions:
            return ErrorClassification.RETRYABLE
        
        # Check HTTP status codes if available
        if hasattr(error, 'status_code') and error.status_code in self.config.retry_on_status_codes:
            return ErrorClassification.RETRYABLE
        
        # Default classification based on error type
        if 'timeout' in error_name.lower() or 'connection' in error_name.lower():
            return ErrorClassification.RETRYABLE
        elif 'rate' in error_name.lower() and 'limit' in error_name.lower():
            return ErrorClassification.RETRYABLE
        elif 'auth' in error_name.lower() or 'permission' in error_name.lower():
            return ErrorClassification.NON_RETRYABLE
        else:
            # Conservative default: don't retry unknown errors
            return ErrorClassification.NON_RETRYABLE
    
    def should_trigger_circuit_breaker(self, error: Exception) -> bool:
        """Determine if this error should trigger circuit breaker."""
        classification = self.classify_error(error)
        return classification in [ErrorClassification.RETRYABLE, ErrorClassification.CIRCUIT_BREAKING]


class RetryMetrics:
    """Comprehensive metrics collection for retry operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.retry_history: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_after_retry_count = 0
        self.total_retry_operations = 0
        self.total_delay_time_ms = 0
        
    def record_retry_operation(self, result: RetryResult):
        """Record a complete retry operation."""
        self.retry_history.append(result)
        self.total_retry_operations += 1
        self.total_delay_time_ms += result.total_time_ms
        
        if result.success and len(result.attempts) > 1:
            self.success_after_retry_count += 1
        
        # Record error types
        for attempt in result.attempts:
            if attempt.error:
                error_name = attempt.error.__class__.__name__
                self.error_counts[error_name] += 1
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics."""
        if not self.retry_history:
            return {"message": "No retry operations recorded"}
        
        recent_operations = list(self.retry_history)
        successful_operations = [op for op in recent_operations if op.success]
        failed_operations = [op for op in recent_operations if not op.success]
        
        total_attempts = sum(len(op.attempts) for op in recent_operations)
        
        return {
            "total_operations": len(recent_operations),
            "successful_operations": len(successful_operations),
            "failed_operations": len(failed_operations),
            "success_rate": len(successful_operations) / len(recent_operations),
            "success_after_retry_rate": self.success_after_retry_count / max(1, self.total_retry_operations),
            "average_attempts_per_operation": total_attempts / len(recent_operations),
            "average_total_time_ms": sum(op.total_time_ms for op in recent_operations) / len(recent_operations),
            "most_common_errors": dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "circuit_breaker_triggers": sum(1 for op in recent_operations if op.circuit_breaker_triggered)
        }


class RobustRetryMechanism:
    """
    Enterprise-grade retry mechanism for LLM APIs.
    
    Features:
    - Multiple backoff strategies
    - Intelligent error classification
    - Circuit breaker integration
    - Comprehensive metrics
    - Async/sync support
    - Timeout handling
    - Jitter to prevent thundering herd
    """
    
    def __init__(self, 
                 config: Optional[RetryConfig] = None,
                 circuit_breaker: Optional[Any] = None):
        self.config = config or RetryConfig()
        self.circuit_breaker = circuit_breaker
        
        # Initialize components
        self.error_classifier = ErrorClassifier(self.config)
        self.metrics = RetryMetrics()
        
        # Backoff calculator factory
        self.backoff_calculators = {
            RetryStrategy.EXPONENTIAL_BACKOFF: ExponentialBackoffCalculator(),
            RetryStrategy.LINEAR_BACKOFF: LinearBackoffCalculator(),
            RetryStrategy.FIXED_DELAY: FixedDelayCalculator(),
        }
        
        logger.info(f"Retry mechanism initialized with config: {self.config}")
    
    async def execute_with_retry(self, 
                                operation: Callable,
                                *args,
                                **kwargs) -> RetryResult:
        """Execute an async operation with retry logic."""
        start_time = time.time()
        attempts = []
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            # Check circuit breaker before attempt
            if self.circuit_breaker and not self.circuit_breaker.should_allow_request():
                logger.warning("Circuit breaker blocked retry attempt")
                return RetryResult(
                    success=False,
                    attempts=attempts,
                    final_error=Exception("Circuit breaker is open"),
                    total_time_ms=(time.time() - start_time) * 1000,
                    circuit_breaker_triggered=True
                )
            
            try:
                # Execute the operation with timeout if configured
                if self.config.timeout_ms:
                    result = await asyncio.wait_for(
                        operation(*args, **kwargs),
                        timeout=self.config.timeout_ms / 1000
                    )
                else:
                    result = await operation(*args, **kwargs)
                
                # Success!
                attempt_time = (time.time() - attempt_start) * 1000
                attempts.append(RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=attempt_start,
                    delay_ms=0,
                    success=True,
                    response_time_ms=attempt_time
                ))
                
                # Record success in circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                retry_result = RetryResult(
                    success=True,
                    attempts=attempts,
                    final_result=result,
                    total_time_ms=(time.time() - start_time) * 1000
                )
                
                self.metrics.record_retry_operation(retry_result)
                return retry_result
            
            except Exception as error:
                attempt_time = (time.time() - attempt_start) * 1000
                
                # Classify the error
                error_classification = self.error_classifier.classify_error(error)
                
                # Record failure in circuit breaker
                if self.circuit_breaker and self.error_classifier.should_trigger_circuit_breaker(error):
                    from .circuit_breaker import FailureType
                    failure_type = self._map_error_to_failure_type(error)
                    self.circuit_breaker.record_failure(
                        failure_type=failure_type,
                        error_message=str(error)
                    )
                
                # Record attempt
                attempts.append(RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=attempt_start,
                    delay_ms=0,
                    error=error,
                    success=False,
                    response_time_ms=attempt_time
                ))
                
                # Check if we should retry
                if (error_classification == ErrorClassification.NON_RETRYABLE or 
                    attempt_num >= self.config.max_attempts):
                    # Final failure
                    retry_result = RetryResult(
                        success=False,
                        attempts=attempts,
                        final_error=error,
                        total_time_ms=(time.time() - start_time) * 1000
                    )
                    
                    self.metrics.record_retry_operation(retry_result)
                    return retry_result
                
                # Calculate delay for next attempt
                if attempt_num < self.config.max_attempts:
                    delay_ms = self._calculate_delay(attempt_num)
                    attempts[-1].delay_ms = delay_ms
                    
                    logger.info(f"Retry attempt {attempt_num} failed with {error.__class__.__name__}: {error}")
                    logger.info(f"Retrying in {delay_ms:.0f}ms (attempt {attempt_num + 1}/{self.config.max_attempts})")
                    
                    await asyncio.sleep(delay_ms / 1000)
        
        # Should not reach here, but handle it gracefully
        retry_result = RetryResult(
            success=False,
            attempts=attempts,
            final_error=Exception("Max attempts reached"),
            total_time_ms=(time.time() - start_time) * 1000
        )
        
        self.metrics.record_retry_operation(retry_result)
        return retry_result
    
    def execute_with_retry_sync(self, 
                               operation: Callable,
                               *args,
                               **kwargs) -> RetryResult:
        """Execute a sync operation with retry logic."""
        start_time = time.time()
        attempts = []
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            # Check circuit breaker before attempt
            if self.circuit_breaker and not self.circuit_breaker.should_allow_request():
                logger.warning("Circuit breaker blocked retry attempt")
                return RetryResult(
                    success=False,
                    attempts=attempts,
                    final_error=Exception("Circuit breaker is open"),
                    total_time_ms=(time.time() - start_time) * 1000,
                    circuit_breaker_triggered=True
                )
            
            try:
                result = operation(*args, **kwargs)
                
                # Success!
                attempt_time = (time.time() - attempt_start) * 1000
                attempts.append(RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=attempt_start,
                    delay_ms=0,
                    success=True,
                    response_time_ms=attempt_time
                ))
                
                # Record success in circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                retry_result = RetryResult(
                    success=True,
                    attempts=attempts,
                    final_result=result,
                    total_time_ms=(time.time() - start_time) * 1000
                )
                
                self.metrics.record_retry_operation(retry_result)
                return retry_result
            
            except Exception as error:
                attempt_time = (time.time() - attempt_start) * 1000
                
                # Classify the error
                error_classification = self.error_classifier.classify_error(error)
                
                # Record failure in circuit breaker
                if self.circuit_breaker and self.error_classifier.should_trigger_circuit_breaker(error):
                    from .circuit_breaker import FailureType
                    failure_type = self._map_error_to_failure_type(error)
                    self.circuit_breaker.record_failure(
                        failure_type=failure_type,
                        error_message=str(error)
                    )
                
                # Record attempt
                attempts.append(RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=attempt_start,
                    delay_ms=0,
                    error=error,
                    success=False,
                    response_time_ms=attempt_time
                ))
                
                # Check if we should retry
                if (error_classification == ErrorClassification.NON_RETRYABLE or 
                    attempt_num >= self.config.max_attempts):
                    # Final failure
                    retry_result = RetryResult(
                        success=False,
                        attempts=attempts,
                        final_error=error,
                        total_time_ms=(time.time() - start_time) * 1000
                    )
                    
                    self.metrics.record_retry_operation(retry_result)
                    return retry_result
                
                # Calculate delay for next attempt
                if attempt_num < self.config.max_attempts:
                    delay_ms = self._calculate_delay(attempt_num)
                    attempts[-1].delay_ms = delay_ms
                    
                    logger.info(f"Retry attempt {attempt_num} failed with {error.__class__.__name__}: {error}")
                    logger.info(f"Retrying in {delay_ms:.0f}ms (attempt {attempt_num + 1}/{self.config.max_attempts})")
                    
                    time.sleep(delay_ms / 1000)
        
        # Should not reach here, but handle it gracefully
        retry_result = RetryResult(
            success=False,
            attempts=attempts,
            final_error=Exception("Max attempts reached"),
            total_time_ms=(time.time() - start_time) * 1000
        )
        
        self.metrics.record_retry_operation(retry_result)
        return retry_result
    
    def _calculate_delay(self, attempt_num: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.config.strategy == RetryStrategy.CUSTOM:
            # For custom strategy, use exponential backoff as fallback
            calculator = self.backoff_calculators[RetryStrategy.EXPONENTIAL_BACKOFF]
        else:
            calculator = self.backoff_calculators[self.config.strategy]
        
        return calculator.calculate_delay(attempt_num, self.config.base_delay_ms, self.config)
    
    def _map_error_to_failure_type(self, error: Exception):
        """Map error to circuit breaker failure type."""
        from .circuit_breaker import FailureType
        
        error_name = error.__class__.__name__.lower()
        
        if 'timeout' in error_name:
            return FailureType.TIMEOUT
        elif 'rate' in error_name and 'limit' in error_name:
            return FailureType.RATE_LIMIT
        elif 'validation' in error_name or 'contract' in error_name:
            return FailureType.VALIDATION_ERROR
        elif 'api' in error_name:
            return FailureType.API_ERROR
        elif 'network' in error_name or 'connection' in error_name:
            return FailureType.NETWORK_ERROR
        else:
            return FailureType.UNKNOWN
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current retry metrics."""
        return self.metrics.get_retry_statistics()
    
    def reset_metrics(self):
        """Reset retry metrics."""
        self.metrics = RetryMetrics()


# Decorator for easy retry integration
def retry(config: Optional[RetryConfig] = None, circuit_breaker: Optional[Any] = None):
    """Decorator to add retry functionality to functions."""
    
    def decorator(func):
        retry_mechanism = RobustRetryMechanism(config, circuit_breaker)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                result = await retry_mechanism.execute_with_retry(func, *args, **kwargs)
                if result.success:
                    return result.final_result
                else:
                    raise result.final_error
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                result = retry_mechanism.execute_with_retry_sync(func, *args, **kwargs)
                if result.success:
                    return result.final_result
                else:
                    raise result.final_error
            return sync_wrapper
    
    return decorator


# Usage examples
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example usage of the robust retry mechanism."""
        
        # Configure retry behavior
        config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay_ms=1000,
            max_delay_ms=30000,
            jitter=True,
            retryable_exceptions=['ConnectionError', 'TimeoutError', 'RateLimitError']
        )
        
        # Create retry mechanism
        retry_mechanism = RobustRetryMechanism(config)
        
        # Example operation that fails sometimes
        async def flaky_operation(attempt_id: int):
            if attempt_id % 3 == 0:  # Fail 2/3 of the time
                raise ConnectionError(f"Simulated failure for attempt {attempt_id}")
            return f"Success on attempt {attempt_id}"
        
        # Test retry mechanism
        for i in range(5):
            print(f"\n--- Test {i + 1} ---")
            result = await retry_mechanism.execute_with_retry(flaky_operation, i)
            
            if result.success:
                print(f"✅ Operation succeeded: {result.final_result}")
                print(f"   Attempts made: {len(result.attempts)}")
                print(f"   Total time: {result.total_time_ms:.1f}ms")
            else:
                print(f"❌ Operation failed: {result.final_error}")
                print(f"   Attempts made: {len(result.attempts)}")
                print(f"   Total time: {result.total_time_ms:.1f}ms")
        
        # Print metrics
        print(f"\n📊 Retry Metrics:")
        metrics = retry_mechanism.get_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
    
    # Run example
    asyncio.run(example_usage())