"""
Reliability Module for LLM Contracts

This module provides robust reliability mechanisms for LLM API interactions,
including circuit breakers and retry mechanisms with comprehensive monitoring.
"""

from .circuit_breaker import (
    RobustCircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FailureType,
    FailureRecord,
    CircuitBreakerMetrics,
    HealthChecker,
    LLMHealthChecker,
    circuit_breaker
)

from .retry_mechanism import (
    RobustRetryMechanism,
    RetryConfig,
    RetryStrategy,
    RetryResult,
    RetryAttempt,
    ErrorClassification,
    BackoffCalculator,
    ExponentialBackoffCalculator,
    LinearBackoffCalculator,
    FixedDelayCalculator,
    ErrorClassifier,
    RetryMetrics,
    retry
)

__all__ = [
    # Circuit Breaker
    'RobustCircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    'FailureType',
    'FailureRecord',
    'CircuitBreakerMetrics',
    'HealthChecker',
    'LLMHealthChecker',
    'circuit_breaker',
    
    # Retry Mechanism
    'RobustRetryMechanism',
    'RetryConfig',
    'RetryStrategy',
    'RetryResult',
    'RetryAttempt',
    'ErrorClassification',
    'BackoffCalculator',
    'ExponentialBackoffCalculator',
    'LinearBackoffCalculator',
    'FixedDelayCalculator',
    'ErrorClassifier',
    'RetryMetrics',
    'retry'
]