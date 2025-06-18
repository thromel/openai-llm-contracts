"""
Robust Circuit Breaker Implementation for LLM API Contracts

This module provides a comprehensive circuit breaker pattern implementation
designed specifically for LLM API reliability, with features like:
- Multiple failure threshold types
- Adaptive timeout strategies
- Health check mechanisms
- Metrics collection and monitoring
- Recovery strategies
"""

import time
import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from enum import Enum
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states with clear semantics."""
    CLOSED = "closed"          # Normal operation - requests allowed
    OPEN = "open"              # Circuit breaker tripped - requests blocked
    HALF_OPEN = "half_open"    # Testing recovery - limited requests allowed


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """Record of a failure with detailed context."""
    timestamp: float
    failure_type: FailureType
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    contract_name: Optional[str] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Failure thresholds
    failure_threshold: int = 5              # Consecutive failures to open circuit
    failure_rate_threshold: float = 0.5     # Failure rate (0.0-1.0) over window
    failure_rate_window: int = 60           # Window in seconds for failure rate calculation
    
    # Timeout and recovery
    timeout_seconds: int = 60               # How long to stay open
    half_open_max_calls: int = 3            # Max calls to allow in half-open state
    half_open_success_threshold: int = 2    # Successes needed to close circuit
    
    # Advanced features
    adaptive_timeout: bool = True           # Enable adaptive timeout based on failure patterns
    health_check_enabled: bool = True       # Enable periodic health checks
    health_check_interval: int = 30         # Health check interval in seconds
    
    # Metrics and monitoring
    metrics_window_size: int = 1000         # Max failure records to keep
    enable_detailed_metrics: bool = True    # Track detailed performance metrics


class CircuitBreakerMetrics:
    """Comprehensive metrics collection for circuit breaker."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state_transitions: List[Tuple[float, CircuitState, CircuitState]] = []
        self.failure_records: deque = deque(maxlen=config.metrics_window_size)
        self.success_count = 0
        self.total_requests = 0
        self.state_durations: Dict[CircuitState, List[float]] = defaultdict(list)
        self.current_state_start = time.time()
        self._lock = threading.Lock()
    
    def record_request(self, success: bool, failure_type: Optional[FailureType] = None, 
                      error_message: str = "", contract_name: Optional[str] = None):
        """Record a request outcome."""
        with self._lock:
            self.total_requests += 1
            if success:
                self.success_count += 1
            else:
                failure_record = FailureRecord(
                    timestamp=time.time(),
                    failure_type=failure_type or FailureType.UNKNOWN,
                    error_message=error_message,
                    contract_name=contract_name
                )
                self.failure_records.append(failure_record)
    
    def record_state_transition(self, from_state: CircuitState, to_state: CircuitState):
        """Record a state transition."""
        with self._lock:
            now = time.time()
            duration = now - self.current_state_start
            self.state_durations[from_state].append(duration)
            self.state_transitions.append((now, from_state, to_state))
            self.current_state_start = now
    
    def get_failure_rate(self, window_seconds: int = None) -> float:
        """Calculate failure rate over specified window."""
        if window_seconds is None:
            window_seconds = self.config.failure_rate_window
        
        cutoff_time = time.time() - window_seconds
        recent_failures = sum(1 for record in self.failure_records 
                            if record.timestamp > cutoff_time)
        
        # Estimate total requests in window (simplified)
        window_ratio = window_seconds / max(1, time.time() - (self.failure_records[0].timestamp if self.failure_records else time.time()))
        estimated_requests = max(1, int(self.total_requests * window_ratio))
        
        return recent_failures / estimated_requests
    
    def get_consecutive_failures(self) -> int:
        """Get count of consecutive recent failures."""
        count = 0
        for record in reversed(self.failure_records):
            if record.timestamp > time.time() - 300:  # Last 5 minutes
                count += 1
            else:
                break
        return count
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "success_count": self.success_count,
                "success_rate": self.success_count / max(1, self.total_requests),
                "failure_rate_1min": self.get_failure_rate(60),
                "failure_rate_5min": self.get_failure_rate(300),
                "consecutive_failures": self.get_consecutive_failures(),
                "state_transitions": len(self.state_transitions),
                "avg_state_durations": {
                    state.value: statistics.mean(durations) if durations else 0
                    for state, durations in self.state_durations.items()
                },
                "failure_types": self._get_failure_type_breakdown(),
                "recent_failures": list(self.failure_records)[-10:]  # Last 10 failures
            }
    
    def _get_failure_type_breakdown(self) -> Dict[str, int]:
        """Get breakdown of failure types."""
        breakdown = defaultdict(int)
        for record in self.failure_records:
            breakdown[record.failure_type.value] += 1
        return dict(breakdown)


class HealthChecker(ABC):
    """Abstract base class for health check implementations."""
    
    @abstractmethod
    async def check_health(self) -> bool:
        """Perform health check. Return True if healthy, False otherwise."""
        pass


class LLMHealthChecker(HealthChecker):
    """Health checker specifically designed for LLM APIs."""
    
    def __init__(self, client_factory: Callable, test_prompt: str = "Health check"):
        self.client_factory = client_factory
        self.test_prompt = test_prompt
    
    async def check_health(self) -> bool:
        """Check LLM API health with a simple test request."""
        try:
            client = self.client_factory()
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": self.test_prompt}],
                max_tokens=10,
                timeout=5
            )
            return bool(response and response.choices)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False


class RobustCircuitBreaker:
    """
    Enterprise-grade circuit breaker implementation for LLM APIs.
    
    Features:
    - Multiple failure detection strategies
    - Adaptive timeout based on failure patterns
    - Health check integration
    - Comprehensive metrics and monitoring
    - Thread-safe operation
    - Async/sync support
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None,
                 health_checker: Optional[HealthChecker] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.health_checker = health_checker
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[float] = None
        self.last_state_change = time.time()
        self.half_open_attempts = 0
        self.half_open_successes = 0
        
        # Metrics and monitoring
        self.metrics = CircuitBreakerMetrics(self.config)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Callbacks
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    def should_allow_request(self) -> bool:
        """Determine if a request should be allowed through the circuit breaker."""
        with self._lock:
            now = time.time()
            
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if timeout has expired
                if self.last_failure_time and (now - self.last_failure_time) >= self._get_current_timeout():
                    self._transition_to_half_open()
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self.half_open_attempts < self.config.half_open_max_calls:
                    self.half_open_attempts += 1
                    return True
                return False
            
            return False
    
    def record_success(self, execution_time: Optional[float] = None):
        """Record a successful operation."""
        with self._lock:
            self.metrics.record_request(success=True)
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                
                # Check if we should close the circuit
                if self.half_open_successes >= self.config.half_open_success_threshold:
                    self._transition_to_closed()
            
            logger.debug(f"Circuit breaker '{self.name}' recorded success")
    
    def record_failure(self, 
                      failure_type: FailureType = FailureType.UNKNOWN,
                      error_message: str = "",
                      contract_name: Optional[str] = None):
        """Record a failed operation."""
        with self._lock:
            now = time.time()
            self.last_failure_time = now
            
            self.metrics.record_request(
                success=False,
                failure_type=failure_type,
                error_message=error_message,
                contract_name=contract_name
            )
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                if self._should_open_circuit():
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens the circuit
                self._transition_to_open()
            
            logger.warning(f"Circuit breaker '{self.name}' recorded failure: {failure_type.value} - {error_message}")
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on failure patterns."""
        # Check consecutive failures
        consecutive_failures = self.metrics.get_consecutive_failures()
        if consecutive_failures >= self.config.failure_threshold:
            logger.info(f"Opening circuit due to {consecutive_failures} consecutive failures")
            return True
        
        # Check failure rate
        failure_rate = self.metrics.get_failure_rate()
        if failure_rate >= self.config.failure_rate_threshold:
            logger.info(f"Opening circuit due to failure rate {failure_rate:.2%}")
            return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()
        self._reset_half_open_counters()
        
        self.metrics.record_state_transition(old_state, self.state)
        
        if self.on_state_change:
            self.on_state_change(old_state, self.state)
        
        logger.warning(f"Circuit breaker '{self.name}' opened")
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self._reset_half_open_counters()
        
        self.metrics.record_state_transition(old_state, self.state)
        
        if self.on_state_change:
            self.on_state_change(old_state, self.state)
        
        logger.info(f"Circuit breaker '{self.name}' half-opened for testing")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self._reset_half_open_counters()
        
        self.metrics.record_state_transition(old_state, self.state)
        
        if self.on_state_change:
            self.on_state_change(old_state, self.state)
        
        logger.info(f"Circuit breaker '{self.name}' closed - normal operation resumed")
    
    def _reset_half_open_counters(self):
        """Reset half-open state counters."""
        self.half_open_attempts = 0
        self.half_open_successes = 0
    
    def _get_current_timeout(self) -> int:
        """Get current timeout, potentially adaptive based on failure patterns."""
        base_timeout = self.config.timeout_seconds
        
        if not self.config.adaptive_timeout:
            return base_timeout
        
        # Adaptive timeout based on recent failure patterns
        recent_failures = self.metrics.get_consecutive_failures()
        if recent_failures > 10:
            # Exponential backoff for severe failures
            multiplier = min(2 ** (recent_failures // 5), 8)  # Cap at 8x
            return base_timeout * multiplier
        
        return base_timeout
    
    async def start_health_checks(self):
        """Start periodic health checks if health checker is configured."""
        if not self.health_checker or not self.config.health_check_enabled:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Started health checks for circuit breaker '{self.name}'")
    
    async def stop_health_checks(self):
        """Stop health check task."""
        self._shutdown = True
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped health checks for circuit breaker '{self.name}'")
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self.state == CircuitState.OPEN:
                    # Only perform health checks when circuit is open
                    is_healthy = await self.health_checker.check_health()
                    
                    if is_healthy:
                        logger.info(f"Health check passed for '{self.name}', transitioning to half-open")
                        with self._lock:
                            self._transition_to_half_open()
                    else:
                        logger.debug(f"Health check failed for '{self.name}', remaining open")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for '{self.name}': {e}")
    
    def force_open(self):
        """Manually force circuit breaker to OPEN state."""
        with self._lock:
            if self.state != CircuitState.OPEN:
                self._transition_to_open()
        logger.warning(f"Circuit breaker '{self.name}' manually forced open")
    
    def force_close(self):
        """Manually force circuit breaker to CLOSED state."""
        with self._lock:
            if self.state != CircuitState.CLOSED:
                self._transition_to_closed()
        logger.info(f"Circuit breaker '{self.name}' manually forced closed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "last_failure_time": self.last_failure_time,
                "last_state_change": self.last_state_change,
                "time_in_current_state": time.time() - self.last_state_change,
                "half_open_attempts": self.half_open_attempts,
                "half_open_successes": self.half_open_successes,
                "current_timeout": self._get_current_timeout(),
                "metrics": self.metrics.get_health_report()
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record outcome."""
        if exc_type is None:
            self.record_success()
        else:
            # Classify the exception
            failure_type = self._classify_exception(exc_type, exc_val)
            self.record_failure(
                failure_type=failure_type,
                error_message=str(exc_val) if exc_val else str(exc_type)
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - record outcome."""
        if exc_type is None:
            self.record_success()
        else:
            failure_type = self._classify_exception(exc_type, exc_val)
            self.record_failure(
                failure_type=failure_type,
                error_message=str(exc_val) if exc_val else str(exc_type)
            )
    
    def _classify_exception(self, exc_type, exc_val) -> FailureType:
        """Classify exception type for metrics."""
        if exc_type.__name__ in ['TimeoutError', 'asyncio.TimeoutError']:
            return FailureType.TIMEOUT
        elif 'RateLimitError' in exc_type.__name__:
            return FailureType.RATE_LIMIT
        elif 'ValidationError' in exc_type.__name__ or 'ContractViolationError' in exc_type.__name__:
            return FailureType.VALIDATION_ERROR
        elif 'APIError' in exc_type.__name__:
            return FailureType.API_ERROR
        elif 'NetworkError' in exc_type.__name__ or 'ConnectionError' in exc_type.__name__:
            return FailureType.NETWORK_ERROR
        else:
            return FailureType.UNKNOWN


# Decorator for automatic circuit breaker integration
def circuit_breaker(name: str, 
                   config: Optional[CircuitBreakerConfig] = None,
                   health_checker: Optional[HealthChecker] = None):
    """Decorator to add circuit breaker protection to functions."""
    
    def decorator(func):
        breaker = RobustCircuitBreaker(name, config, health_checker)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                if not breaker.should_allow_request():
                    raise Exception(f"Circuit breaker '{name}' is open")
                
                async with breaker:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                if not breaker.should_allow_request():
                    raise Exception(f"Circuit breaker '{name}' is open")
                
                with breaker:
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


# Usage examples and tests
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example usage of the robust circuit breaker."""
        
        # Configure circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_rate_threshold=0.5,
            timeout_seconds=30,
            adaptive_timeout=True,
            health_check_enabled=True
        )
        
        # Create circuit breaker
        breaker = RobustCircuitBreaker("example", config)
        
        # Example of normal usage
        for i in range(10):
            if breaker.should_allow_request():
                try:
                    # Simulate some operation
                    if i % 3 == 0:  # Simulate occasional failures
                        raise Exception("Simulated failure")
                    
                    # Simulate success
                    await asyncio.sleep(0.1)
                    breaker.record_success()
                    print(f"Request {i}: Success")
                
                except Exception as e:
                    breaker.record_failure(
                        failure_type=FailureType.API_ERROR,
                        error_message=str(e)
                    )
                    print(f"Request {i}: Failed - {e}")
            else:
                print(f"Request {i}: Blocked by circuit breaker")
        
        # Print status
        status = breaker.get_status()
        print(f"\nCircuit breaker status: {status}")
    
    # Run example
    asyncio.run(example_usage())