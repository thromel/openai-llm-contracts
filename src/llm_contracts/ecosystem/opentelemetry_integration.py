"""OpenTelemetry integration for LLM Design by Contract framework.

This module provides comprehensive observability through OpenTelemetry,
including tracing, metrics, and logging for contract execution.
"""

import time
import asyncio
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, ContextManager
from enum import Enum
import logging
import json

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Tracer, Span, Status, StatusCode
    from opentelemetry.metrics import Meter, Counter, Histogram, UpDownCounter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import ConsoleMetricsExporter, MetricsExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.semantic_conventions.trace import SpanAttributes
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Mock classes when OpenTelemetry is not available
    class Tracer:
        pass
    class Span:
        pass
    class Status:
        pass
    class StatusCode:
        pass
    class Meter:
        pass
    class Counter:
        pass
    class Histogram:
        pass
    class UpDownCounter:
        pass
    class TracerProvider:
        pass
    class MeterProvider:
        pass
    class BatchSpanProcessor:
        pass
    class MetricsExporter:
        pass
    class OTLPSpanExporter:
        pass
    class OTLPMetricExporter:
        pass
    class RequestsInstrumentor:
        pass
    class LoggingInstrumentor:
        pass
    class SpanAttributes:
        pass
    OPENTELEMETRY_AVAILABLE = False

# Import contract framework components
from ..contracts.base import ContractBase, ValidationResult
from ..core.exceptions import ContractViolationError, ValidationError

logger = logging.getLogger(__name__)


class TelemetryLevel(Enum):
    """Levels of telemetry detail."""
    MINIMAL = "minimal"      # Basic metrics only
    STANDARD = "standard"    # Metrics + basic tracing
    DETAILED = "detailed"    # Full tracing + detailed metrics
    DEBUG = "debug"          # All telemetry + debug info


@dataclass
class ContractSpanContext:
    """Context information for contract spans."""
    contract_name: str
    contract_type: str
    operation: str
    input_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_span_id: Optional[str] = None
    
    def to_attributes(self) -> Dict[str, Any]:
        """Convert to OpenTelemetry span attributes."""
        attributes = {
            "llm_contract.name": self.contract_name,
            "llm_contract.type": self.contract_type,
            "llm_contract.operation": self.operation,
        }
        
        if self.input_size is not None:
            attributes["llm_contract.input_size"] = self.input_size
        
        if self.parent_span_id:
            attributes["llm_contract.parent_span_id"] = self.parent_span_id
        
        # Add metadata with prefix
        for key, value in self.metadata.items():
            attributes[f"llm_contract.metadata.{key}"] = str(value)
        
        return attributes


class ContractTracer:
    """Tracer for contract execution with OpenTelemetry."""
    
    def __init__(self,
                 service_name: str = "llm_contracts",
                 telemetry_level: TelemetryLevel = TelemetryLevel.STANDARD):
        """Initialize contract tracer.
        
        Args:
            service_name: Name of the service for tracing
            telemetry_level: Level of telemetry detail
        """
        self.service_name = service_name
        self.telemetry_level = telemetry_level
        
        if OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(service_name)
        else:
            self.tracer = None
        
        self.active_spans = {}
        self.span_metrics = {
            "total_spans": 0,
            "active_spans": 0,
            "failed_spans": 0,
            "span_durations": []
        }
    
    def start_contract_span(self,
                          span_context: ContractSpanContext,
                          parent_span: Optional[Span] = None) -> Optional[Span]:
        """Start a new contract span.
        
        Args:
            span_context: Context information for the span
            parent_span: Parent span if this is a child span
            
        Returns:
            The created span or None if tracing is disabled
        """
        if not OPENTELEMETRY_AVAILABLE or not self.tracer:
            return None
        
        try:
            span_name = f"contract.{span_context.operation}"
            
            # Create span with parent context if provided
            if parent_span:
                with trace.use_span(parent_span):
                    span = self.tracer.start_span(span_name)
            else:
                span = self.tracer.start_span(span_name)
            
            # Set attributes
            attributes = span_context.to_attributes()
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            # Record span
            span_id = getattr(span, 'context', {}).get('span_id', str(id(span)))
            self.active_spans[span_id] = {
                "span": span,
                "context": span_context,
                "start_time": time.time()
            }
            
            self.span_metrics["total_spans"] += 1
            self.span_metrics["active_spans"] += 1
            
            return span
            
        except Exception as e:
            logger.error(f"Failed to start contract span: {e}")
            return None
    
    def end_contract_span(self,
                        span: Span,
                        result: Optional[ValidationResult] = None,
                        error: Optional[Exception] = None) -> None:
        """End a contract span with result information.
        
        Args:
            span: The span to end
            result: Validation result if available
            error: Exception if validation failed
        """
        if not span or not OPENTELEMETRY_AVAILABLE:
            return
        
        try:
            # Find span in active spans
            span_id = getattr(span, 'context', {}).get('span_id', str(id(span)))
            span_info = self.active_spans.get(span_id)
            
            if span_info:
                duration = time.time() - span_info["start_time"]
                self.span_metrics["span_durations"].append(duration)
                self.span_metrics["active_spans"] -= 1
                del self.active_spans[span_id]
            
            # Set result attributes
            if result:
                span.set_attribute("llm_contract.validation.is_valid", result.is_valid)
                if result.error_message:
                    span.set_attribute("llm_contract.validation.error_message", result.error_message)
                if result.auto_fixed_content is not None:
                    span.set_attribute("llm_contract.validation.auto_fixed", True)
            
            # Set error status
            if error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                self.span_metrics["failed_spans"] += 1
            elif result and not result.is_valid:
                span.set_status(Status(StatusCode.ERROR, "Validation failed"))
                self.span_metrics["failed_spans"] += 1
            else:
                span.set_status(Status(StatusCode.OK))
            
            span.end()
            
        except Exception as e:
            logger.error(f"Failed to end contract span: {e}")
    
    def trace_contract_execution(self,
                               contract: ContractBase,
                               operation: str = "validate") -> Callable:
        """Decorator to trace contract execution.
        
        Args:
            contract: The contract being executed
            operation: The operation being performed
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_context = ContractSpanContext(
                    contract_name=contract.contract_name,
                    contract_type=contract.contract_type,
                    operation=operation,
                    input_size=len(str(args[0])) if args else None
                )
                
                span = self.start_contract_span(span_context)
                error = None
                result = None
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    self.end_contract_span(span, result, error)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_context = ContractSpanContext(
                    contract_name=contract.contract_name,
                    contract_type=contract.contract_type,
                    operation=operation,
                    input_size=len(str(args[0])) if args else None
                )
                
                span = self.start_contract_span(span_context)
                error = None
                result = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    self.end_contract_span(span, result, error)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tracer metrics."""
        metrics = self.span_metrics.copy()
        
        if metrics["span_durations"]:
            metrics["avg_span_duration"] = sum(metrics["span_durations"]) / len(metrics["span_durations"])
            metrics["max_span_duration"] = max(metrics["span_durations"])
            metrics["min_span_duration"] = min(metrics["span_durations"])
        
        metrics["failure_rate"] = metrics["failed_spans"] / max(metrics["total_spans"], 1)
        
        return metrics


class MetricsCollector:
    """Collector for contract execution metrics."""
    
    def __init__(self,
                 service_name: str = "llm_contracts",
                 meter_name: str = "llm_contract_metrics"):
        """Initialize metrics collector.
        
        Args:
            service_name: Name of the service
            meter_name: Name of the meter
        """
        self.service_name = service_name
        self.meter_name = meter_name
        
        if OPENTELEMETRY_AVAILABLE:
            self.meter = metrics.get_meter(meter_name)
            
            # Create metrics instruments
            self.validation_counter = self.meter.create_counter(
                name="llm_contract_validations_total",
                description="Total number of contract validations",
                unit="1"
            )
            
            self.validation_duration = self.meter.create_histogram(
                name="llm_contract_validation_duration_seconds",
                description="Duration of contract validations",
                unit="s"
            )
            
            self.violation_counter = self.meter.create_counter(
                name="llm_contract_violations_total",
                description="Total number of contract violations",
                unit="1"
            )
            
            self.auto_fix_counter = self.meter.create_counter(
                name="llm_contract_auto_fixes_total",
                description="Total number of successful auto-fixes",
                unit="1"
            )
            
            self.active_contracts_gauge = self.meter.create_up_down_counter(
                name="llm_contract_active_contracts",
                description="Number of currently active contracts",
                unit="1"
            )
            
        else:
            self.meter = None
            self.validation_counter = None
            self.validation_duration = None
            self.violation_counter = None
            self.auto_fix_counter = None
            self.active_contracts_gauge = None
        
        # Local metrics for when OpenTelemetry is not available
        self.local_metrics = {
            "validations_total": 0,
            "violations_total": 0,
            "auto_fixes_total": 0,
            "active_contracts": 0,
            "validation_durations": []
        }
    
    def record_validation(self,
                         contract_name: str,
                         contract_type: str,
                         duration: float,
                         is_valid: bool,
                         auto_fixed: bool = False) -> None:
        """Record a contract validation event.
        
        Args:
            contract_name: Name of the contract
            contract_type: Type of the contract
            duration: Duration of the validation in seconds
            is_valid: Whether validation passed
            auto_fixed: Whether auto-fix was applied
        """
        attributes = {
            "contract_name": contract_name,
            "contract_type": contract_type,
            "result": "valid" if is_valid else "invalid"
        }
        
        # Record with OpenTelemetry if available
        if self.validation_counter:
            self.validation_counter.add(1, attributes)
        
        if self.validation_duration:
            self.validation_duration.record(duration, attributes)
        
        if not is_valid and self.violation_counter:
            self.violation_counter.add(1, attributes)
        
        if auto_fixed and self.auto_fix_counter:
            self.auto_fix_counter.add(1, attributes)
        
        # Record in local metrics
        self.local_metrics["validations_total"] += 1
        self.local_metrics["validation_durations"].append(duration)
        
        if not is_valid:
            self.local_metrics["violations_total"] += 1
        
        if auto_fixed:
            self.local_metrics["auto_fixes_total"] += 1
    
    def increment_active_contracts(self, count: int = 1) -> None:
        """Increment the number of active contracts."""
        if self.active_contracts_gauge:
            self.active_contracts_gauge.add(count)
        
        self.local_metrics["active_contracts"] += count
    
    def decrement_active_contracts(self, count: int = 1) -> None:
        """Decrement the number of active contracts."""
        if self.active_contracts_gauge:
            self.active_contracts_gauge.add(-count)
        
        self.local_metrics["active_contracts"] = max(0, self.local_metrics["active_contracts"] - count)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = self.local_metrics.copy()
        
        if summary["validation_durations"]:
            durations = summary["validation_durations"]
            summary["avg_validation_duration"] = sum(durations) / len(durations)
            summary["max_validation_duration"] = max(durations)
            summary["min_validation_duration"] = min(durations)
        
        summary["violation_rate"] = summary["violations_total"] / max(summary["validations_total"], 1)
        summary["auto_fix_rate"] = summary["auto_fixes_total"] / max(summary["violations_total"], 1)
        
        return summary


class OpenTelemetryIntegration:
    """Main OpenTelemetry integration for the contract framework."""
    
    def __init__(self,
                 service_name: str = "llm_contracts",
                 telemetry_level: TelemetryLevel = TelemetryLevel.STANDARD,
                 otlp_endpoint: Optional[str] = None,
                 console_export: bool = True):
        """Initialize OpenTelemetry integration.
        
        Args:
            service_name: Name of the service
            telemetry_level: Level of telemetry detail
            otlp_endpoint: OTLP endpoint for exporting data
            console_export: Whether to export to console
        """
        self.service_name = service_name
        self.telemetry_level = telemetry_level
        self.otlp_endpoint = otlp_endpoint
        self.console_export = console_export
        
        self.tracer = ContractTracer(service_name, telemetry_level)
        self.metrics_collector = MetricsCollector(service_name)
        
        self.initialized = False
        self.instrumentation_enabled = False
    
    def initialize(self) -> bool:
        """Initialize OpenTelemetry providers and exporters.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, using local metrics only")
            self.initialized = True
            return False
        
        try:
            # Initialize trace provider
            trace_provider = TracerProvider()
            trace.set_tracer_provider(trace_provider)
            
            # Initialize meter provider
            meter_provider = MeterProvider()
            metrics.set_meter_provider(meter_provider)
            
            # Configure exporters
            if self.otlp_endpoint:
                # OTLP exporters
                otlp_trace_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                trace_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))
                
                # Note: Metric export configuration would depend on the specific setup
            
            if self.console_export:
                # Console exporters for development
                console_exporter = ConsoleMetricsExporter()
                # Note: Console span exporter setup would be here
            
            self.initialized = True
            logger.info("OpenTelemetry integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.initialized = True  # Use local metrics
            return False
    
    def enable_auto_instrumentation(self) -> None:
        """Enable automatic instrumentation for common libraries."""
        if not OPENTELEMETRY_AVAILABLE or self.instrumentation_enabled:
            return
        
        try:
            # Enable automatic instrumentation
            RequestsInstrumentor().instrument()
            LoggingInstrumentor().instrument()
            
            self.instrumentation_enabled = True
            logger.info("Auto-instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable auto-instrumentation: {e}")
    
    def create_contract_span(self,
                           contract: ContractBase,
                           operation: str = "validate") -> ContextManager:
        """Create a context manager for contract span.
        
        Args:
            contract: The contract being executed
            operation: The operation being performed
            
        Returns:
            Context manager for the span
        """
        return ContractSpan(self.tracer, contract, operation, self.metrics_collector)
    
    def instrument_contract(self, contract: ContractBase) -> ContractBase:
        """Instrument a contract with telemetry.
        
        Args:
            contract: The contract to instrument
            
        Returns:
            The instrumented contract
        """
        # Wrap the validate_async method
        original_validate = contract.validate_async
        
        @self.tracer.trace_contract_execution(contract, "validate")
        async def instrumented_validate(data: Any) -> ValidationResult:
            start_time = time.time()
            
            try:
                result = await original_validate(data)
                
                # Record metrics
                duration = time.time() - start_time
                self.metrics_collector.record_validation(
                    contract_name=contract.contract_name,
                    contract_type=contract.contract_type,
                    duration=duration,
                    is_valid=result.is_valid,
                    auto_fixed=result.auto_fixed_content is not None
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                self.metrics_collector.record_validation(
                    contract_name=contract.contract_name,
                    contract_type=contract.contract_type,
                    duration=duration,
                    is_valid=False
                )
                raise
        
        contract.validate_async = instrumented_validate
        return contract
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get comprehensive telemetry summary."""
        return {
            "service_name": self.service_name,
            "telemetry_level": self.telemetry_level.value,
            "initialized": self.initialized,
            "instrumentation_enabled": self.instrumentation_enabled,
            "tracer_metrics": self.tracer.get_metrics(),
            "collector_metrics": self.metrics_collector.get_metrics_summary()
        }


class ContractSpan:
    """Context manager for contract spans."""
    
    def __init__(self,
                 tracer: ContractTracer,
                 contract: ContractBase,
                 operation: str,
                 metrics_collector: MetricsCollector):
        """Initialize contract span context manager.
        
        Args:
            tracer: The contract tracer
            contract: The contract being executed
            operation: The operation being performed
            metrics_collector: The metrics collector
        """
        self.tracer = tracer
        self.contract = contract
        self.operation = operation
        self.metrics_collector = metrics_collector
        self.span = None
        self.start_time = None
        self.result = None
        self.error = None
    
    def __enter__(self):
        """Enter the span context."""
        self.start_time = time.time()
        
        span_context = ContractSpanContext(
            contract_name=self.contract.contract_name,
            contract_type=self.contract.contract_type,
            operation=self.operation
        )
        
        self.span = self.tracer.start_contract_span(span_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the span context."""
        self.error = exc_val
        
        # End span
        self.tracer.end_contract_span(self.span, self.result, self.error)
        
        # Record metrics
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.record_validation(
                contract_name=self.contract.contract_name,
                contract_type=self.contract.contract_type,
                duration=duration,
                is_valid=self.result.is_valid if self.result else False,
                auto_fixed=self.result.auto_fixed_content is not None if self.result else False
    )
    
    def set_result(self, result: ValidationResult) -> None:
        """Set the validation result."""
        self.result = result


# Convenience functions

def setup_telemetry(service_name: str = "llm_contracts",
                   telemetry_level: TelemetryLevel = TelemetryLevel.STANDARD,
                   otlp_endpoint: Optional[str] = None,
                   enable_auto_instrumentation: bool = True) -> OpenTelemetryIntegration:
    """Set up OpenTelemetry integration for contract framework.
    
    Args:
        service_name: Name of the service
        telemetry_level: Level of telemetry detail
        otlp_endpoint: OTLP endpoint for exporting data
        enable_auto_instrumentation: Whether to enable auto-instrumentation
        
    Returns:
        Configured OpenTelemetry integration instance
    """
    integration = OpenTelemetryIntegration(
        service_name=service_name,
        telemetry_level=telemetry_level,
        otlp_endpoint=otlp_endpoint
    )
    
    integration.initialize()
    
    if enable_auto_instrumentation:
        integration.enable_auto_instrumentation()
    
    return integration


def trace_contract_execution(contract: ContractBase,
                           operation: str = "validate") -> Callable:
    """Decorator to add tracing to contract execution.
    
    Args:
        contract: The contract being executed
        operation: The operation being performed
        
    Returns:
        Decorator function
    """
    # Create a minimal tracer if none exists
    tracer = ContractTracer()
    return tracer.trace_contract_execution(contract, operation)


# Example usage
def example_opentelemetry_integration():
    """Example of OpenTelemetry integration usage."""
    if not OPENTELEMETRY_AVAILABLE:
        print("OpenTelemetry not available for example")
        return
    
    # Set up telemetry
    integration = setup_telemetry(
        service_name="example_llm_service",
        telemetry_level=TelemetryLevel.DETAILED,
        otlp_endpoint="http://localhost:4317"
    )
    
    print(f"Telemetry initialized: {integration.initialized}")
    print(f"Summary: {integration.get_telemetry_summary()}")