"""Ecosystem integration components for LLM Design by Contract framework.

This module provides comprehensive integration with popular LLM ecosystem tools
including LangChain, Guardrails.ai migration support, and Pydantic model validation.
"""

from .langchain_integration import (
    ContractLLM,
    LangChainContractValidator,
    ChainContractOrchestrator,
    ContractOutputParser,
    ContractAgent,
    ContractTool,
    ContractCallbackHandler,
    create_contract_chain,
    create_contract_agent
)

from .guardrails_adapter import (
    GuardrailsAdapter,
    GuardrailsMigrator,
    GuardrailsValidator,
    convert_guardrails_to_contract,
    migrate_guardrails_config
)

from .pydantic_integration import (
    PydanticContract,
    PydanticValidator,
    ModelBasedContract,
    create_pydantic_contract,
    pydantic_to_contract_schema
)

from .opentelemetry_integration import (
    OpenTelemetryIntegration,
    ContractTracer,
    ContractSpan,
    MetricsCollector,
    setup_telemetry,
    trace_contract_execution
)

__all__ = [
    # LangChain Integration
    "ContractLLM",
    "LangChainContractValidator", 
    "ChainContractOrchestrator",
    "ContractOutputParser",
    "ContractAgent",
    "ContractTool",
    "ContractCallbackHandler",
    "create_contract_chain",
    "create_contract_agent",
    
    # Guardrails.ai Migration
    "GuardrailsAdapter",
    "GuardrailsMigrator", 
    "GuardrailsValidator",
    "convert_guardrails_to_contract",
    "migrate_guardrails_config",
    
    # Pydantic Integration
    "PydanticContract",
    "PydanticValidator",
    "ModelBasedContract", 
    "create_pydantic_contract",
    "pydantic_to_contract_schema",
    
    # OpenTelemetry Integration
    "OpenTelemetryIntegration",
    "ContractTracer",
    "ContractSpan",
    "MetricsCollector",
    "setup_telemetry",
    "trace_contract_execution",
]