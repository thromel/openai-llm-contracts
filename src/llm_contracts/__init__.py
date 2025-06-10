"""
LLM Design by Contract Framework

A comprehensive Design by Contract framework for Large Language Model APIs
that provides input/output validation, temporal contracts, streaming support,
and multi-platform compatibility.
"""

__version__ = "0.1.0"
__author__ = "Romel"

# Core interfaces and exceptions
from .core.interfaces import ValidationResult
from .core.exceptions import ContractViolationError, ProviderError

# Main provider
from .providers.openai_provider import ImprovedOpenAIProvider

# Common contracts
from .contracts.base import (
    ContractBase,
    PromptLengthContract,
    JSONFormatContract,
    ContentPolicyContract,
    PromptInjectionContract,
    ResponseTimeContract,
    ConversationConsistencyContract,
    MedicalDisclaimerContract,
)

# Language support
from .language.integration import llmcl_contract, llmcl_to_contract

# Import submodules for easier access
from . import providers
from . import validators
from . import contracts
# Temporarily comment out to fix import issues
# from . import experiments
# from . import conversation
# from . import streaming
# from . import ecosystem
# from . import plugins

__all__ = [
    # Core
    "ValidationResult",
    "ContractViolationError",
    "ProviderError",
    # Provider
    "ImprovedOpenAIProvider",
    # Contracts
    "ContractBase",
    "PromptLengthContract",
    "JSONFormatContract",
    "ContentPolicyContract",
    "PromptInjectionContract",
    "ResponseTimeContract",
    "ConversationConsistencyContract",
    "MedicalDisclaimerContract",
    # Language
    "llmcl_contract",
    "llmcl_to_contract",
    # Submodules
    "providers",
    "validators",
    "contracts",
    # Temporarily commented out
    # "experiments",
    # "conversation",
    # "streaming",
    # "ecosystem",
    # "plugins",
]
