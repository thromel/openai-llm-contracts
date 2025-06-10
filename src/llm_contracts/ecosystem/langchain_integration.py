"""LangChain integration for LLM Design by Contract framework.

This module provides comprehensive integration with LangChain, including
ContractLLM wrapper, chain-level contract orchestration, and integration
with LangChain's output parsers and agent frameworks.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
from enum import Enum
import logging

try:
    from langchain.llms.base import LLM
    from langchain.chat_models.base import BaseChatModel
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.schema.output import LLMResult, ChatResult
    from langchain.schema.output_parser import BaseOutputParser
    from langchain.agents import Agent, AgentExecutor
    from langchain.tools import BaseTool
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.chains.base import Chain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Mock classes when LangChain is not available
    class LLM:
        pass
    class BaseChatModel:
        pass
    class BaseMessage:
        pass
    class HumanMessage(BaseMessage):
        pass
    class AIMessage(BaseMessage):
        pass
    class SystemMessage(BaseMessage):
        pass
    class LLMResult:
        pass
    class ChatResult:
        pass
    class BaseOutputParser:
        pass
    class Agent:
        pass
    class AgentExecutor:
        pass
    class BaseTool:
        pass
    class BaseCallbackHandler:
        pass
    class Chain:
        pass
    LANGCHAIN_AVAILABLE = False

# Import contract framework components
from ..contracts.base import ContractBase, ValidationResult
# from ..validators.input_validator import PerformanceOptimizedInputValidator  # Avoid potential circular imports
# from ..validators.output_validator import PerformanceOptimizedOutputValidator  # Circular import
from ..core.exceptions import ContractViolationError
# from ..utils.telemetry import log_contract_execution  # Function not available

logger = logging.getLogger(__name__)


class ContractExecutionMode(Enum):
    """Modes for contract execution in LangChain integration."""
    STRICT = "strict"           # Fail on any contract violation
    PERMISSIVE = "permissive"   # Log violations but continue
    AUTO_RETRY = "auto_retry"   # Retry with auto-remediation
    CALLBACK = "callback"       # Use callback for violation handling


@dataclass
class ContractExecutionContext:
    """Context for contract execution in LangChain chains."""
    chain_id: str
    step_name: str
    input_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional['ContractExecutionContext'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chain_id": self.chain_id,
            "step_name": self.step_name,
            "input_data": self.input_data,
            "metadata": self.metadata,
            "parent_context": self.parent_context.to_dict() if self.parent_context else None
        }


class ContractLLM(LLM if LANGCHAIN_AVAILABLE else object):
    """LangChain LLM wrapper with contract enforcement."""
    
    def __init__(self,
                 base_llm: Union[LLM, BaseChatModel],
                 input_contracts: Optional[List[ContractBase]] = None,
                 output_contracts: Optional[List[ContractBase]] = None,
                 execution_mode: ContractExecutionMode = ContractExecutionMode.STRICT,
                 violation_handler: Optional[Callable] = None,
                 context_manager: Optional[Any] = None):
        """Initialize ContractLLM wrapper.
        
        Args:
            base_llm: The underlying LangChain LLM to wrap
            input_contracts: List of input validation contracts
            output_contracts: List of output validation contracts
            execution_mode: How to handle contract violations
            violation_handler: Custom violation handler function
            context_manager: Optional conversation context manager
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LangChain integration")
            
        super().__init__()
        self.base_llm = base_llm
        self.input_contracts = input_contracts or []
        self.output_contracts = output_contracts or []
        self.execution_mode = execution_mode
        self.violation_handler = violation_handler
        self.context_manager = context_manager
        
        # Initialize validators
        self.input_validator = PerformanceOptimizedInputValidator(self.input_contracts)
        self.output_validator = PerformanceOptimizedOutputValidator(self.output_contracts)
        
        # Execution metrics
        self.execution_metrics = {
            "total_calls": 0,
            "contract_violations": 0,
            "auto_remediation_success": 0,
            "retries": 0
        }
        
    @property
    def _llm_type(self) -> str:
        """Return the LLM type."""
        return f"contract_{self.base_llm._llm_type}"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call the LLM with contract enforcement."""
        return asyncio.run(self._acall(prompt, stop, **kwargs))
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Async call the LLM with contract enforcement."""
        self.execution_metrics["total_calls"] += 1
        
        # Create execution context
        context = ContractExecutionContext(
            chain_id=kwargs.get("chain_id", "unknown"),
            step_name=kwargs.get("step_name", "llm_call"),
            input_data={"prompt": prompt, "stop": stop, **kwargs}
        )
        
        try:
            # Input validation
            await self._validate_input(prompt, context)
            
            # Call base LLM
            if hasattr(self.base_llm, '_acall'):
                response = await self.base_llm._acall(prompt, stop, **kwargs)
            else:
                response = self.base_llm._call(prompt, stop, **kwargs)
            
            # Output validation
            validated_response = await self._validate_output(response, context)
            
            return validated_response
            
        except ContractViolationError as e:
            return await self._handle_violation(e, context, prompt, stop, **kwargs)
    
    async def _validate_input(self, prompt: str, context: ContractExecutionContext) -> None:
        """Validate input against input contracts."""
        if not self.input_contracts:
            return
            
        input_data = {"prompt": prompt, "context": context.to_dict()}
        
        try:
            result = await self.input_validator.validate_async(input_data)
            if not result.is_valid:
                raise ContractViolationError(f"Input validation failed: {result.error_message}")
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            if self.execution_mode == ContractExecutionMode.STRICT:
                raise
    
    async def _validate_output(self, response: str, context: ContractExecutionContext) -> str:
        """Validate output against output contracts."""
        if not self.output_contracts:
            return response
            
        output_data = {"response": response, "context": context.to_dict()}
        
        try:
            result = await self.output_validator.validate_async(output_data)
            if not result.is_valid:
                if result.auto_fixed_content:
                    logger.info("Output auto-remediated successfully")
                    self.execution_metrics["auto_remediation_success"] += 1
                    return result.auto_fixed_content
                else:
                    raise ContractViolationError(f"Output validation failed: {result.error_message}")
            return response
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            if self.execution_mode == ContractExecutionMode.STRICT:
                raise
            return response
    
    async def _handle_violation(self, 
                              violation: ContractViolationError,
                              context: ContractExecutionContext,
                              prompt: str,
                              stop: Optional[List[str]] = None,
                              **kwargs) -> str:
        """Handle contract violations based on execution mode."""
        self.execution_metrics["contract_violations"] += 1
        
        if self.violation_handler:
            try:
                return await self.violation_handler(violation, context, prompt, stop, **kwargs)
            except Exception as e:
                logger.error(f"Custom violation handler failed: {e}")
        
        if self.execution_mode == ContractExecutionMode.STRICT:
            raise violation
        elif self.execution_mode == ContractExecutionMode.PERMISSIVE:
            logger.warning(f"Contract violation (permissive mode): {violation}")
            # Return original call without contracts
            if hasattr(self.base_llm, '_acall'):
                return await self.base_llm._acall(prompt, stop, **kwargs)
            else:
                return self.base_llm._call(prompt, stop, **kwargs)
        elif self.execution_mode == ContractExecutionMode.AUTO_RETRY:
            return await self._retry_with_remediation(prompt, stop, context, **kwargs)
        else:
            raise violation
    
    async def _retry_with_remediation(self,
                                    prompt: str,
                                    stop: Optional[List[str]],
                                    context: ContractExecutionContext,
                                    max_retries: int = 3,
                                    **kwargs) -> str:
        """Retry with auto-remediation."""
        for attempt in range(max_retries):
            try:
                self.execution_metrics["retries"] += 1
                
                # Apply input remediation if available
                remediated_prompt = await self._remediate_input(prompt, context)
                
                # Call base LLM
                if hasattr(self.base_llm, '_acall'):
                    response = await self.base_llm._acall(remediated_prompt, stop, **kwargs)
                else:
                    response = self.base_llm._call(remediated_prompt, stop, **kwargs)
                
                # Validate output
                validated_response = await self._validate_output(response, context)
                return validated_response
                
            except ContractViolationError:
                if attempt == max_retries - 1:
                    raise
                continue
    
    async def _remediate_input(self, prompt: str, context: ContractExecutionContext) -> str:
        """Apply input remediation strategies."""
        # Basic remediation - could be enhanced with LLM-based remediation
        remediated = prompt
        
        # Remove potentially problematic patterns
        import re
        remediated = re.sub(r'\b(ignore|disregard|forget)\s+(previous|above|instructions?)\b', '', remediated, flags=re.IGNORECASE)
        
        # Add safety instructions if context suggests it
        if "safety" in str(context.metadata).lower():
            remediated = f"Please respond safely and appropriately: {remediated}"
        
        return remediated
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        return {
            **self.execution_metrics,
            "violation_rate": self.execution_metrics["contract_violations"] / max(self.execution_metrics["total_calls"], 1),
            "auto_remediation_rate": self.execution_metrics["auto_remediation_success"] / max(self.execution_metrics["contract_violations"], 1)
        }


class LangChainContractValidator:
    """Validates LangChain components for contract compatibility."""
    
    def __init__(self):
        self.validation_cache = {}
        
    def validate_chain_compatibility(self, chain: Chain) -> ValidationResult:
        """Validate that a chain is compatible with contract enforcement."""
        try:
            # Check if chain has required methods
            required_methods = ['__call__', '_call', 'input_keys', 'output_keys']
            missing_methods = [method for method in required_methods if not hasattr(chain, method)]
            
            if missing_methods:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Chain missing required methods: {missing_methods}"
                )
            
            # Check input/output key compatibility
            if hasattr(chain, 'input_keys') and len(chain.input_keys) == 0:
                return ValidationResult(
                    is_valid=False,
                    error_message="Chain has no input keys defined"
                )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Chain validation error: {e}"
            )
    
    def validate_llm_compatibility(self, llm: Union[LLM, BaseChatModel]) -> ValidationResult:
        """Validate that an LLM is compatible with contract enforcement."""
        try:
            # Check basic LLM interface
            if not hasattr(llm, '_call') and not hasattr(llm, '_acall'):
                return ValidationResult(
                    is_valid=False,
                    error_message="LLM missing required _call or _acall method"
                )
            
            # Check if it's a proper LangChain LLM
            if not isinstance(llm, (LLM, BaseChatModel)):
                return ValidationResult(
                    is_valid=False,
                    error_message="Object is not a valid LangChain LLM or ChatModel"
                )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"LLM validation error: {e}"
            )


class ChainContractOrchestrator:
    """Orchestrates contract enforcement across LangChain chains."""
    
    def __init__(self):
        self.chain_contracts = {}  # chain_id -> contracts
        self.execution_history = []
        self.orchestration_metrics = {
            "chains_orchestrated": 0,
            "total_executions": 0,
            "contract_violations": 0
        }
    
    def register_chain_contracts(self,
                                chain_id: str,
                                input_contracts: Optional[List[ContractBase]] = None,
                                output_contracts: Optional[List[ContractBase]] = None,
                                intermediate_contracts: Optional[Dict[str, List[ContractBase]]] = None):
        """Register contracts for a specific chain."""
        self.chain_contracts[chain_id] = {
            "input_contracts": input_contracts or [],
            "output_contracts": output_contracts or [],
            "intermediate_contracts": intermediate_contracts or {}
        }
        self.orchestration_metrics["chains_orchestrated"] += 1
    
    async def execute_chain_with_contracts(self,
                                         chain: Chain,
                                         chain_id: str,
                                         inputs: Dict[str, Any],
                                         **kwargs) -> Dict[str, Any]:
        """Execute a chain with contract enforcement."""
        self.orchestration_metrics["total_executions"] += 1
        
        execution_context = ContractExecutionContext(
            chain_id=chain_id,
            step_name="chain_execution",
            input_data=inputs,
            metadata=kwargs
        )
        
        try:
            # Validate inputs
            await self._validate_chain_inputs(chain_id, inputs, execution_context)
            
            # Execute chain
            if hasattr(chain, 'acall'):
                result = await chain.acall(inputs, **kwargs)
            else:
                result = chain(inputs, **kwargs)
            
            # Validate outputs
            validated_result = await self._validate_chain_outputs(chain_id, result, execution_context)
            
            # Record execution
            self.execution_history.append({
                "chain_id": chain_id,
                "timestamp": time.time(),
                "success": True,
                "context": execution_context.to_dict()
            })
            
            return validated_result
            
        except Exception as e:
            self.orchestration_metrics["contract_violations"] += 1
            self.execution_history.append({
                "chain_id": chain_id,
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
                "context": execution_context.to_dict()
            })
            raise
    
    async def _validate_chain_inputs(self,
                                   chain_id: str,
                                   inputs: Dict[str, Any],
                                   context: ContractExecutionContext):
        """Validate chain inputs against registered contracts."""
        if chain_id not in self.chain_contracts:
            return
        
        input_contracts = self.chain_contracts[chain_id]["input_contracts"]
        if not input_contracts:
            return
        
        validator = PerformanceOptimizedInputValidator(input_contracts)
        result = await validator.validate_async(inputs)
        
        if not result.is_valid:
            raise ContractViolationError(f"Chain input validation failed: {result.error_message}")
    
    async def _validate_chain_outputs(self,
                                    chain_id: str,
                                    outputs: Dict[str, Any],
                                    context: ContractExecutionContext) -> Dict[str, Any]:
        """Validate chain outputs against registered contracts."""
        if chain_id not in self.chain_contracts:
            return outputs
        
        output_contracts = self.chain_contracts[chain_id]["output_contracts"]
        if not output_contracts:
            return outputs
        
        validator = PerformanceOptimizedOutputValidator(output_contracts)
        result = await validator.validate_async(outputs)
        
        if not result.is_valid:
            if result.auto_fixed_content:
                return result.auto_fixed_content
            else:
                raise ContractViolationError(f"Chain output validation failed: {result.error_message}")
        
        return outputs
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get orchestration metrics."""
        return {
            **self.orchestration_metrics,
            "violation_rate": self.orchestration_metrics["contract_violations"] / max(self.orchestration_metrics["total_executions"], 1),
            "execution_history_size": len(self.execution_history)
        }


class ContractOutputParser(BaseOutputParser if LANGCHAIN_AVAILABLE else object):
    """LangChain output parser with contract enforcement."""
    
    def __init__(self,
                 base_parser: Optional[BaseOutputParser] = None,
                 output_contracts: Optional[List[ContractBase]] = None,
                 auto_remediation: bool = True):
        """Initialize contract output parser.
        
        Args:
            base_parser: Base LangChain output parser
            output_contracts: Output validation contracts
            auto_remediation: Enable automatic output remediation
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LangChain integration")
            
        self.base_parser = base_parser
        self.output_contracts = output_contracts or []
        self.auto_remediation = auto_remediation
        
        if self.output_contracts:
            self.output_validator = PerformanceOptimizedOutputValidator(self.output_contracts)
    
    def parse(self, text: str) -> Any:
        """Parse text with contract enforcement."""
        # Apply base parser first
        if self.base_parser:
            parsed_output = self.base_parser.parse(text)
        else:
            parsed_output = text
        
        # Apply contract validation
        if self.output_contracts:
            validation_data = {
                "parsed_output": parsed_output,
                "original_text": text
            }
            
            result = asyncio.run(self.output_validator.validate_async(validation_data))
            
            if not result.is_valid:
                if self.auto_remediation and result.auto_fixed_content:
                    return result.auto_fixed_content
                else:
                    raise ContractViolationError(f"Output parsing failed contract validation: {result.error_message}")
        
        return parsed_output
    
    def get_format_instructions(self) -> str:
        """Get format instructions including contract requirements."""
        instructions = []
        
        if self.base_parser and hasattr(self.base_parser, 'get_format_instructions'):
            instructions.append(self.base_parser.get_format_instructions())
        
        if self.output_contracts:
            contract_instructions = []
            for contract in self.output_contracts:
                if hasattr(contract, 'get_format_instructions'):
                    contract_instructions.append(contract.get_format_instructions())
            
            if contract_instructions:
                instructions.append("Contract Requirements:\n" + "\n".join(contract_instructions))
        
        return "\n\n".join(instructions)


class ContractAgent(Agent if LANGCHAIN_AVAILABLE else object):
    """LangChain agent with contract enforcement."""
    
    def __init__(self,
                 base_agent: Agent,
                 action_contracts: Optional[List[ContractBase]] = None,
                 observation_contracts: Optional[List[ContractBase]] = None):
        """Initialize contract agent."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LangChain integration")
            
        self.base_agent = base_agent
        self.action_contracts = action_contracts or []
        self.observation_contracts = observation_contracts or []
        
        # Initialize validators
        if self.action_contracts:
            self.action_validator = PerformanceOptimizedInputValidator(self.action_contracts)
        if self.observation_contracts:
            self.observation_validator = PerformanceOptimizedOutputValidator(self.observation_contracts)
    
    def plan(self, intermediate_steps, **kwargs):
        """Plan next action with contract enforcement."""
        planned_action = self.base_agent.plan(intermediate_steps, **kwargs)
        
        # Validate planned action
        if self.action_contracts and planned_action:
            action_data = {
                "action": planned_action.tool,
                "action_input": planned_action.tool_input,
                "intermediate_steps": intermediate_steps
            }
            
            result = asyncio.run(self.action_validator.validate_async(action_data))
            if not result.is_valid:
                raise ContractViolationError(f"Agent action validation failed: {result.error_message}")
        
        return planned_action


class ContractTool(BaseTool if LANGCHAIN_AVAILABLE else object):
    """LangChain tool with contract enforcement."""
    
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 input_contracts: Optional[List[ContractBase]] = None,
                 output_contracts: Optional[List[ContractBase]] = None):
        """Initialize contract tool."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LangChain integration")
            
        super().__init__(name=name, description=description, func=func)
        
        self.input_contracts = input_contracts or []
        self.output_contracts = output_contracts or []
        
        # Initialize validators
        if self.input_contracts:
            self.input_validator = PerformanceOptimizedInputValidator(self.input_contracts)
        if self.output_contracts:
            self.output_validator = PerformanceOptimizedOutputValidator(self.output_contracts)
    
    def _run(self, query: str) -> str:
        """Run tool with contract enforcement."""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """Async run tool with contract enforcement."""
        # Validate input
        if self.input_contracts:
            input_data = {"query": query}
            result = await self.input_validator.validate_async(input_data)
            if not result.is_valid:
                raise ContractViolationError(f"Tool input validation failed: {result.error_message}")
        
        # Execute tool function
        if asyncio.iscoroutinefunction(self.func):
            output = await self.func(query)
        else:
            output = self.func(query)
        
        # Validate output
        if self.output_contracts:
            output_data = {"output": output, "query": query}
            result = await self.output_validator.validate_async(output_data)
            if not result.is_valid:
                if result.auto_fixed_content:
                    return result.auto_fixed_content
                else:
                    raise ContractViolationError(f"Tool output validation failed: {result.error_message}")
        
        return output


class ContractCallbackHandler(BaseCallbackHandler if LANGCHAIN_AVAILABLE else object):
    """LangChain callback handler for contract monitoring."""
    
    def __init__(self):
        """Initialize contract callback handler."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LangChain integration")
            
        super().__init__()
        self.contract_events = []
        self.violation_count = 0
        self.execution_count = 0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts executing."""
        self.execution_count += 1
        self.contract_events.append({
            "event": "llm_start",
            "timestamp": time.time(),
            "prompts": prompts,
            "metadata": kwargs
        })
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes executing."""
        self.contract_events.append({
            "event": "llm_end",
            "timestamp": time.time(),
            "response": response,
            "metadata": kwargs
        })
    
    def on_llm_error(self, error, **kwargs):
        """Called when LLM encounters an error."""
        if isinstance(error, ContractViolationError):
            self.violation_count += 1
        
        self.contract_events.append({
            "event": "llm_error",
            "timestamp": time.time(),
            "error": str(error),
            "error_type": type(error).__name__,
            "metadata": kwargs
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get callback metrics."""
        return {
            "total_executions": self.execution_count,
            "contract_violations": self.violation_count,
            "violation_rate": self.violation_count / max(self.execution_count, 1),
            "total_events": len(self.contract_events)
        }


# Convenience functions

def create_contract_chain(chain: Chain,
                        chain_id: str,
                        input_contracts: Optional[List[ContractBase]] = None,
                        output_contracts: Optional[List[ContractBase]] = None,
                        orchestrator: Optional[ChainContractOrchestrator] = None) -> ChainContractOrchestrator:
    """Create a contract-enforced chain."""
    if orchestrator is None:
        orchestrator = ChainContractOrchestrator()
    
    orchestrator.register_chain_contracts(
        chain_id=chain_id,
        input_contracts=input_contracts,
        output_contracts=output_contracts
    )
    
    return orchestrator


def create_contract_agent(base_agent: Agent,
                        action_contracts: Optional[List[ContractBase]] = None,
                        observation_contracts: Optional[List[ContractBase]] = None) -> ContractAgent:
    """Create a contract-enforced agent."""
    return ContractAgent(
        base_agent=base_agent,
        action_contracts=action_contracts,
        observation_contracts=observation_contracts
    )


# Example usage and demonstration
def example_langchain_integration():
    """Example of LangChain integration usage."""
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available for example")
        return
    
    # This would be implemented with actual LangChain components
    # when LangChain is available in the environment
    print("LangChain integration example would be implemented here")