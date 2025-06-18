"""
PyContract-style syntax support for LLM Contracts

This demonstrates how we can extend the existing LLMCL syntax to support
PyContract-like parameter validation syntax.
"""

from typing import Any, Dict, Optional
from src.llm_contracts.contracts.base import InputContract, ValidationResult
from src.llm_contracts.language.integration import llmcl_contract

# Example 1: Class-based approach with PyContract-like validation
class ParameterContract(InputContract):
    """Contract that validates parameters using PyContract-like expressions."""
    
    def __init__(self, param_name: str, constraint: str):
        """
        Args:
            param_name: Parameter to validate (e.g., 'temperature')
            constraint: PyContract-style constraint (e.g., 'float,>=0,<=2')
        """
        super().__init__(
            name=f"{param_name}_validation",
            description=f"Validate {param_name}: {constraint}"
        )
        self.param_name = param_name
        self.constraint = constraint
        self._parse_constraint()
    
    def _parse_constraint(self):
        """Parse PyContract-style constraint string."""
        parts = self.constraint.split(',')
        self.type_check = None
        self.min_val = None
        self.max_val = None
        
        for part in parts:
            part = part.strip()
            if part in ['int', 'float', 'str', 'bool']:
                self.type_check = part
            elif part.startswith('>='):
                self.min_val = float(part[2:])
            elif part.startswith('<='):
                self.max_val = float(part[2:])
            elif part.startswith('>'):
                self.min_val = float(part[1:])
            elif part.startswith('<'):
                self.max_val = float(part[1:])
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        if not isinstance(data, dict) or self.param_name not in data:
            return ValidationResult(True, f"Parameter {self.param_name} not present")
        
        value = data[self.param_name]
        
        # Type checking
        if self.type_check:
            type_map = {'int': int, 'float': (int, float), 'str': str, 'bool': bool}
            expected_type = type_map.get(self.type_check)
            if not isinstance(value, expected_type):
                return ValidationResult(
                    False, 
                    f"{self.param_name} must be {self.type_check}, got {type(value).__name__}"
                )
        
        # Range checking for numeric types
        if isinstance(value, (int, float)):
            if self.min_val is not None and value < self.min_val:
                return ValidationResult(
                    False,
                    f"{self.param_name} must be >= {self.min_val}, got {value}",
                    auto_fix_suggestion={self.param_name: self.min_val}
                )
            if self.max_val is not None and value > self.max_val:
                return ValidationResult(
                    False,
                    f"{self.param_name} must be <= {self.max_val}, got {value}",
                    auto_fix_suggestion={self.param_name: self.max_val}
                )
        
        return ValidationResult(True, f"{self.param_name} validation passed")


# Example 2: Extended LLMCL syntax with PyContract-style parameter constraints
pycontract_style_llmcl = """
contract LLMParameters {
    # PyContract-style parameter validation
    param temperature: float,>=0,<=2
        message: "Temperature must be between 0 and 2"
        auto_fix: clamp(temperature, 0, 2)
    
    param top_p: float,>=0,<=1
        message: "top_p must be between 0 and 1"
        auto_fix: clamp(top_p, 0, 1)
    
    param max_tokens: int,>0,<=4096
        message: "max_tokens must be positive and <= 4096"
    
    param n: int,>0,<=10
        message: "n (number of completions) must be between 1 and 10"
    
    # Mutual exclusivity constraint
    require not (exists(temperature) and exists(top_p))
        message: "Use either temperature or top_p, not both"
    
    # Combined constraints
    require not (temperature > 1.5 and max_tokens > 2048)
        message: "High temperature with large token limit may produce inconsistent results"
}
"""

# Example 3: Decorator with PyContract-style syntax
def pycontract_decorator(**param_constraints):
    """Decorator that applies PyContract-style constraints to function parameters."""
    def decorator(func):
        contracts = {}
        for param, constraint in param_constraints.items():
            contracts[param] = ParameterContract(param, constraint)
        
        def wrapper(**kwargs):
            # Validate parameters
            for param, contract in contracts.items():
                result = contract.validate(kwargs)
                if not result.is_valid:
                    raise ValueError(f"Contract violation: {result.message}")
            return func(**kwargs)
        
        return wrapper
    return decorator


# Example usage:
@pycontract_decorator(
    temperature='float,>=0,<=2',
    top_p='float,>=0,<=1',
    max_tokens='int,>0,<=4096'
)
def generate_text(prompt: str, **kwargs):
    """Generate text with validated parameters."""
    # Parameters are already validated by decorator
    return f"Generating with params: {kwargs}"


# Example 4: Proposed LLMCL syntax extension
enhanced_llmcl_syntax = """
contract EnhancedValidation {
    # Current LLMCL syntax
    require len(prompt) > 0 and len(prompt) < 4000
    
    # Proposed PyContract-like extensions:
    # Type annotations
    require type(prompt) == str
    require type(messages) == list
    
    # Parameter contracts with type and range
    param temperature: float[0:2]  # Compact range syntax
    param top_p: float[0:1]
    param presence_penalty: float[-2:2]
    param frequency_penalty: float[-2:2]
    
    # Collection constraints
    require all(type(msg) == dict for msg in messages)
    require all('role' in msg and 'content' in msg for msg in messages)
    
    # Output type contracts
    ensure type(response) == str or type(response) == dict
    ensure type(response) == dict => 'choices' in response
    
    # Numeric constraints with units (proposed)
    ensure response_time < 10s
    ensure token_count < 4096 tokens
}
"""


# Example 5: Integration with existing system
def create_pycontract_style_validation():
    """Show how to integrate PyContract-style validation with existing system."""
    from src.llm_contracts.providers.openai_provider import ImprovedOpenAIProvider
    
    # Create provider
    provider = ImprovedOpenAIProvider(api_key="...")
    
    # Add PyContract-style parameter contracts
    provider.add_input_contract(ParameterContract('temperature', 'float,>=0,<=2'))
    provider.add_input_contract(ParameterContract('top_p', 'float,>=0,<=1'))
    provider.add_input_contract(ParameterContract('max_tokens', 'int,>0,<=4096'))
    provider.add_input_contract(ParameterContract('n', 'int,>=1,<=10'))
    
    return provider


if __name__ == "__main__":
    # Test the PyContract-style validation
    contract = ParameterContract('temperature', 'float,>=0,<=2')
    
    # Test cases
    test_cases = [
        {'temperature': 1.5},  # Valid
        {'temperature': 2.5},  # Too high
        {'temperature': -0.5}, # Too low
        {'temperature': "1"},  # Wrong type
        {},                    # Missing parameter (valid - optional)
    ]
    
    for test in test_cases:
        result = contract.validate(test)
        print(f"Input: {test}")
        print(f"Valid: {result.is_valid}")
        print(f"Message: {result.message}")
        if result.auto_fix_suggestion:
            print(f"Fix: {result.auto_fix_suggestion}")
        print()