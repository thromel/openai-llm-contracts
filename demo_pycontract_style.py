#!/usr/bin/env python3
"""
Demo: PyContract-style parameter validation in LLM Contracts

This demonstrates how to use PyContract-like syntax for parameter validation
in the LLM Contracts framework.
"""

import asyncio
from typing import Dict, Any

# Import the PyContract-style implementation
from pycontract_style_example import ParameterContract, pycontract_decorator

# Import existing framework components
from src.llm_contracts.providers.openai_provider import ImprovedOpenAIProvider
from src.llm_contracts.contracts.base import InputContract, ValidationResult


class PyContractValidator(InputContract):
    """Validator that supports multiple PyContract-style parameter constraints."""
    
    def __init__(self, constraints: Dict[str, str]):
        """
        Args:
            constraints: Dict mapping parameter names to PyContract-style constraints
                        e.g., {'temperature': 'float,>=0,<=2', 'top_p': 'float,>=0,<=1'}
        """
        super().__init__(
            name="pycontract_parameters",
            description="PyContract-style parameter validation"
        )
        self.param_contracts = {
            param: ParameterContract(param, constraint)
            for param, constraint in constraints.items()
        }
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate all parameter constraints."""
        if not isinstance(data, dict):
            return ValidationResult(True, "No parameters to validate")
        
        # Check each parameter constraint
        for param, contract in self.param_contracts.items():
            result = contract.validate(data, context)
            if not result.is_valid:
                return result
        
        # Additional cross-parameter validations
        if 'temperature' in data and 'top_p' in data:
            return ValidationResult(
                False,
                "Both temperature and top_p are set. Use only one for better results.",
                auto_fix_suggestion="Remove either temperature or top_p parameter"
            )
        
        return ValidationResult(True, "All parameter constraints satisfied")


def demo_basic_usage():
    """Demonstrate basic PyContract-style validation."""
    print("=== Basic PyContract-style Validation ===\n")
    
    # Create individual parameter contracts
    temp_contract = ParameterContract('temperature', 'float,>=0,<=2')
    top_p_contract = ParameterContract('top_p', 'float,>=0,<=1')
    max_tokens_contract = ParameterContract('max_tokens', 'int,>0,<=4096')
    
    # Test data
    test_params = [
        {'temperature': 1.5, 'max_tokens': 1000},  # Valid
        {'temperature': 2.5, 'max_tokens': 1000},  # Temperature too high
        {'top_p': 0.5, 'max_tokens': 5000},       # max_tokens too high
        {'temperature': 'high'},                    # Wrong type
    ]
    
    for params in test_params:
        print(f"Testing parameters: {params}")
        
        # Validate temperature
        if 'temperature' in params:
            result = temp_contract.validate(params)
            print(f"  Temperature: {result.message}")
            if result.auto_fix_suggestion:
                print(f"    Fix: {result.auto_fix_suggestion}")
        
        # Validate top_p
        if 'top_p' in params:
            result = top_p_contract.validate(params)
            print(f"  Top-p: {result.message}")
        
        # Validate max_tokens
        result = max_tokens_contract.validate(params)
        print(f"  Max tokens: {result.message}")
        print()


def demo_decorator_usage():
    """Demonstrate PyContract-style decorator."""
    print("\n=== PyContract-style Decorator ===\n")
    
    @pycontract_decorator(
        temperature='float,>=0,<=2',
        top_p='float,>=0,<=1',
        max_tokens='int,>0,<=4096',
        presence_penalty='float,>=-2,<=2'
    )
    def generate_completion(prompt: str, **params):
        """Simulated LLM completion with validated parameters."""
        return f"Generated text for '{prompt}' with params: {params}"
    
    # Valid call
    try:
        result = generate_completion(
            prompt="Hello world",
            temperature=0.7,
            max_tokens=100
        )
        print(f"Success: {result}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Invalid call - temperature too high
    try:
        result = generate_completion(
            prompt="Hello world",
            temperature=3.0,  # Invalid!
            max_tokens=100
        )
        print(f"Success: {result}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Invalid call - wrong type
    try:
        result = generate_completion(
            prompt="Hello world",
            temperature="very hot",  # Invalid type!
            max_tokens=100
        )
        print(f"Success: {result}")
    except ValueError as e:
        print(f"Error: {e}")


async def demo_provider_integration():
    """Demonstrate integration with OpenAI provider."""
    print("\n\n=== Provider Integration ===\n")
    
    # Create mock provider for demo (replace with real API key for actual use)
    provider = ImprovedOpenAIProvider(api_key="mock-key-for-demo")
    
    # Add PyContract-style validation
    provider.add_input_contract(PyContractValidator({
        'temperature': 'float,>=0,<=2',
        'top_p': 'float,>=0,<=1',
        'max_tokens': 'int,>0,<=4096',
        'n': 'int,>=1,<=10',
        'presence_penalty': 'float,>=-2,<=2',
        'frequency_penalty': 'float,>=-2,<=2'
    }))
    
    # Test cases
    test_cases = [
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100
        },
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 2.5,  # Invalid - too high
            "max_tokens": 100
        },
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_p": 0.9,  # Invalid - both temperature and top_p set
            "max_tokens": 100
        }
    ]
    
    for i, params in enumerate(test_cases):
        print(f"Test case {i+1}: {params}")
        try:
            # Validate input contracts
            validation_result = await provider.validate_input(params)
            if validation_result.is_valid:
                print("  ✓ Validation passed")
            else:
                print(f"  ✗ Validation failed: {validation_result.message}")
                if validation_result.auto_fix_suggestion:
                    print(f"    Suggestion: {validation_result.auto_fix_suggestion}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()


def demo_extended_syntax():
    """Show proposed LLMCL syntax extensions."""
    print("\n=== Proposed LLMCL Syntax Extensions ===\n")
    
    print("Current LLMCL syntax:")
    print("""
contract ParameterValidation {
    require temperature >= 0 and temperature <= 2
    require top_p >= 0 and top_p <= 1
    require not (exists(temperature) and exists(top_p))
}
""")
    
    print("\nProposed PyContract-style extensions:")
    print("""
contract ParameterValidation {
    # Compact parameter constraints
    param temperature: float[0:2]
    param top_p: float[0:1]
    param max_tokens: int(0:4096]
    param n: int[1:10]
    
    # Type constraints
    require type(messages) == list
    require all(type(msg) == dict for msg in messages)
    
    # Cross-parameter constraints
    require not (exists(temperature) and exists(top_p))
        message: "Use either temperature or top_p, not both"
    
    # Output type contracts
    ensure type(response) == dict
    ensure 'choices' in response
}
""")


if __name__ == "__main__":
    print("PyContract-style Parameter Validation Demo")
    print("=" * 50)
    
    # Run demos
    demo_basic_usage()
    demo_decorator_usage()
    asyncio.run(demo_provider_integration())
    demo_extended_syntax()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nKey benefits of PyContract-style syntax:")
    print("- Concise and readable parameter constraints")
    print("- Type checking with range validation") 
    print("- Auto-fix suggestions for invalid values")
    print("- Easy integration with existing contract system")
    print("- Familiar syntax for Python developers")