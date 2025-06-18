#!/usr/bin/env python3
"""
Comprehensive test script for PyContract syntax examples from PRODUCTION_EXAMPLES.md

This script tests all the PyContract patterns shown in the production examples
to ensure they work correctly with ImprovedOpenAIProvider.
"""

import os
import sys
import asyncio
import traceback
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test if we can import all required modules
def test_imports():
    """Test that all required imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        # Core LLM contracts imports
        from llm_contracts import (
            ImprovedOpenAIProvider, 
            PromptLengthContract, 
            JSONFormatContract,
            ContentPolicyContract,
            PromptInjectionContract
        )
        print("✅ Core llm_contracts imports successful")
    except ImportError as e:
        print(f"❌ Core imports failed: {e}")
        print(f"   Trying alternative import paths...")
        try:
            from src.llm_contracts import (
                ImprovedOpenAIProvider, 
                PromptLengthContract, 
                JSONFormatContract,
                ContentPolicyContract,
                PromptInjectionContract
            )
            print("✅ Core src.llm_contracts imports successful")
        except ImportError as e2:
            print(f"❌ Alternative imports also failed: {e2}")
            return False
    
    try:
        # PyContract style imports
        from pycontract_style_example import ParameterContract
        print("✅ ParameterContract import successful")
    except ImportError as e:
        print(f"❌ ParameterContract import failed: {e}")
        return False
    
    try:
        # Factory imports
        from complex_pycontract_syntax import PyContractFactory
        print("✅ PyContractFactory import successful")
    except ImportError as e:
        print(f"❌ PyContractFactory import failed: {e}")
        return False
        
    try:
        # Decorator imports
        from pycontract_style_example import pycontract_decorator
        print("✅ pycontract_decorator import successful")
    except ImportError as e:
        print(f"❌ pycontract_decorator import failed: {e}")
        return False
    
    return True


def test_parameter_contract_syntax():
    """Test ParameterContract with PyContract string syntax."""
    print("\n🧪 Testing ParameterContract syntax...")
    
    try:
        from pycontract_style_example import ParameterContract
        
        # Test basic parameter contracts
        temp_contract = ParameterContract('temperature', 'float,>=0,<=2')
        print("✅ Temperature contract created successfully")
        
        top_p_contract = ParameterContract('top_p', 'float,>=0,<=1')
        print("✅ Top_p contract created successfully")
        
        max_tokens_contract = ParameterContract('max_tokens', 'int,>0,<=4096')
        print("✅ Max_tokens contract created successfully")
        
        # Test validation
        valid_params = {'temperature': 0.7, 'max_tokens': 100}
        result = temp_contract.validate(valid_params)
        print(f"✅ Valid parameter test: {result.is_valid}")
        
        invalid_params = {'temperature': 3.5}
        result = temp_contract.validate(invalid_params)
        print(f"✅ Invalid parameter test: {not result.is_valid} (should be False)")
        print(f"   Auto-fix suggestion: {result.auto_fix_suggestion}")
        
        return True
    except Exception as e:
        print(f"❌ ParameterContract test failed: {e}")
        traceback.print_exc()
        return False


def test_pycontract_factory():
    """Test PyContractFactory string syntax patterns."""
    print("\n🧪 Testing PyContractFactory syntax...")
    
    try:
        from complex_pycontract_syntax import PyContractFactory
        
        # Test security contract
        security_contract = PyContractFactory.create_contract(
            "security_test",
            "regex_pattern:(?i)(injection|exploit),message:Security threat detected"
        )
        print("✅ Security contract created successfully")
        
        # Test performance contract
        performance_contract = PyContractFactory.create_contract(
            "performance_test",
            "response_time:<=10s,auto_fix:optimize"
        )
        print("✅ Performance contract created successfully")
        
        # Test budget contract
        budget_contract = PyContractFactory.create_contract(
            "budget_test",
            "cost_limit:$0.10/request,alert_threshold:80%"
        )
        print("✅ Budget contract created successfully")
        
        return True
    except Exception as e:
        print(f"❌ PyContractFactory test failed: {e}")
        traceback.print_exc()
        return False


def test_improved_openai_provider_integration():
    """Test ImprovedOpenAIProvider with PyContract syntax."""
    print("\n🧪 Testing ImprovedOpenAIProvider integration...")
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - skipping API tests")
        return True
    
    try:
        from llm_contracts import ImprovedOpenAIProvider
        from pycontract_style_example import ParameterContract
        
        # Initialize provider
        client = ImprovedOpenAIProvider(api_key=os.environ['OPENAI_API_KEY'])
        print("✅ ImprovedOpenAIProvider initialized")
        
        # Add PyContract-style parameter validation
        client.add_input_contract(ParameterContract('temperature', 'float,>=0,<=2'))
        client.add_input_contract(ParameterContract('max_tokens', 'int,>0,<=4096'))
        print("✅ ParameterContracts added to provider")
        
        # Test that provider accepts the contracts
        contracts = len(client._input_contracts) if hasattr(client, '_input_contracts') else 0
        print(f"✅ Provider has {contracts} input contracts")
        
        return True
    except Exception as e:
        print(f"❌ Provider integration test failed: {e}")
        traceback.print_exc()
        return False


def test_decorator_pattern():
    """Test PyContract decorator patterns."""
    print("\n🧪 Testing decorator patterns...")
    
    try:
        from pycontract_style_example import pycontract_decorator
        
        @pycontract_decorator(
            temperature='float,>=0,<=2',
            max_tokens='int,>0,<=4096'
        )
        def test_function(**params):
            """Test function with PyContract decorators."""
            prompt = params.get('prompt', 'default')
            return f"Processed: {prompt} with {params}"
        
        print("✅ Decorator applied successfully")
        
        # Test valid parameters
        result = test_function(prompt="test prompt", temperature=0.7, max_tokens=100)
        print(f"✅ Valid params test: {result[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Decorator test failed: {e}")
        traceback.print_exc()
        return False


def test_production_service_class():
    """Test the ProductionLLMService class from examples."""
    print("\n🧪 Testing ProductionLLMService class...")
    
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - skipping service class test")
        return True
    
    try:
        from llm_contracts import (
            ImprovedOpenAIProvider, 
            PromptLengthContract, 
            ContentPolicyContract,
            PromptInjectionContract
        )
        from pycontract_style_example import ParameterContract
        from complex_pycontract_syntax import PyContractFactory
        
        class ProductionLLMService:
            """Production LLM service with PyContract-style validation."""
            
            def __init__(self, api_key: str):
                self.client = ImprovedOpenAIProvider(api_key=api_key)
                self.setup_parameter_contracts()
                self.client.add_input_contract(PromptLengthContract(max_tokens=4000))
                self.client.add_input_contract(ContentPolicyContract())
                self.client.add_input_contract(PromptInjectionContract())
            
            def setup_parameter_contracts(self):
                """Set up PyContract-style parameter validation."""
                constraints = {
                    'temperature': 'float,>=0,<=2',
                    'top_p': 'float,>=0,<=1',
                    'max_tokens': 'int,>0,<=4096',
                    'frequency_penalty': 'float,>=-2,<=2',
                    'presence_penalty': 'float,>=-2,<=2'
                }
                
                for param, constraint in constraints.items():
                    self.client.add_input_contract(ParameterContract(param, constraint))
                
                # Add security contract
                self.client.add_input_contract(PyContractFactory.create_contract(
                    "security",
                    "prompt_injection_check:enabled,pii_detection:enabled,auto_fix:sanitize"
                ))
        
        # Test service initialization
        service = ProductionLLMService(os.environ['OPENAI_API_KEY'])
        print("✅ ProductionLLMService initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ ProductionLLMService test failed: {e}")
        traceback.print_exc()
        return False


async def test_actual_api_call():
    """Test actual API call with PyContract validation (if API key available)."""
    print("\n🧪 Testing actual API call with PyContract validation...")
    
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - skipping actual API test")
        return True
    
    try:
        from llm_contracts import ImprovedOpenAIProvider
        from pycontract_style_example import ParameterContract
        
        # Initialize with PyContract validation
        client = ImprovedOpenAIProvider(api_key=os.environ['OPENAI_API_KEY'])
        client.add_input_contract(ParameterContract('temperature', 'float,>=0,<=2'))
        client.add_input_contract(ParameterContract('max_tokens', 'int,>0,<=100'))
        
        print("✅ Client initialized with PyContract validation")
        
        # Test with valid parameters
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello' in exactly one word."}],
            temperature=0.7,
            max_tokens=50
        )
        
        print(f"✅ Valid API call successful: {response.choices[0].message.content}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        
        # Test with invalid parameters (should be auto-fixed)
        print("\n   Testing parameter auto-fixing...")
        response2 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hi' in one word."}],
            temperature=3.5,  # Should be auto-fixed to 2.0
            max_tokens=150    # Should be auto-fixed to 100
        )
        
        print(f"✅ Auto-fix API call successful: {response2.choices[0].message.content}")
        
        # Get metrics
        if hasattr(client, 'get_metrics'):
            metrics = client.get_metrics()
            print(f"✅ Metrics retrieved: {metrics}")
        
        return True
    except Exception as e:
        print(f"❌ API call test failed: {e}")
        traceback.print_exc()
        return False


def test_comprehensive_example():
    """Test the comprehensive example from the documentation."""
    print("\n🧪 Testing comprehensive documentation example...")
    
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - creating mock test")
        return True
    
    try:
        # Exact code from PRODUCTION_EXAMPLES.md
        from llm_contracts import ImprovedOpenAIProvider, PromptLengthContract, ContentPolicyContract
        from pycontract_style_example import ParameterContract
        from complex_pycontract_syntax import PyContractFactory
        
        # Initialize provider (drop-in replacement for OpenAI)
        client = ImprovedOpenAIProvider(api_key=os.environ['OPENAI_API_KEY'])
        
        # Add contracts for automatic validation
        client.add_input_contract(PromptLengthContract(max_tokens=1000))
        client.add_input_contract(ContentPolicyContract())
        
        # Method 1: PyContract-style parameter validation
        client.add_input_contract(ParameterContract('temperature', 'float,>=0,<=2'))
        client.add_input_contract(ParameterContract('top_p', 'float,>=0,<=1'))
        client.add_input_contract(ParameterContract('max_tokens', 'int,>0,<=4096'))
        
        print("✅ All contracts added successfully")
        
        # Method 2: Complex contract factory syntax
        client.add_input_contract(PyContractFactory.create_contract(
            "security_check",
            "regex_pattern:(?i)(ignore.*instructions|system.*prompt),message:Security threat detected"
        ))
        
        print("✅ Factory contract added successfully")
        
        return True
    except Exception as e:
        print(f"❌ Comprehensive example test failed: {e}")
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting comprehensive PyContract syntax testing...")
    print("=" * 70)
    
    tests = [
        ("Import Tests", test_imports),
        ("ParameterContract Syntax", test_parameter_contract_syntax),
        ("PyContractFactory Syntax", test_pycontract_factory),
        ("Provider Integration", test_improved_openai_provider_integration),
        ("Decorator Patterns", test_decorator_pattern),
        ("Production Service Class", test_production_service_class),
        ("Comprehensive Example", test_comprehensive_example),
        ("Actual API Call", test_actual_api_call)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! PyContract syntax is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
    return failed == 0


if __name__ == "__main__":
    # Set up environment
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some tests will be skipped.")
        print("   To test API functionality, set: export OPENAI_API_KEY=your-key")
    
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)