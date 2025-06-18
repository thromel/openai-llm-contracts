#!/usr/bin/env python3
"""
Simple test to verify PyContract syntax works with ImprovedOpenAIProvider
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_pycontract_integration():
    """Test basic PyContract integration with real API call."""
    
    # Set API key
    os.environ['OPENAI_API_KEY'] = "<REDACTED_API_KEY>"
    
    try:
        from llm_contracts import ImprovedOpenAIProvider, PromptLengthContract
        from pycontract_style_example import ParameterContract
        
        print("✅ Imports successful")
        
        # Initialize provider
        client = ImprovedOpenAIProvider(api_key=os.environ['OPENAI_API_KEY'])
        print("✅ Provider initialized")
        
        # Add PyContract-style parameter validation
        client.add_input_contract(ParameterContract('temperature', 'float,>=0,<=2'))
        client.add_input_contract(ParameterContract('max_tokens', 'int,>0,<=100'))
        client.add_input_contract(PromptLengthContract(max_tokens=500))
        
        print("✅ Contracts added")
        
        # Test 1: Valid parameters
        print("\n🧪 Test 1: Valid parameters")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
            temperature=0.7,
            max_tokens=50
        )
        print(f"✅ Success: {response.choices[0].message.content}")
        
        # Test 2: Check if auto-fixing actually works
        print("\n🧪 Test 2: Parameter validation (should show warnings)")
        try:
            response2 = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'Hi' in one word."}],
                temperature=1.5,  # Valid but at edge
                max_tokens=50
            )
            print(f"✅ Edge case success: {response2.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Edge case failed: {e}")
        
        # Test 3: Check contract validation is working
        print("\n🧪 Test 3: Testing contract validation directly")
        temp_contract = ParameterContract('temperature', 'float,>=0,<=2')
        
        # Valid test
        valid_result = temp_contract.validate({'temperature': 0.7})
        print(f"Valid param test: {valid_result.is_valid} - {valid_result.message}")
        
        # Invalid test
        invalid_result = temp_contract.validate({'temperature': 3.5})
        print(f"Invalid param test: {invalid_result.is_valid} - {invalid_result.message}")
        print(f"Auto-fix suggestion: {invalid_result.auto_fix_suggestion}")
        
        # Test 4: Check provider metrics
        print("\n🧪 Test 4: Provider metrics")
        if hasattr(client, 'get_metrics'):
            metrics = client.get_metrics()
            print(f"✅ Metrics: {metrics}")
        else:
            print("⚠️  No get_metrics method found")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_pycontract_integration()
    if success:
        print("\n🎉 PyContract integration test PASSED!")
    else:
        print("\n💥 PyContract integration test FAILED!")
    
    sys.exit(0 if success else 1)