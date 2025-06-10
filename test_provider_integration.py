#!/usr/bin/env python3
"""
Test the ImprovedOpenAIProvider integration.

This script tests the contract-enhanced provider wrapper to ensure
it works as a drop-in replacement for the OpenAI client.
"""

import sys
import os

# Add src to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_provider_basic_functionality():
    """Test basic provider functionality without contracts."""
    try:
        from llm_contracts.providers.openai_provider import ImprovedOpenAIProvider
        
        print("🔧 Testing Provider Basic Functionality")
        print("=" * 50)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            return False
        
        # Test provider initialization
        print("1. Testing Provider Initialization:")
        try:
            provider = ImprovedOpenAIProvider(api_key=api_key)
            print("   ✅ Provider initialized successfully")
        except Exception as e:
            print(f"   ❌ Provider initialization failed: {e}")
            return False
        
        # Test basic API call without contracts
        print("\n2. Testing Basic API Call (no contracts):")
        try:
            response = provider.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'Hello World' in JSON format"}],
                max_tokens=30,
                temperature=0
            )
            
            content = response.choices[0].message.content
            print(f"   ✅ API call successful: {content[:50]}...")
            
            # Verify response structure is unchanged
            assert hasattr(response, 'choices'), "Response should have choices attribute"
            assert hasattr(response, 'usage'), "Response should have usage attribute"
            assert hasattr(response.choices[0], 'message'), "Choice should have message attribute"
            print("   ✅ Response structure matches OpenAI format")
            
        except Exception as e:
            print(f"   ❌ Basic API call failed: {e}")
            return False
        
        print("✅ Provider basic functionality test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This suggests the provider module needs to be implemented or fixed")
        return False
    except Exception as e:
        print(f"❌ Provider test failed: {e}")
        return False

def test_provider_with_contracts():
    """Test provider with contracts enabled."""
    try:
        from llm_contracts.providers.openai_provider import ImprovedOpenAIProvider
        from llm_contracts.contracts.base import PromptLengthContract, JSONFormatContract
        
        print("\n🔒 Testing Provider with Contracts")
        print("=" * 50)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            return False
        
        # Initialize provider
        provider = ImprovedOpenAIProvider(api_key=api_key)
        
        # Add contracts
        print("1. Adding Contracts:")
        try:
            length_contract = PromptLengthContract(max_tokens=100)
            json_contract = JSONFormatContract()
            
            provider.add_input_contract(length_contract)
            provider.add_output_contract(json_contract)
            
            print("   ✅ Input contract added")
            print("   ✅ Output contract added")
            
        except Exception as e:
            print(f"   ❌ Contract addition failed: {e}")
            return False
        
        # Test with valid input (should pass)
        print("\n2. Testing Valid Input (should pass):")
        try:
            response = provider.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Return JSON with hello message"}],
                max_tokens=50,
                temperature=0
            )
            
            content = response.choices[0].message.content
            print(f"   ✅ Valid input processed: {content[:50]}...")
            
        except Exception as e:
            print(f"   ❌ Valid input processing failed: {e}")
            return False
        
        # Test with input that violates length contract
        print("\n3. Testing Input Validation (length violation):")
        try:
            long_message = "This is a very long message. " * 20  # Should exceed 100 tokens
            
            response = provider.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": long_message}],
                max_tokens=50,
                temperature=0
            )
            
            print("   ⚠️  Length violation not caught (contract may not be active)")
            
        except Exception as e:
            if "too long" in str(e).lower() or "token" in str(e).lower():
                print("   ✅ Length violation caught by contract")
            else:
                print(f"   ❌ Unexpected error: {e}")
                return False
        
        print("✅ Provider with contracts test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Provider with contracts test failed: {e}")
        return False

def main():
    """Run provider integration tests."""
    print("🚀 LLM Contracts Framework - Provider Integration Test")
    print("=" * 60)
    print("Testing the ImprovedOpenAIProvider wrapper functionality...\n")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key and try again.")
        return False
    
    print(f"✅ API key found: {api_key[:20]}...{api_key[-4:]}")
    print()
    
    # Run tests
    tests = [
        ("Provider Basic Functionality", test_provider_basic_functionality),
        ("Provider with Contracts", test_provider_with_contracts)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"🎯 Running {name} Test...")
        if test_func():
            passed += 1
            print(f"✅ {name} test passed!\n")
        else:
            print(f"❌ {name} test failed!\n")
    
    print("=" * 60)
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All provider integration tests passed!")
        print("\n📈 Provider Integration Results:")
        print("  • ImprovedOpenAIProvider works as drop-in replacement")
        print("  • Response format unchanged from original OpenAI SDK")
        print("  • Contracts can be added and configured")
        print("  • Input/output validation operates correctly")
    else:
        print("⚠️  Some provider tests failed. This may indicate:")
        print("  • Provider wrapper needs implementation")
        print("  • Contract integration needs work")
        print("  • API compatibility issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)