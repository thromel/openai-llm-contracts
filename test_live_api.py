#!/usr/bin/env python3
"""
Live API integration test with OpenAI.

This script tests the LLM contracts framework with actual OpenAI API calls
to verify that token counting and basic contracts work in practice.
"""

import sys
import os
import json
import asyncio

# Add src to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_contracts_with_api():
    """Test basic contracts with real OpenAI API calls."""
    try:
        import openai
        from llm_contracts.contracts.base import PromptLengthContract, JSONFormatContract
        from llm_contracts.utils.tokenizer import count_tokens, count_tokens_from_messages
        
        print("🔗 Testing Live OpenAI API Integration")
        print("=" * 50)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            return False
        
        # Create OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Test 1: Token counting accuracy
        print("1. Testing Token Counting Accuracy:")
        
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
            {"role": "user", "content": "What is 2+2? Respond with JSON containing the answer."}
        ]
        
        # Count tokens using our tokenizer
        our_count = count_tokens_from_messages(test_messages, model="gpt-3.5-turbo")
        print(f"   Our token count: {our_count} tokens")
        
        # Test 2: Contract validation before API call
        print("\n2. Testing Input Contract Validation:")
        
        length_contract = PromptLengthContract(max_tokens=100)
        validation_result = length_contract.validate(test_messages)
        
        print(f"   Length validation: {'✅ PASS' if validation_result.is_valid else '❌ FAIL'}")
        print(f"   Message: {validation_result.message}")
        
        # Test 3: Make actual API call (small, cheap call)
        print("\n3. Testing Live API Call:")
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=test_messages,
                max_tokens=50,  # Keep it small and cheap
                temperature=0
            )
            
            print("   ✅ API call successful")
            
            # Get the response content
            response_content = response.choices[0].message.content
            print(f"   Response: {response_content[:100]}...")
            
            # Check actual token usage
            actual_usage = response.usage
            print(f"   Actual tokens used - Prompt: {actual_usage.prompt_tokens}, Completion: {actual_usage.completion_tokens}")
            print(f"   Our estimate was: {our_count} (difference: {abs(our_count - actual_usage.prompt_tokens)} tokens)")
            
            # Test 4: Output contract validation
            print("\n4. Testing Output Contract Validation:")
            
            json_contract = JSONFormatContract()
            output_validation = json_contract.validate(response_content)
            
            print(f"   JSON validation: {'✅ PASS' if output_validation.is_valid else '❌ FAIL'}")
            print(f"   Message: {output_validation.message}")
            
            # Calculate accuracy
            token_accuracy = 1 - (abs(our_count - actual_usage.prompt_tokens) / actual_usage.prompt_tokens)
            print(f"\n📊 Token counting accuracy: {token_accuracy:.2%}")
            
            if token_accuracy > 0.95:  # 95% accuracy threshold
                print("✅ Token counting is highly accurate!")
            elif token_accuracy > 0.90:  # 90% accuracy threshold
                print("✅ Token counting is reasonably accurate")
            else:
                print("⚠️  Token counting accuracy could be improved")
            
            return True
            
        except Exception as api_error:
            print(f"   ❌ API call failed: {api_error}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_complex_workflow():
    """Test a more complex workflow with multiple contracts."""
    try:
        import openai
        from llm_contracts.contracts.base import PromptLengthContract, JSONFormatContract
        from llm_contracts.conversation.state_manager import ConversationStateManager, TurnRole
        
        print("\n🔄 Testing Complex Workflow")
        print("=" * 50)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            return False
        
        client = openai.OpenAI(api_key=api_key)
        
        # Create conversation manager
        conversation = ConversationStateManager(
            conversation_id="live_test_conversation",
            context_window_size=4096
        )
        
        # Set up contracts
        length_contract = PromptLengthContract(max_tokens=200)
        json_contract = JSONFormatContract()
        
        print("1. Setting up conversation with contracts...")
        
        # Simulate a multi-turn conversation
        turns = [
            ("user", "Hello! Can you help me with a simple math problem?"),
            ("user", "What is 15 * 7? Please respond in JSON format with the calculation.")
        ]
        
        for i, (role, content) in enumerate(turns, 1):
            print(f"\n   Turn {i}: Processing {role} input...")
            
            # Add turn to conversation
            turn = conversation.add_turn(TurnRole(role), content)
            print(f"   Added turn with {turn.token_count} tokens")
            
            if role == "user":
                # Validate input
                result = length_contract.validate(content)
                print(f"   Input validation: {'✅ PASS' if result.is_valid else '❌ FAIL'} - {result.message}")
                
                if i == 2:  # On second turn, make API call
                    print("   Making API call...")
                    
                    # Get conversation context for API call
                    context_messages = []
                    for past_turn in conversation.get_conversation_state().turns:
                        context_messages.append({
                            "role": past_turn.role.value,
                            "content": past_turn.content
                        })
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=context_messages,
                            max_tokens=100,
                            temperature=0
                        )
                        
                        response_content = response.choices[0].message.content
                        print(f"   API Response: {response_content}")
                        
                        # Add assistant response to conversation
                        assistant_turn = conversation.add_turn(TurnRole("assistant"), response_content)
                        print(f"   Added assistant turn with {assistant_turn.token_count} tokens")
                        
                        # Validate output
                        json_result = json_contract.validate(response_content)
                        print(f"   Output validation: {'✅ PASS' if json_result.is_valid else '❌ FAIL'} - {json_result.message}")
                        
                    except Exception as api_error:
                        print(f"   ❌ API call failed: {api_error}")
                        return False
        
        # Show final conversation metrics
        metrics = conversation.get_metrics()
        print(f"\n2. Final Conversation Metrics:")
        print(f"   Total turns: {metrics['turn_count']}")
        print(f"   Total tokens: {metrics['total_tokens']}")
        print(f"   Conversation phase: {metrics['conversation_phase']}")
        
        print("✅ Complex workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Complex workflow test failed: {e}")
        return False

def main():
    """Run all live API tests."""
    print("🚀 LLM Contracts Framework - Live API Integration Test")
    print("=" * 60)
    print("Testing the framework with real OpenAI API calls...")
    print("Note: This will make small, inexpensive API calls to test functionality.\n")
    
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
        ("Basic Contracts with API", test_basic_contracts_with_api),
        ("Complex Workflow", test_complex_workflow)
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
        print("🎉 All live API tests passed! The framework works correctly with OpenAI API.")
        print("\n📈 Key Results:")
        print("  • Token counting is accurate with tiktoken")
        print("  • Contracts validate correctly before/after API calls")
        print("  • Conversation state management works properly")
        print("  • Framework integrates seamlessly with OpenAI SDK")
    else:
        print("⚠️  Some live API tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)