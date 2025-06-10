#!/usr/bin/env python3
"""
Simple test script to verify the tiktoken-based tokenizer works correctly.
"""

import sys
import os

# Add src to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_tokenizer():
    """Test the tokenizer functionality."""
    try:
        from llm_contracts.utils.tokenizer import (
            count_tokens,
            count_tokens_from_messages,
            get_model_context_limit,
            is_within_context_limit,
            truncate_to_limit
        )
        
        print("✅ Successfully imported tokenizer module")
        
        # Test basic token counting
        test_text = "Hello, world! This is a test message."
        token_count = count_tokens(test_text)
        print(f"✅ Basic token counting: '{test_text}' = {token_count} tokens")
        
        # Test message format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ]
        message_tokens = count_tokens_from_messages(messages)
        print(f"✅ Message token counting: {len(messages)} messages = {message_tokens} tokens")
        
        # Test context limits
        gpt4_limit = get_model_context_limit("gpt-4")
        gpt35_limit = get_model_context_limit("gpt-3.5-turbo")
        print(f"✅ Context limits: GPT-4 = {gpt4_limit}, GPT-3.5 = {gpt35_limit}")
        
        # Test within limit check
        short_text = "Short text"
        is_within = is_within_context_limit(short_text, "gpt-4")
        print(f"✅ Within limit check: '{short_text}' within GPT-4 limit = {is_within}")
        
        # Test truncation
        long_text = "This is a very long text that will be truncated. " * 100
        original_tokens = count_tokens(long_text)
        truncated = truncate_to_limit(long_text, "gpt-4", reserve_tokens=100)
        truncated_tokens = count_tokens(truncated)
        print(f"✅ Truncation: {original_tokens} -> {truncated_tokens} tokens")
        
        print("\n🎉 All tokenizer tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure tiktoken is installed: pip install tiktoken")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_contract_integration():
    """Test tokenizer integration with contracts."""
    try:
        from llm_contracts.contracts.base import PromptLengthContract
        from llm_contracts.core.interfaces import ValidationResult
        
        print("\n--- Testing Contract Integration ---")
        
        # Create a length contract
        contract = PromptLengthContract(max_tokens=100)
        
        # Test with short text (should pass)
        short_text = "This is a short message."
        result = contract.validate(short_text)
        print(f"✅ Short text validation: {result.is_valid} - {result.message}")
        
        # Test with long text (should fail)
        long_text = "This is a very long message that exceeds the token limit. " * 10
        result = contract.validate(long_text)
        print(f"✅ Long text validation: {result.is_valid} - {result.message}")
        
        # Test with messages format
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        result = contract.validate(messages)
        print(f"✅ Messages validation: {result.is_valid} - {result.message}")
        
        print("🎉 Contract integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Contract integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenAI tiktoken-based tokenizer...\n")
    
    success1 = test_tokenizer()
    success2 = test_contract_integration()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The tokenizer is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)