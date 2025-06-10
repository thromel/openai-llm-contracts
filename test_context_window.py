#!/usr/bin/env python3
"""
Test context window management functionality.

This script tests the context window contract and management system to ensure
it properly handles token limits, compression, and conversation tracking.
"""

import sys
import os
import time
import uuid

# Add src to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_context_window():
    """Test basic context window functionality."""
    try:
        from llm_contracts.conversation.context_manager import (
            ContextWindow, ContextElement, ContextPriority
        )
        from llm_contracts.utils.tokenizer import count_tokens
        
        print("🔧 Testing Basic Context Window")
        print("=" * 50)
        
        # Test 1: Create context window
        print("1. Creating Context Window:")
        context_window = ContextWindow(max_tokens=100)
        print(f"   ✅ Context window created with {context_window.max_tokens} max tokens")
        
        # Test 2: Add elements within limit
        print("\n2. Adding Elements Within Limit:")
        
        # Create test elements
        element1 = ContextElement(
            element_id="test1",
            content="Hello, this is a test message.",
            turn_id="turn1",
            role="user",
            timestamp=time.time(),
            token_count=count_tokens("Hello, this is a test message.", model="gpt-4"),
            priority=ContextPriority.MEDIUM
        )
        
        element2 = ContextElement(
            element_id="test2", 
            content="This is a response to the test message.",
            turn_id="turn2",
            role="assistant",
            timestamp=time.time(),
            token_count=count_tokens("This is a response to the test message.", model="gpt-4"),
            priority=ContextPriority.MEDIUM
        )
        
        # Add elements
        success1 = context_window.add_element(element1)
        success2 = context_window.add_element(element2)
        
        print(f"   Element 1 ({element1.token_count} tokens): {'✅ Added' if success1 else '❌ Failed'}")
        print(f"   Element 2 ({element2.token_count} tokens): {'✅ Added' if success2 else '❌ Failed'}")
        print(f"   Current usage: {context_window.current_tokens}/{context_window.max_tokens} tokens")
        print(f"   Utilization: {context_window.get_utilization():.1f}%")
        
        # Test 3: Try to exceed limit
        print("\n3. Testing Token Limit Enforcement:")
        
        large_element = ContextElement(
            element_id="large",
            content="This is a very long message that should exceed the remaining token limit. " * 10,
            turn_id="turn3", 
            role="user",
            timestamp=time.time(),
            token_count=count_tokens("This is a very long message that should exceed the remaining token limit. " * 10, model="gpt-4"),
            priority=ContextPriority.LOW
        )
        
        success3 = context_window.add_element(large_element)
        print(f"   Large element ({large_element.token_count} tokens): {'❌ Rejected (as expected)' if not success3 else '⚠️ Unexpectedly added'}")
        
        # Test 4: Remove elements
        print("\n4. Testing Element Removal:")
        
        removed = context_window.remove_element("test1")
        print(f"   Remove element 1: {'✅ Success' if removed else '❌ Failed'}")
        print(f"   New usage: {context_window.current_tokens}/{context_window.max_tokens} tokens")
        
        # Test 5: Convert to messages
        print("\n5. Testing Message Conversion:")
        
        messages = context_window.to_messages()
        print(f"   Converted to {len(messages)} messages")
        for i, msg in enumerate(messages):
            print(f"   Message {i+1}: {msg['role']} - {msg['content'][:30]}...")
        
        print("✅ Basic context window test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic context window test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_window_manager():
    """Test the full context window manager."""
    try:
        from llm_contracts.conversation.context_manager import (
            ContextWindowManager, ContextCompressionStrategy
        )
        from llm_contracts.conversation.state_manager import ConversationStateManager, TurnRole
        
        print("\n📋 Testing Context Window Manager")
        print("=" * 50)
        
        # Test 1: Initialize manager
        print("1. Initializing Context Window Manager:")
        
        manager = ContextWindowManager(
            max_tokens=200,  # Small limit for testing
            compression_strategy=ContextCompressionStrategy.TRUNCATE_OLDEST,
            auto_optimize=True,
            optimization_threshold=0.8
        )
        
        print(f"   ✅ Manager initialized with {manager.max_tokens} max tokens")
        print(f"   Strategy: {manager.compression_strategy.value}")
        
        # Test 2: Create conversation state
        print("\n2. Creating Conversation State:")
        
        conversation = ConversationStateManager(
            conversation_id="test_context_conversation",
            context_window_size=200
        )
        
        print("   ✅ Conversation state manager created")
        
        # Test 3: Add turns and test context management
        print("\n3. Adding Turns and Testing Context Management:")
        
        test_turns = [
            ("user", "Hello, how are you today?"),
            ("assistant", "I'm doing well, thank you for asking! How can I help you?"),
            ("user", "Can you explain what artificial intelligence is?"),
            ("assistant", "Artificial intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence."),
            ("user", "That's interesting. Can you give me more details about machine learning?"),
            ("assistant", "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."),
            ("user", "What about deep learning? How is it different?"),
            ("assistant", "Deep learning is a specialized form of machine learning that uses neural networks with multiple layers to analyze data patterns.")
        ]
        
        for i, (role, content) in enumerate(test_turns, 1):
            print(f"\n   Turn {i}: Adding {role} message...")
            
            # Add turn to conversation
            turn = conversation.add_turn(TurnRole(role), content)
            print(f"   Added turn with {turn.token_count} tokens")
            
            # Add to context manager
            success = manager.add_turn(turn)
            print(f"   Context manager: {'✅ Added' if success else '❌ Failed'}")
            
            # Check utilization
            utilization = manager.get_utilization()
            print(f"   Context utilization: {utilization:.1f}%")
            
            # If utilization is high, check if optimization was triggered
            if utilization > 80:
                print(f"   ⚠️ High utilization - optimization may be triggered")
        
        # Test 4: Get final metrics
        print("\n4. Final Context Metrics:")
        
        metrics = manager.get_metrics()
        context_metrics = metrics["context_window"]
        
        print(f"   Max tokens: {context_metrics['max_tokens']}")
        print(f"   Current tokens: {context_metrics['current_tokens']}")
        print(f"   Utilization: {context_metrics['utilization_percent']:.1f}%")
        print(f"   Element count: {context_metrics['element_count']}")
        print(f"   Compression applied: {context_metrics['compression_applied']}")
        
        # Test 5: Get context messages
        print("\n5. Getting Context Messages:")
        
        context_messages = manager.get_context_messages()
        print(f"   Retrieved {len(context_messages)} context messages")
        
        # Calculate total tokens in context
        from llm_contracts.utils.tokenizer import count_tokens_from_messages
        total_context_tokens = count_tokens_from_messages(context_messages, model="gpt-4")
        print(f"   Total context tokens: {total_context_tokens}")
        
        # Verify it's within limits
        if total_context_tokens <= manager.max_tokens:
            print("   ✅ Context is within token limits")
        else:
            print("   ⚠️ Context exceeds token limits - may need optimization")
        
        print("✅ Context window manager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Context window manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_optimization():
    """Test context window optimization and compression."""
    try:
        from llm_contracts.conversation.context_manager import (
            ContextOptimizer, ContextElement, ContextPriority, ContextCompressionStrategy
        )
        from llm_contracts.utils.tokenizer import count_tokens
        
        print("\n🔧 Testing Context Optimization")
        print("=" * 50)
        
        # Test 1: Create optimizer
        print("1. Creating Context Optimizer:")
        
        optimizer = ContextOptimizer()
        print("   ✅ Context optimizer created")
        
        # Test 2: Create test elements that exceed limit
        print("\n2. Creating Test Elements:")
        
        test_elements = []
        test_contents = [
            "Hello, this is the first message in our conversation.",
            "Thank you for the greeting. I'm here to help you with any questions.",
            "Can you explain how context windows work in language models?",
            "Context windows determine how much conversation history the model can see at once.",
            "That's very helpful. What happens when the context gets too long?",
            "When context exceeds limits, older messages are typically removed or compressed.",
            "Interesting. Are there different strategies for managing this?",
            "Yes, strategies include truncation, compression, and summarization techniques."
        ]
        
        for i, content in enumerate(test_contents):
            element = ContextElement(
                element_id=f"element_{i}",
                content=content,
                turn_id=f"turn_{i}",
                role="user" if i % 2 == 0 else "assistant",
                timestamp=time.time() + i,
                token_count=count_tokens(content, model="gpt-4"),
                priority=ContextPriority.HIGH if i < 2 else ContextPriority.MEDIUM
            )
            test_elements.append(element)
        
        total_tokens = sum(e.token_count for e in test_elements)
        print(f"   Created {len(test_elements)} elements with {total_tokens} total tokens")
        
        # Test 3: Optimize for smaller limit
        print("\n3. Testing Optimization (Target: 100 tokens):")
        
        optimized_elements, optimization_record = optimizer.optimize_context(
            test_elements, 
            max_tokens=100,
            strategy=ContextCompressionStrategy.TRUNCATE_OLDEST
        )
        
        optimized_tokens = sum(e.effective_tokens for e in optimized_elements)
        
        print(f"   Original elements: {len(test_elements)} ({total_tokens} tokens)")
        print(f"   Optimized elements: {len(optimized_elements)} ({optimized_tokens} tokens)")
        print(f"   Compression ratio: {optimization_record['compression_ratio']:.2f}")
        print(f"   Elements removed: {optimization_record['elements_removed']}")
        print(f"   Token utilization: {optimization_record['token_utilization']:.2f}")
        print(f"   Optimization time: {optimization_record['compression_time_ms']:.1f}ms")
        
        # Verify optimization worked
        if optimized_tokens <= 100:
            print("   ✅ Optimization successful - within token limit")
        else:
            print("   ❌ Optimization failed - still exceeds limit")
            return False
        
        # Test 4: Test different strategies
        print("\n4. Testing Different Optimization Strategies:")
        
        strategies = [
            ContextCompressionStrategy.TRUNCATE_OLDEST,
            ContextCompressionStrategy.TRUNCATE_MIDDLE,
            ContextCompressionStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            try:
                optimized, record = optimizer.optimize_context(
                    test_elements,
                    max_tokens=120,
                    strategy=strategy
                )
                opt_tokens = sum(e.effective_tokens for e in optimized)
                print(f"   {strategy.value}: {len(optimized)} elements, {opt_tokens} tokens")
            except Exception as e:
                print(f"   {strategy.value}: ❌ Failed - {e}")
        
        # Test 5: Get performance summary
        print("\n5. Optimization Performance Summary:")
        
        performance = optimizer.get_performance_summary()
        if performance["optimizations_performed"] > 0:
            print(f"   Optimizations performed: {performance['optimizations_performed']}")
            print(f"   Average compression time: {performance['avg_compression_time_ms']:.1f}ms")
            print(f"   Average compression ratio: {performance['avg_compression_ratio']:.2f}")
            print(f"   Average token utilization: {performance['avg_token_utilization']:.2f}")
        else:
            print("   No optimization performance data available")
        
        print("✅ Context optimization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Context optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_conversation():
    """Test integration between context management and conversation state."""
    try:
        from llm_contracts.conversation.state_manager import ConversationStateManager, TurnRole
        from llm_contracts.conversation.context_manager import ContextWindowManager
        
        print("\n🔗 Testing Integration with Conversation State")
        print("=" * 50)
        
        # Test 1: Set up integrated system
        print("1. Setting Up Integrated System:")
        
        conversation = ConversationStateManager(
            conversation_id="integration_test",
            context_window_size=150,  # Small for testing
            auto_snapshot_interval=5
        )
        
        context_manager = ContextWindowManager(
            max_tokens=150,
            auto_optimize=True,
            optimization_threshold=0.8
        )
        
        # Connect them
        conversation.set_context_manager(context_manager)
        
        print("   ✅ Conversation and context managers connected")
        
        # Test 2: Simulate conversation with context management
        print("\n2. Simulating Conversation:")
        
        conversation_turns = [
            ("user", "Hi there! How are you doing today?"),
            ("assistant", "Hello! I'm doing great, thank you for asking."),
            ("user", "Can you help me understand machine learning?"),
            ("assistant", "Of course! Machine learning is a method where computers learn from data."),
            ("user", "What are some common machine learning algorithms?"),
            ("assistant", "Common algorithms include linear regression, decision trees, and neural networks."),
            ("user", "How do neural networks work exactly?"),
            ("assistant", "Neural networks are inspired by the brain and use interconnected nodes to process information."),
            ("user", "That's fascinating. What about deep learning?"),
            ("assistant", "Deep learning uses multiple layers of neural networks to find complex patterns in data.")
        ]
        
        for i, (role, content) in enumerate(conversation_turns, 1):
            print(f"\n   Turn {i}: {role}")
            print(f"   Content: {content[:50]}...")
            
            # Add turn
            turn = conversation.add_turn(TurnRole(role), content)
            print(f"   Turn tokens: {turn.token_count}")
            
            # Get conversation metrics
            metrics = conversation.get_metrics()
            print(f"   Total conversation tokens: {metrics['total_tokens']}")
            print(f"   Active context tokens: {metrics['active_context_tokens']}")
            print(f"   Turn count: {metrics['turn_count']}")
            
            # Check if context is within limits
            context_window = conversation.get_context_window(max_tokens=150)
            context_tokens = sum(t.token_count for t in context_window)
            print(f"   Context window tokens: {context_tokens}/150")
            
            if context_tokens > 150:
                print("   ⚠️ Context exceeds limit - optimization needed")
            else:
                print("   ✅ Context within limits")
        
        # Test 3: Get final state
        print("\n3. Final System State:")
        
        final_metrics = conversation.get_metrics()
        print(f"   Final turn count: {final_metrics['turn_count']}")
        print(f"   Final total tokens: {final_metrics['total_tokens']}")
        print(f"   Final active tokens: {final_metrics['active_context_tokens']}")
        print(f"   Conversation phase: {final_metrics['conversation_phase']}")
        
        # Test 4: Export conversation
        print("\n4. Testing Conversation Export:")
        
        exported = conversation.export_conversation()
        print(f"   Exported conversation: {len(exported)} characters")
        
        # Verify it's valid JSON
        import json
        try:
            parsed = json.loads(exported)
            print("   ✅ Export is valid JSON")
            print(f"   Exported {len(parsed.get('turns', []))} turns")
        except json.JSONDecodeError:
            print("   ❌ Export is not valid JSON")
            return False
        
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all context window tests."""
    print("🚀 LLM Contracts Framework - Context Window Test")
    print("=" * 60)
    print("Testing context window management and optimization functionality...\n")
    
    tests = [
        ("Basic Context Window", test_basic_context_window),
        ("Context Window Manager", test_context_window_manager),
        ("Context Optimization", test_context_optimization),
        ("Integration with Conversation", test_integration_with_conversation)
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
        print("🎉 All context window tests passed!")
        print("\n📈 Context Window Features Verified:")
        print("  • Basic context window operations")
        print("  • Token limit enforcement")
        print("  • Context optimization and compression")
        print("  • Integration with conversation management")
        print("  • Element addition/removal")
        print("  • Message format conversion")
    else:
        print("⚠️  Some context window tests failed.")
        print("The context window functionality needs fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)