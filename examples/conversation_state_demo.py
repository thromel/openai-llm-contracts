"""
Demonstration of Advanced Conversation State Management

This example showcases the comprehensive conversation state management system including:
1. Conversation state tracking across multiple turns
2. Temporal contract enforcement (ALWAYS, EVENTUALLY, NEXT, etc.)
3. Context window management with intelligent compression
4. Conversation invariants (personality, facts, tone, topics)
5. Sophisticated memory system (semantic, episodic, working memory)
6. Real-time state transitions and snapshots
"""

import asyncio
import time
from typing import List, Dict, Any

# Import conversation state management components
from llm_contracts.conversation.state_manager import (
    ConversationStateManager, TurnRole, ConversationPhase
)
from llm_contracts.conversation.temporal_contracts import (
    TemporalValidator, AlwaysContract, EventuallyContract, NextContract,
    WithinContract, create_politeness_contract, create_question_answer_contract
)
from llm_contracts.conversation.context_manager import (
    ContextWindowManager, ContextCompressionStrategy, ContextPriority
)
from llm_contracts.conversation.conversation_invariants import (
    InvariantTracker, PersonalityInvariant, FactualConsistencyInvariant,
    ToneInvariant, TopicBoundaryInvariant,
    create_personality_invariant, create_topic_boundary_invariant
)
from llm_contracts.conversation.conversation_memory import (
    ConversationMemory, MemoryType, MemoryPriority, create_memory_system
)


class ConversationDemo:
    """Demo class for conversation state management."""
    
    def __init__(self):
        # Initialize core state manager
        self.state_manager = ConversationStateManager(
            conversation_id="demo_conversation",
            context_window_size=2048,
            auto_snapshot_interval=5
        )
        
        # Initialize context manager
        self.context_manager = ContextWindowManager(
            max_tokens=2048,
            compression_strategy=ContextCompressionStrategy.ADAPTIVE,
            auto_optimize=True
        )
        
        # Initialize memory system
        self.memory_system = create_memory_system(
            max_memories=5000,
            working_capacity=7,
            auto_cleanup=True
        )
        
        # Initialize temporal validator
        self.temporal_validator = TemporalValidator()
        
        # Initialize invariant tracker
        self.invariant_tracker = InvariantTracker()
        
        # Register components with state manager
        self.state_manager.set_context_manager(self.context_manager)
        self.state_manager.set_memory_store(self.memory_system)
        
        # Set up event handlers
        self.state_manager.add_state_change_handler(self._on_state_change)
        self.state_manager.add_violation_handler(self._on_violation)
        
        # Demo statistics
        self.demo_stats = {
            "turns_processed": 0,
            "violations_detected": 0,
            "auto_fixes_applied": 0,
            "context_optimizations": 0,
            "memories_created": 0,
        }
    
    def setup_contracts_and_invariants(self):
        """Set up temporal contracts and conversation invariants."""
        print("🔧 Setting up temporal contracts and invariants...")
        
        # Temporal contracts
        politeness_contract = create_politeness_contract()
        self.temporal_validator.add_contract(politeness_contract)
        
        qa_contract = create_question_answer_contract()
        self.temporal_validator.add_contract(qa_contract)
        
        # Add custom temporal contracts
        greeting_contract = EventuallyContract(
            "greeting_eventually",
            condition=lambda turn, state: any(word in turn.content.lower() 
                                           for word in ["hello", "hi", "hey", "good morning"]),
            window_size=3,
            description="Should greet within first few turns"
        )
        self.temporal_validator.add_contract(greeting_contract)
        
        topic_acknowledgment = WithinContract(
            "topic_acknowledgment",
            condition=lambda turn, state: "understand" in turn.content.lower() 
                                        or "got it" in turn.content.lower(),
            window_size=5,
            description="Should acknowledge understanding within 5 turns"
        )
        self.temporal_validator.add_contract(topic_acknowledgment)
        
        # Register temporal validator
        self.state_manager.register_temporal_contract(self.temporal_validator)
        
        # Conversation invariants
        personality_invariant = create_personality_invariant("helpful")
        self.invariant_tracker.add_invariant(personality_invariant)
        
        factual_invariant = FactualConsistencyInvariant()
        self.invariant_tracker.add_invariant(factual_invariant)
        
        tone_invariant = ToneInvariant(
            target_tone="professional",
            tone_indicators={
                "professional": ["professional", "formal", "business", "courteous"],
                "casual": ["casual", "relaxed", "chill", "cool"],
                "friendly": ["friendly", "warm", "nice", "pleasant"]
            }
        )
        self.invariant_tracker.add_invariant(tone_invariant)
        
        topic_invariant = create_topic_boundary_invariant(
            allowed_topics=["programming", "technology", "software", "AI", "machine learning"],
            forbidden_topics=["politics", "religion", "personal information"]
        )
        self.invariant_tracker.add_invariant(topic_invariant)
        
        # Register invariant tracker
        self.state_manager.register_invariant_tracker(self.invariant_tracker)
        
        print(f"✅ Set up {len(self.temporal_validator.contracts)} temporal contracts")
        print(f"✅ Set up {len(self.invariant_tracker.invariants)} conversation invariants")
    
    async def simulate_conversation(self):
        """Simulate a comprehensive conversation."""
        print("\n🎭 Starting conversation simulation...")
        
        # Define conversation scenario
        conversation_turns = [
            # Opening
            ("user", "Hello! I'm interested in learning about programming."),
            ("assistant", "Hi there! I'd be happy to help you learn programming. What specific programming language or concept would you like to explore?"),
            
            # Topic development
            ("user", "I've heard Python is good for beginners. Is that true?"),
            ("assistant", "Yes, Python is excellent for beginners! It has simple, readable syntax and is very versatile. Python is used for web development, data science, AI, and more."),
            
            # Fact establishment
            ("user", "What makes Python different from other languages?"),
            ("assistant", "Python emphasizes readability and simplicity. It uses indentation for code blocks instead of braces, has dynamic typing, and comes with a large standard library. It's also interpreted rather than compiled."),
            
            # Question-answer pattern
            ("user", "How long does it typically take to learn Python?"),
            ("assistant", "I understand your eagerness to learn! With consistent practice, you can grasp Python basics in 2-3 months. However, becoming proficient takes 6-12 months of regular coding."),
            
            # Memory and context test
            ("user", "You mentioned Python is interpreted. Can you explain what that means?"),
            ("assistant", "Certainly! When I said Python is interpreted, I meant that Python code is executed line by line by an interpreter program, rather than being compiled into machine code first. This makes development faster but execution slightly slower."),
            
            # Potential violation (to test invariants)
            ("user", "What do you think about the current political situation?"),
            ("assistant", "I understand you're curious, but I'd prefer to keep our conversation focused on programming topics. Is there anything else about Python or programming in general you'd like to explore?"),
            
            # Return to topic
            ("user", "That's fine. Can you recommend some good Python resources?"),
            ("assistant", "Absolutely! For beginners, I recommend starting with the official Python tutorial at python.org, then trying interactive platforms like Codecademy or Python.org's own beginner's guide. Books like 'Automate the Boring Stuff with Python' are also excellent."),
            
            # Wrap up
            ("user", "Thank you! This has been very helpful."),
            ("assistant", "You're very welcome! I'm glad I could help you get started with Python. Feel free to ask if you have more questions as you begin your programming journey. Good luck!"),
        ]
        
        # Process each turn
        for i, (role, content) in enumerate(conversation_turns):
            await self._process_turn(role, content, i + 1)
            await asyncio.sleep(0.1)  # Small delay for realism
        
        print(f"\n✅ Conversation completed: {len(conversation_turns)} turns processed")
    
    async def _process_turn(self, role: str, content: str, turn_number: int):
        """Process a single conversation turn."""
        print(f"\n--- Turn {turn_number} ({role.upper()}) ---")
        print(f"💬 {content}")
        
        # Add turn to state manager
        turn = self.state_manager.add_turn(role, content, {"turn_number": turn_number})
        
        # Process through memory system
        memory_results = self.memory_system.process_turn(turn, {
            "topic": "programming" if "python" in content.lower() or "programming" in content.lower() else "general",
            "turn_number": turn_number
        })
        
        # Add to context manager
        context_added = self.context_manager.add_turn(turn)
        
        # Validate with temporal contracts
        temporal_results = self.temporal_validator.validate_turn(turn, self.state_manager.state)
        
        # Check invariants
        invariant_violations = self.invariant_tracker.check_turn(turn, self.state_manager.state)
        
        # Update demo statistics
        self.demo_stats["turns_processed"] += 1
        if memory_results["facts_extracted"]:
            self.demo_stats["memories_created"] += len(memory_results["facts_extracted"])
        if memory_results["episode_stored"]:
            self.demo_stats["memories_created"] += 1
        
        # Report on processing
        print(f"📊 Processing results:")
        print(f"   • Context added: {'✅' if context_added else '❌'}")
        print(f"   • Facts extracted: {len(memory_results.get('facts_extracted', []))}")
        print(f"   • Episode stored: {'✅' if memory_results.get('episode_stored') else '❌'}")
        print(f"   • Temporal contracts: {len([r for r in temporal_results if r.is_valid])} valid, {len([r for r in temporal_results if not r.is_valid])} violated")
        print(f"   • Invariant violations: {len(invariant_violations)}")
        
        # Handle violations
        if invariant_violations:
            self.demo_stats["violations_detected"] += len(invariant_violations)
            for violation in invariant_violations:
                print(f"   ⚠️  Invariant violation: {violation.violation_message}")
                if violation.auto_fix_suggestion:
                    print(f"   💡 Auto-fix suggestion: {violation.auto_fix_suggestion}")
                    self.demo_stats["auto_fixes_applied"] += 1
        
        temporal_violations = [r for r in temporal_results if not r.is_valid]
        if temporal_violations:
            for violation in temporal_violations:
                print(f"   ⚠️  Temporal violation: {violation.message}")
    
    def _on_state_change(self, transition):
        """Handle state change events."""
        print(f"🔄 State change: {transition.change_type.name} - {transition.trigger}")
    
    def _on_violation(self, contract_name: str, violation_msg: str):
        """Handle violation events."""
        print(f"⚠️  Violation in {contract_name}: {violation_msg}")
        self.demo_stats["violations_detected"] += 1
    
    def display_comprehensive_report(self):
        """Display comprehensive conversation analysis."""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE CONVERSATION ANALYSIS")
        print("=" * 80)
        
        # Basic conversation metrics
        print("\n📈 CONVERSATION METRICS:")
        state_metrics = self.state_manager.get_metrics()
        print(f"   • Total turns: {state_metrics['turn_count']}")
        print(f"   • Conversation phase: {state_metrics['conversation_phase']}")
        print(f"   • Total tokens: {state_metrics['total_tokens']}")
        print(f"   • Active context tokens: {state_metrics['active_context_tokens']}")
        print(f"   • Snapshots created: {state_metrics['snapshots_created']}")
        
        # Context management metrics
        print("\n🧠 CONTEXT MANAGEMENT:")
        context_metrics = self.context_manager.get_metrics()
        print(f"   • Context utilization: {context_metrics['context_window']['utilization_percent']:.1f}%")
        print(f"   • Elements in context: {context_metrics['context_window']['element_count']}")
        print(f"   • Compression applied: {context_metrics['context_window']['compression_applied']}")
        print(f"   • Optimizations triggered: {context_metrics['management_metrics']['optimizations_triggered']}")
        
        # Memory system metrics
        print("\n🧮 MEMORY SYSTEM:")
        memory_summary = self.memory_system.get_memory_summary()
        print(f"   • Memories created: {memory_summary['metrics']['memories_created']}")
        print(f"   • Facts extracted: {memory_summary['metrics']['facts_extracted']}")
        print(f"   • Episodes stored: {memory_summary['metrics']['episodes_stored']}")
        print(f"   • Working memory items: {memory_summary['working_memory']['active_items']}")
        print(f"   • Cleanups performed: {memory_summary['metrics']['cleanups_performed']}")
        
        # Temporal contracts analysis
        print("\n⏰ TEMPORAL CONTRACTS:")
        contract_status = self.temporal_validator.get_contract_status()
        for name, status in contract_status.items():
            status_emoji = "✅" if status['status'] == 'satisfied' else "❌"
            print(f"   {status_emoji} {name}: {status['evaluations']} evaluations, {status['violations']} violations")
        
        # Invariants analysis
        print("\n🔒 CONVERSATION INVARIANTS:")
        invariant_status = self.invariant_tracker.get_invariant_status()
        for name, status in invariant_status.items():
            violation_rate = status.get('violation_rate', 0) * 100
            health_emoji = "✅" if violation_rate < 10 else "⚠️" if violation_rate < 30 else "❌"
            print(f"   {health_emoji} {name}: {status['check_count']} checks, {violation_rate:.1f}% violation rate")
        
        # Demo statistics
        print("\n📊 DEMO STATISTICS:")
        for key, value in self.demo_stats.items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        # Conversation history sample
        print("\n📜 CONVERSATION HISTORY (Sample):")
        history = self.state_manager.get_conversation_history()
        for i, msg in enumerate(history[:3]):  # Show first 3 messages
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            content_preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
            print(f"   {role_emoji} {msg['role'].title()}: {content_preview}")
        
        if len(history) > 3:
            print(f"   ... and {len(history) - 3} more messages")
        
        print("\n✨ Analysis complete!")


async def demo_advanced_features():
    """Demonstrate advanced conversation state management features."""
    print("\n🚀 ADVANCED FEATURES DEMO")
    print("=" * 50)
    
    demo = ConversationDemo()
    
    # Set up contracts and invariants
    demo.setup_contracts_and_invariants()
    
    # Simulate conversation
    await demo.simulate_conversation()
    
    # Display comprehensive analysis
    demo.display_comprehensive_report()


def demo_context_compression():
    """Demonstrate context compression strategies."""
    print("\n🗜️  CONTEXT COMPRESSION DEMO")
    print("=" * 40)
    
    from llm_contracts.conversation.context_manager import (
        ContextElement, ContextOptimizer, ContextCompressionStrategy, ContextPriority
    )
    
    # Create sample context elements
    elements = []
    for i in range(15):
        priority = ContextPriority.HIGH if i < 3 else ContextPriority.MEDIUM if i < 8 else ContextPriority.LOW
        element = ContextElement(
            element_id=f"elem_{i}",
            content=f"This is message {i} with varying importance and length. " * (3 if priority == ContextPriority.HIGH else 2 if priority == ContextPriority.MEDIUM else 1),
            turn_id=f"turn_{i}",
            role="user" if i % 2 == 0 else "assistant",
            timestamp=time.time() - (15 - i) * 60,  # Simulate time progression
            token_count=50 * (3 if priority == ContextPriority.HIGH else 2 if priority == ContextPriority.MEDIUM else 1),
            priority=priority,
            importance_score=1.0 - (i * 0.05)  # Decreasing importance
        )
        elements.append(element)
    
    original_tokens = sum(e.effective_tokens for e in elements)
    print(f"📊 Original context: {len(elements)} elements, {original_tokens} tokens")
    
    # Test different compression strategies
    optimizer = ContextOptimizer()
    strategies = [
        ContextCompressionStrategy.TRUNCATE_OLDEST,
        ContextCompressionStrategy.TRUNCATE_MIDDLE,
        ContextCompressionStrategy.SEMANTIC_COMPRESSION,
        ContextCompressionStrategy.ADAPTIVE
    ]
    
    for strategy in strategies:
        optimized, record = optimizer.optimize_context(elements, max_tokens=400, strategy=strategy)
        compression_ratio = record['compression_ratio']
        elements_kept = len(optimized)
        
        print(f"   📈 {strategy.value}:")
        print(f"      • Elements kept: {elements_kept}/{len(elements)}")
        print(f"      • Compression ratio: {compression_ratio:.2f}")
        print(f"      • Final tokens: {record['final_tokens']}")
        print(f"      • Importance preserved: {record['importance_preservation']:.2f}")
    
    # Show performance summary
    performance = optimizer.get_performance_summary()
    print(f"\n⚡ Performance summary:")
    print(f"   • Optimizations performed: {performance['optimizations_performed']}")
    print(f"   • Average compression time: {performance['avg_compression_time_ms']:.1f}ms")
    print(f"   • Average importance preservation: {performance['avg_importance_preservation']:.2f}")


def demo_memory_retrieval():
    """Demonstrate memory retrieval capabilities."""
    print("\n🧠 MEMORY RETRIEVAL DEMO")
    print("=" * 35)
    
    # Create memory system
    memory_system = create_memory_system()
    
    # Add sample memories
    sample_facts = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning requires data",
        "Algorithms solve problems efficiently",
        "Databases store information",
        "APIs connect different systems",
        "Version control tracks code changes",
        "Testing ensures code quality",
    ]
    
    print("📝 Adding sample memories...")
    
    # Create mock turns for each fact
    for i, fact in enumerate(sample_facts):
        mock_turn = type('MockTurn', (), {
            'turn_id': f'turn_{i}',
            'timestamp': time.time() - (len(sample_facts) - i) * 100,
            'role': type('MockRole', (), {'value': 'assistant'}),
            'content': fact,
            'metadata': {}
        })()
        
        memory_system.process_turn(mock_turn, {'topic': 'programming'})
    
    # Test different retrieval strategies
    query = "programming language"
    strategies = ["recency", "relevance", "importance", "hybrid"]
    
    print(f"\n🔍 Testing retrieval for query: '{query}'")
    
    for strategy in strategies:
        memories = memory_system.get_relevant_memories(query, strategy=strategy, limit=3)
        print(f"\n   📋 {strategy.title()} strategy:")
        for i, memory in enumerate(memories[:3], 1):
            content_preview = memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
            print(f"      {i}. {content_preview} (strength: {memory.get_current_strength():.2f})")
    
    # Show memory summary
    summary = memory_system.get_memory_summary()
    print(f"\n📊 Memory system summary:")
    print(f"   • Total memories: {summary['metrics']['memories_created']}")
    print(f"   • Facts extracted: {summary['metrics']['facts_extracted']}")
    print(f"   • Episodes stored: {summary['metrics']['episodes_stored']}")


async def run_complete_demo():
    """Run the complete conversation state management demo."""
    print("🎯 LLM CONVERSATION STATE MANAGEMENT DEMO")
    print("=" * 60)
    print("This demo showcases advanced conversation state management including:")
    print("• Real-time state tracking and transitions")
    print("• Temporal contract enforcement")
    print("• Intelligent context window management")
    print("• Conversation invariants validation")
    print("• Sophisticated memory systems")
    print("• Performance optimization")
    print()
    
    try:
        # Run main conversation demo
        await demo_advanced_features()
        
        # Run additional feature demos
        demo_context_compression()
        demo_memory_retrieval()
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("✅ Maintains conversation context across multiple turns")
        print("✅ Enforces behavioral contracts and invariants")
        print("✅ Optimizes memory usage with intelligent compression")
        print("✅ Provides rich analytics and state tracking")
        print("✅ Scales efficiently for long conversations")
        print("✅ Enables sophisticated LLM applications")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_complete_demo())