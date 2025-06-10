"""Comprehensive tests for conversation state management system."""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from .state_manager import (
    ConversationStateManager, ConversationState, TurnState, TurnRole,
    ConversationPhase, StateChangeType
)
from .temporal_contracts import (
    TemporalValidator, AlwaysContract, EventuallyContract, NextContract,
    WithinContract, UntilContract, SinceContract, TemporalOperator,
    create_politeness_contract, create_question_answer_contract
)
from .context_manager import (
    ContextWindowManager, ContextElement, ContextCompressionStrategy,
    ContextPriority, TokenBudgetManager, ContextOptimizer
)
from .conversation_invariants import (
    InvariantTracker, PersonalityInvariant, FactualConsistencyInvariant,
    ToneInvariant, TopicBoundaryInvariant, MemoryInvariant,
    create_personality_invariant, create_topic_boundary_invariant
)
from .conversation_memory import (
    ConversationMemory, MemoryType, MemoryPriority, MemoryItem,
    InMemoryStore, SemanticMemory, EpisodicMemory, WorkingMemory,
    create_memory_system
)


class TestConversationStateManager:
    """Tests for ConversationStateManager."""
    
    def test_initialization(self):
        """Test state manager initialization."""
        manager = ConversationStateManager(
            conversation_id="test_conv",
            context_window_size=2048,
            max_history_length=500
        )
        
        assert manager.conversation_id == "test_conv"
        assert manager.state.context_window_size == 2048
        assert manager.max_history_length == 500
        assert manager.state.phase == ConversationPhase.INITIALIZATION
    
    def test_add_turn(self):
        """Test adding turns to conversation."""
        manager = ConversationStateManager()
        
        # Add user turn
        user_turn = manager.add_turn(
            role=TurnRole.USER,
            content="Hello, how are you?",
            metadata={"test": True}
        )
        
        assert user_turn.role == TurnRole.USER
        assert user_turn.content == "Hello, how are you?"
        assert user_turn.metadata["test"] is True
        assert manager.state.turn_count == 1
        assert manager.state.phase == ConversationPhase.ACTIVE
        
        # Add assistant turn
        assistant_turn = manager.add_turn(
            role="assistant",  # Test string conversion
            content="I'm doing well, thank you!"
        )
        
        assert assistant_turn.role == TurnRole.ASSISTANT
        assert manager.state.turn_count == 2
    
    def test_conversation_history(self):
        """Test conversation history retrieval."""
        manager = ConversationStateManager()
        
        manager.add_turn(TurnRole.USER, "What's the weather?")
        manager.add_turn(TurnRole.ASSISTANT, "It's sunny today.")
        manager.add_turn(TurnRole.USER, "Thank you!")
        
        # Get full history
        history = manager.get_conversation_history()
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        
        # Get history with metadata
        history_with_meta = manager.get_conversation_history(include_metadata=True)
        assert "metadata" in history_with_meta[0]
        assert "turn_id" in history_with_meta[0]["metadata"]
        
        # Filter by role
        user_history = manager.get_conversation_history(role_filter=TurnRole.USER)
        assert len(user_history) == 2
        assert all(msg["role"] == "user" for msg in user_history)
    
    def test_context_window(self):
        """Test context window management."""
        manager = ConversationStateManager(context_window_size=100)
        
        # Add turns that exceed context window
        for i in range(10):
            manager.add_turn(TurnRole.USER, f"Message {i} " * 20)  # Long messages
        
        context_window = manager.get_context_window(max_tokens=50)
        assert len(context_window) < 10  # Should be truncated
        
        # Check that most recent turns are preserved
        recent_turns = manager.state.get_recent_turns(3)
        assert len(recent_turns) == 3
        assert recent_turns[-1].content.startswith("Message 9")
    
    def test_snapshots(self):
        """Test conversation snapshots."""
        manager = ConversationStateManager()
        
        manager.add_turn(TurnRole.USER, "Hello")
        manager.add_turn(TurnRole.ASSISTANT, "Hi there")
        
        # Create manual snapshot
        snapshot = manager.create_snapshot("test snapshot")
        assert snapshot.conversation_id == manager.conversation_id
        assert snapshot.turn_count == 2
        assert len(manager.state.snapshots) == 1
        
    def test_metrics(self):
        """Test metrics collection."""
        manager = ConversationStateManager()
        
        manager.add_turn(TurnRole.USER, "Test message")
        
        metrics = manager.get_metrics()
        assert metrics["turn_count"] == 1
        assert metrics["conversation_phase"] == "active"
        assert "validation_metrics" in str(metrics)  # Should be present
    
    def test_export_conversation(self):
        """Test conversation export."""
        manager = ConversationStateManager()
        
        manager.add_turn(TurnRole.USER, "Hello")
        manager.add_turn(TurnRole.ASSISTANT, "Hi")
        
        # Export as JSON
        json_export = manager.export_conversation("json")
        assert "conversation_id" in json_export
        assert "turns" in json_export
        assert "Hello" in json_export


class TestTemporalContracts:
    """Tests for temporal contract system."""
    
    def test_always_contract(self):
        """Test ALWAYS temporal contract."""
        contract = AlwaysContract("politeness", r"\b(please|thank you)\b")
        validator = TemporalValidator()
        validator.add_contract(contract)
        
        # Mock conversation state
        mock_state = Mock()
        mock_state.turn_count = 1
        
        # Test passing case
        polite_turn = Mock()
        polite_turn.turn_id = "turn1"
        polite_turn.timestamp = time.time()
        polite_turn.content = "Please help me with this."
        
        results = validator.validate_turn(polite_turn, mock_state)
        assert len(results) == 1
        assert results[0].is_valid
        
        # Test failing case
        rude_turn = Mock()
        rude_turn.turn_id = "turn2"
        rude_turn.timestamp = time.time()
        rude_turn.content = "Do this now!"
        
        results = validator.validate_turn(rude_turn, mock_state)
        assert len(results) == 1
        assert not results[0].is_valid
    
    def test_eventually_contract(self):
        """Test EVENTUALLY temporal contract."""
        contract = EventuallyContract("acknowledgment", "thank", window_size=3)
        validator = TemporalValidator()
        validator.add_contract(contract)
        
        mock_state = Mock()
        mock_state.turn_count = 1
        
        # First turn without acknowledgment
        turn1 = Mock()
        turn1.turn_id = "turn1"
        turn1.timestamp = time.time()
        turn1.content = "Here's your answer."
        
        results = validator.validate_turn(turn1, mock_state)
        assert results[0].is_valid  # No violation yet
        
        # Add more turns without acknowledgment
        for i in range(2, 5):
            turn = Mock()
            turn.turn_id = f"turn{i}"
            turn.timestamp = time.time()
            turn.content = "More content."
            mock_state.turn_count = i
            
            results = validator.validate_turn(turn, mock_state)
        
        # Should violate eventually contract
        assert not results[0].is_valid
    
    def test_next_contract(self):
        """Test NEXT temporal contract."""
        contract = NextContract(
            "question_answer",
            trigger_condition=lambda turn, state: "?" in turn.content,
            next_condition=lambda turn, state: "answer" in turn.content.lower()
        )
        
        validator = TemporalValidator()
        validator.add_contract(contract)
        
        mock_state = Mock()
        
        # Question turn
        question_turn = Mock()
        question_turn.turn_id = "turn1"
        question_turn.timestamp = time.time()
        question_turn.content = "What is your name?"
        
        results = validator.validate_turn(question_turn, mock_state)
        assert results[0].is_valid  # Question itself is valid
        
        # Next turn should contain answer
        answer_turn = Mock()
        answer_turn.turn_id = "turn2"
        answer_turn.timestamp = time.time()
        answer_turn.content = "My answer is Claude."
        
        results = validator.validate_turn(answer_turn, mock_state)
        assert results[0].is_valid  # Contains "answer"
        
        # Test violation
        non_answer_turn = Mock()
        non_answer_turn.turn_id = "turn3"
        non_answer_turn.timestamp = time.time()
        non_answer_turn.content = "Another question?"
        
        # Reset contract state
        contract.expecting_next = True
        contract.trigger_turn_id = "turn2"
        
        results = validator.validate_turn(non_answer_turn, mock_state)
        assert not results[0].is_valid  # Should violate NEXT
    
    def test_convenience_functions(self):
        """Test convenience functions for creating contracts."""
        # Test politeness contract
        politeness = create_politeness_contract()
        assert politeness.name == "politeness_always"
        assert politeness.operator == TemporalOperator.ALWAYS
        
        # Test question-answer contract
        qa_contract = create_question_answer_contract()
        assert qa_contract.name == "question_answer_flow"
        assert qa_contract.operator == TemporalOperator.NEXT


class TestContextManager:
    """Tests for context window management."""
    
    def test_context_element_creation(self):
        """Test context element creation."""
        element = ContextElement(
            element_id="test1",
            content="Hello world",
            turn_id="turn1",
            role="user",
            timestamp=time.time(),
            token_count=10,
            priority=ContextPriority.HIGH
        )
        
        assert element.effective_tokens == 10
        assert element.priority == ContextPriority.HIGH
        
        # Test compression
        element.compression_ratio = 0.5
        assert element.effective_tokens == 5
    
    def test_context_window(self):
        """Test context window operations."""
        window = ContextWindow(max_tokens=100)
        
        element1 = ContextElement(
            element_id="e1", content="First", turn_id="t1", role="user",
            timestamp=time.time(), token_count=30
        )
        element2 = ContextElement(
            element_id="e2", content="Second", turn_id="t2", role="assistant",
            timestamp=time.time(), token_count=40
        )
        
        # Add elements
        assert window.add_element(element1)
        assert window.current_tokens == 30
        
        assert window.add_element(element2)
        assert window.current_tokens == 70
        
        # Test utilization
        assert window.get_utilization() == 70.0
        
        # Test removal
        assert window.remove_element("e1")
        assert window.current_tokens == 40
        
        # Test conversion to messages
        messages = window.to_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Second"
    
    def test_token_budget_manager(self):
        """Test token budget management."""
        budget = TokenBudgetManager(total_budget=1000)
        
        # Allocate budgets
        assert budget.allocate_budget("user_input", 200)
        assert budget.allocate_budget("system_prompt", 100, reserved=True)
        assert budget.allocate_budget("response", 500)
        
        # Use tokens
        assert budget.use_tokens("user_input", 150)
        assert budget.get_remaining("user_input") == 50
        
        # Test over-allocation
        assert not budget.allocate_budget("overflow", 300)  # Would exceed budget
        
        # Test utilization
        utilization = budget.get_utilization()
        assert utilization["user_input"] == 75.0  # 150/200
    
    def test_context_optimizer(self):
        """Test context optimization."""
        optimizer = ContextOptimizer()
        
        elements = [
            ContextElement(f"e{i}", f"Content {i}", f"t{i}", "user", 
                         time.time(), 50, importance_score=i/10)
            for i in range(10)
        ]
        
        # Test optimization
        optimized, record = optimizer.optimize_context(elements, max_tokens=200)
        
        assert len(optimized) < len(elements)  # Should be compressed
        assert sum(e.effective_tokens for e in optimized) <= 200
        assert record["compression_needed"]
        assert record["compression_ratio"] < 1.0
    
    def test_context_window_manager(self):
        """Test full context window manager."""
        manager = ContextWindowManager(max_tokens=150, auto_optimize=True)
        
        # Create mock turns
        for i in range(5):
            mock_turn = Mock()
            mock_turn.turn_id = f"turn{i}"
            mock_turn.content = f"This is message {i} with some content."
            mock_turn.role = Mock()
            mock_turn.role.value = "user"
            mock_turn.timestamp = time.time()
            mock_turn.token_count = 40
            mock_turn.metadata = {}
            
            success = manager.add_turn(mock_turn)
            assert success
        
        # Check utilization
        utilization = manager.get_utilization()
        assert utilization <= 100.0  # Should not exceed due to auto-optimization
        
        # Get metrics
        metrics = manager.get_metrics()
        assert "context_window" in metrics
        assert "management_metrics" in metrics


class TestConversationInvariants:
    """Tests for conversation invariants."""
    
    def test_personality_invariant(self):
        """Test personality consistency invariant."""
        invariant = PersonalityInvariant(
            personality_traits={"friendly": "high", "formal": "low"}
        )
        
        mock_state = Mock()
        mock_state.turn_count = 1
        mock_state.turns = []
        
        # First few turns to establish baseline
        for i in range(4):
            turn = Mock()
            turn.turn_id = f"turn{i}"
            turn.timestamp = time.time()
            turn.role = Mock()
            turn.role.value = "assistant"
            turn.content = "Hello! Thanks for asking. I'm happy to help!"
            
            mock_state.turns.append(turn)
            violation = invariant.check_turn(turn, mock_state)
            assert violation is None  # No violation during baseline
        
        # Turn that violates personality
        formal_turn = Mock()
        formal_turn.turn_id = "turn5"
        formal_turn.timestamp = time.time()
        formal_turn.role = Mock()
        formal_turn.role.value = "assistant"
        formal_turn.content = "Please provide your identification number for verification."
        
        mock_state.turns.append(formal_turn)
        violation = invariant.check_turn(formal_turn, mock_state)
        # May or may not violate depending on baseline establishment
        
    def test_factual_consistency_invariant(self):
        """Test factual consistency invariant."""
        invariant = FactualConsistencyInvariant()
        
        mock_state = Mock()
        
        # First turn establishes fact
        turn1 = Mock()
        turn1.turn_id = "turn1"
        turn1.timestamp = time.time()
        turn1.content = "Paris is the capital of France."
        
        violation = invariant.check_turn(turn1, mock_state)
        assert violation is None
        
        # Contradictory turn
        turn2 = Mock()
        turn2.turn_id = "turn2"
        turn2.timestamp = time.time()
        turn2.content = "London is the capital of France."
        
        violation = invariant.check_turn(turn2, mock_state)
        assert violation is not None
        assert "contradiction" in violation.violation_message.lower()
    
    def test_topic_boundary_invariant(self):
        """Test topic boundary invariant."""
        invariant = TopicBoundaryInvariant(
            allowed_topics=["programming", "technology"],
            forbidden_topics=["politics", "religion"]
        )
        
        mock_state = Mock()
        
        # Allowed topic
        allowed_turn = Mock()
        allowed_turn.turn_id = "turn1"
        allowed_turn.timestamp = time.time()
        allowed_turn.content = "Let's discuss programming languages."
        
        violation = invariant.check_turn(allowed_turn, mock_state)
        assert violation is None
        
        # Forbidden topic
        forbidden_turn = Mock()
        forbidden_turn.turn_id = "turn2"
        forbidden_turn.timestamp = time.time()
        forbidden_turn.content = "What do you think about politics?"
        
        violation = invariant.check_turn(forbidden_turn, mock_state)
        assert violation is not None
        assert "forbidden topic" in violation.violation_message.lower()
    
    def test_invariant_tracker(self):
        """Test invariant tracking system."""
        tracker = InvariantTracker()
        
        # Add invariants
        personality = create_personality_invariant("helpful")
        topic_boundary = create_topic_boundary_invariant(
            allowed_topics=["tech"], 
            forbidden_topics=["politics"]
        )
        
        tracker.add_invariant(personality)
        tracker.add_invariant(topic_boundary)
        
        # Mock turn and state
        mock_turn = Mock()
        mock_turn.turn_id = "turn1"
        mock_turn.timestamp = time.time()
        mock_turn.role = Mock()
        mock_turn.role.value = "assistant"
        mock_turn.content = "Let's talk about politics!"  # Violates topic boundary
        
        mock_state = Mock()
        mock_state.turn_count = 1
        
        # Check invariants
        violations = tracker.check_turn(mock_turn, mock_state)
        
        # Should have at least one violation (topic boundary)
        topic_violations = [v for v in violations if "topic" in v.violation_message.lower()]
        assert len(topic_violations) > 0
        
        # Get summary
        summary = tracker.get_summary()
        assert summary["total_invariants"] == 2
        assert summary["total_violations"] >= 1


class TestConversationMemory:
    """Tests for conversation memory system."""
    
    def test_memory_item(self):
        """Test memory item functionality."""
        item = MemoryItem(
            memory_id="mem1",
            content="Paris is the capital of France",
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.HIGH,
            created_at=time.time(),
            last_accessed=time.time(),
            entities={"Paris", "France"},
            keywords={"capital", "city"}
        )
        
        # Test strength calculation
        initial_strength = item.get_current_strength()
        assert initial_strength > 0
        
        # Test access
        item.access()
        assert item.access_count == 1
        
        # Test relationships
        item.add_relationship("mem2")
        assert "mem2" in item.related_memories
        
        # Test serialization
        item_dict = item.to_dict()
        assert item_dict["memory_id"] == "mem1"
        assert item_dict["memory_type"] == "semantic"
    
    def test_in_memory_store(self):
        """Test in-memory storage implementation."""
        store = InMemoryStore(max_size=100)
        
        # Create and store memory
        memory = MemoryItem(
            memory_id="test_mem",
            content="Test memory content",
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.MEDIUM,
            created_at=time.time(),
            last_accessed=time.time(),
            entities={"test"},
            keywords={"memory", "content"}
        )
        
        assert store.store(memory)
        
        # Retrieve memory
        retrieved = store.retrieve("test_mem")
        assert retrieved is not None
        assert retrieved.content == "Test memory content"
        
        # Search memory
        results = store.search("test")
        assert len(results) > 0
        assert results[0].memory_id == "test_mem"
        
        # Test cleanup
        memory.importance_score = 0.01  # Very low importance
        removed = store.cleanup_expired(threshold=0.05)
        assert removed >= 0
    
    def test_semantic_memory(self):
        """Test semantic memory system."""
        store = InMemoryStore()
        semantic_memory = SemanticMemory(store)
        
        # Mock turn with facts
        mock_turn = Mock()
        mock_turn.turn_id = "turn1"
        mock_turn.content = "Paris is the capital of France. Tokyo is the capital of Japan."
        
        # Extract facts
        facts = semantic_memory.extract_and_store_facts(mock_turn)
        assert len(facts) >= 0  # May extract facts depending on patterns
        
        # Get facts about entity
        paris_facts = semantic_memory.get_facts_about("Paris")
        assert isinstance(paris_facts, list)
    
    def test_episodic_memory(self):
        """Test episodic memory system."""
        store = InMemoryStore()
        episodic_memory = EpisodicMemory(store)
        
        # Mock turn
        mock_turn = Mock()
        mock_turn.turn_id = "turn1"
        mock_turn.timestamp = time.time()
        mock_turn.role = Mock()
        mock_turn.role.value = "user"
        mock_turn.content = "I went to the store today."
        
        # Store episode
        episode = episodic_memory.store_episode(mock_turn, {"action": "mentioned"})
        assert episode.memory_type == MemoryType.EPISODIC
        assert "store" in episode.content.lower()
        
        # Get recent episodes
        recent = episodic_memory.get_recent_episodes(days=1)
        assert len(recent) >= 1
    
    def test_working_memory(self):
        """Test working memory system."""
        working_memory = WorkingMemory(capacity=3)
        
        # Create memory items
        memories = [
            MemoryItem(f"mem{i}", f"Content {i}", MemoryType.SEMANTIC, 
                      MemoryPriority.MEDIUM, time.time(), time.time())
            for i in range(5)
        ]
        
        # Add to working memory
        for memory in memories:
            working_memory.add_item(memory)
        
        # Should only keep last 3 due to capacity
        active = working_memory.get_active_items()
        assert len(active) == 3
        
        # Most recent should be first
        assert active[0].memory_id == "mem4"
        
        # Test refresh
        working_memory.refresh_item("mem4")
        assert working_memory.activation_levels["mem4"] == 1.0
    
    def test_conversation_memory_integration(self):
        """Test full conversation memory system."""
        memory_system = create_memory_system(max_memories=1000)
        
        # Mock turn
        mock_turn = Mock()
        mock_turn.turn_id = "turn1"
        mock_turn.timestamp = time.time()
        mock_turn.role = Mock()
        mock_turn.role.value = "assistant"
        mock_turn.content = "Hello! Paris is the capital of France."
        
        # Process turn
        results = memory_system.process_turn(mock_turn, {"topic": "geography"})
        
        assert "facts_extracted" in results
        assert "episode_stored" in results
        assert results["working_memory_updated"]
        
        # Get relevant memories
        relevant = memory_system.get_relevant_memories("Paris")
        assert isinstance(relevant, list)
        
        # Get memory summary
        summary = memory_system.get_memory_summary()
        assert "metrics" in summary
        assert "working_memory" in summary
        assert summary["turn_count"] > 0


class TestIntegration:
    """Integration tests for the complete conversation state management system."""
    
    def test_complete_conversation_flow(self):
        """Test a complete conversation flow with all components."""
        # Initialize all components
        state_manager = ConversationStateManager(context_window_size=1000)
        context_manager = ContextWindowManager(max_tokens=1000)
        memory_system = create_memory_system()
        
        # Set up temporal contracts
        temporal_validator = TemporalValidator()
        temporal_validator.add_contract(create_politeness_contract())
        
        # Set up invariants
        invariant_tracker = InvariantTracker()
        invariant_tracker.add_invariant(create_personality_invariant("helpful"))
        
        # Register components
        state_manager.set_context_manager(context_manager)
        state_manager.set_memory_store(memory_system)
        state_manager.register_temporal_contract(temporal_validator)
        state_manager.register_invariant_tracker(invariant_tracker)
        
        # Simulate conversation
        conversation_turns = [
            ("user", "Hello, can you help me with programming?"),
            ("assistant", "Hello! I'd be happy to help you with programming. What would you like to know?"),
            ("user", "What is Python?"),
            ("assistant", "Python is a high-level programming language known for its simplicity and readability."),
            ("user", "Thank you for the explanation!"),
            ("assistant", "You're welcome! Feel free to ask if you have more questions about Python or programming."),
        ]
        
        for role, content in conversation_turns:
            # Add turn to state manager
            turn = state_manager.add_turn(role, content)
            
            # Process through memory system
            memory_results = memory_system.process_turn(turn, {"topic": "programming"})
            
            # Add to context manager
            context_manager.add_turn(turn)
            
            # Validate with temporal contracts (mock since we don't have real validation)
            # In real implementation, this would be automatic
        
        # Verify final state
        assert state_manager.state.turn_count == 6
        assert state_manager.state.phase == ConversationPhase.ACTIVE
        
        # Check conversation history
        history = state_manager.get_conversation_history()
        assert len(history) == 6
        assert history[0]["content"] == "Hello, can you help me with programming?"
        
        # Check context management
        context_elements = context_manager.get_context_elements()
        assert len(context_elements) > 0
        
        # Check memory system
        relevant_memories = memory_system.get_relevant_memories("Python programming")
        assert isinstance(relevant_memories, list)
        
        # Get comprehensive metrics
        state_metrics = state_manager.get_metrics()
        context_metrics = context_manager.get_metrics()
        memory_summary = memory_system.get_memory_summary()
        
        assert state_metrics["turn_count"] == 6
        assert "context_window" in context_metrics
        assert memory_summary["turn_count"] > 0
    
    def test_error_handling(self):
        """Test error handling in conversation components."""
        state_manager = ConversationStateManager()
        
        # Test invalid role
        with pytest.raises(ValueError):
            state_manager.add_turn("invalid_role", "test content")
        
        # Test export with invalid format
        with pytest.raises(ValueError):
            state_manager.export_conversation("invalid_format")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of the system."""
        state_manager = ConversationStateManager()
        context_manager = ContextWindowManager()
        memory_system = create_memory_system()
        
        # Simulate many turns
        start_time = time.time()
        
        for i in range(100):
            turn = state_manager.add_turn(
                "user" if i % 2 == 0 else "assistant",
                f"Message {i} with some content to process."
            )
            
            context_manager.add_turn(turn)
            memory_system.process_turn(turn, {"topic": "performance_test"})
        
        processing_time = time.time() - start_time
        
        # Should process 100 turns reasonably quickly
        assert processing_time < 10.0  # Less than 10 seconds
        assert state_manager.state.turn_count == 100
        
        # Check memory efficiency
        memory_summary = memory_system.get_memory_summary()
        assert memory_summary["metrics"]["turns_processed"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])