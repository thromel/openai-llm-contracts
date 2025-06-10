"""Context window management for conversation state.

This module provides intelligent context window management with compression,
prioritization, and optimization for multi-turn conversations.
"""

import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from enum import Enum, auto
import logging
import json

logger = logging.getLogger(__name__)


class ContextCompressionStrategy(Enum):
    """Strategies for context compression."""
    NONE = "none"                    # No compression
    TRUNCATE_OLDEST = "truncate_oldest"      # Remove oldest turns
    TRUNCATE_MIDDLE = "truncate_middle"      # Remove middle turns, keep beginning and end
    SEMANTIC_COMPRESSION = "semantic"        # Compress based on semantic importance
    ROLLING_SUMMARY = "rolling_summary"     # Maintain rolling summary
    ADAPTIVE = "adaptive"                   # Adaptive strategy based on content


class ContextPriority(Enum):
    """Priority levels for context elements."""
    CRITICAL = 5      # Must always be preserved
    HIGH = 4          # High importance
    MEDIUM = 3        # Medium importance  
    LOW = 2           # Low importance
    MINIMAL = 1       # Can be easily compressed/removed


@dataclass
class ContextElement:
    """Represents an element in the context window."""
    element_id: str
    content: str
    turn_id: str
    role: str
    timestamp: float
    token_count: int
    priority: ContextPriority = ContextPriority.MEDIUM
    importance_score: float = 1.0
    compressed: bool = False
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def effective_tokens(self) -> int:
        """Get effective token count after compression."""
        return int(self.token_count * self.compression_ratio)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "element_id": self.element_id,
            "content": self.content,
            "turn_id": self.turn_id,
            "role": self.role,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "priority": self.priority.value,
            "importance_score": self.importance_score,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata,
        }


@dataclass
class ContextWindow:
    """Represents the current context window."""
    elements: List[ContextElement] = field(default_factory=list)
    max_tokens: int = 4096
    current_tokens: int = 0
    compression_applied: bool = False
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_element(self, element: ContextElement) -> bool:
        """Add element to context window if space allows."""
        if self.current_tokens + element.effective_tokens <= self.max_tokens:
            self.elements.append(element)
            self.current_tokens += element.effective_tokens
            return True
        return False
    
    def remove_element(self, element_id: str) -> bool:
        """Remove element from context window."""
        for i, element in enumerate(self.elements):
            if element.element_id == element_id:
                self.current_tokens -= element.effective_tokens
                self.elements.pop(i)
                return True
        return False
    
    def get_utilization(self) -> float:
        """Get context window utilization percentage."""
        return (self.current_tokens / self.max_tokens) * 100
    
    def get_elements_by_priority(self, priority: ContextPriority) -> List[ContextElement]:
        """Get elements by priority level."""
        return [e for e in self.elements if e.priority == priority]
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI message format."""
        return [
            {"role": element.role, "content": element.content}
            for element in self.elements
        ]


class TokenBudgetManager:
    """Manages token budgets for different parts of the context."""
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.allocations: Dict[str, int] = {}
        self.reserved: Dict[str, int] = {}
        self.used: Dict[str, int] = {}
        
    def allocate_budget(self, category: str, tokens: int, reserved: bool = False) -> bool:
        """Allocate token budget to a category."""
        current_allocated = sum(self.allocations.values())
        current_reserved = sum(self.reserved.values())
        
        available = self.total_budget - current_allocated - current_reserved
        
        if tokens <= available:
            self.allocations[category] = tokens
            if reserved:
                self.reserved[category] = tokens
            self.used[category] = 0
            return True
        return False
    
    def use_tokens(self, category: str, tokens: int) -> bool:
        """Use tokens from a category's budget."""
        if category not in self.allocations:
            return False
        
        available = self.allocations[category] - self.used.get(category, 0)
        if tokens <= available:
            self.used[category] = self.used.get(category, 0) + tokens
            return True
        return False
    
    def get_remaining(self, category: str) -> int:
        """Get remaining tokens in a category."""
        if category not in self.allocations:
            return 0
        return self.allocations[category] - self.used.get(category, 0)
    
    def get_utilization(self) -> Dict[str, float]:
        """Get utilization percentage for each category."""
        return {
            category: (self.used.get(category, 0) / allocation) * 100
            for category, allocation in self.allocations.items()
        }


class ContextCompressor(ABC):
    """Abstract base class for context compression strategies."""
    
    @abstractmethod
    def compress(self, elements: List[ContextElement], target_tokens: int) -> List[ContextElement]:
        """Compress context elements to fit target token count."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this compression strategy."""
        pass


class TruncateOldestCompressor(ContextCompressor):
    """Removes oldest elements to fit within token limit."""
    
    def compress(self, elements: List[ContextElement], target_tokens: int) -> List[ContextElement]:
        """Remove oldest elements until under token limit."""
        sorted_elements = sorted(elements, key=lambda x: x.timestamp, reverse=True)
        
        result = []
        current_tokens = 0
        
        for element in sorted_elements:
            if current_tokens + element.effective_tokens <= target_tokens:
                result.append(element)
                current_tokens += element.effective_tokens
            else:
                break
        
        # Restore original order
        result.sort(key=lambda x: x.timestamp)
        return result
    
    def get_strategy_name(self) -> str:
        return "truncate_oldest"


class TruncateMiddleCompressor(ContextCompressor):
    """Keeps beginning and end of conversation, removes middle."""
    
    def __init__(self, keep_start_ratio: float = 0.3, keep_end_ratio: float = 0.5):
        self.keep_start_ratio = keep_start_ratio
        self.keep_end_ratio = keep_end_ratio
    
    def compress(self, elements: List[ContextElement], target_tokens: int) -> List[ContextElement]:
        """Keep start and end elements, remove middle if needed."""
        if not elements:
            return []
        
        # Sort by timestamp
        sorted_elements = sorted(elements, key=lambda x: x.timestamp)
        
        # Calculate tokens for start and end
        start_tokens = int(target_tokens * self.keep_start_ratio)
        end_tokens = int(target_tokens * self.keep_end_ratio)
        
        # Select start elements
        start_elements = []
        current_tokens = 0
        for element in sorted_elements:
            if current_tokens + element.effective_tokens <= start_tokens:
                start_elements.append(element)
                current_tokens += element.effective_tokens
            else:
                break
        
        # Select end elements
        end_elements = []
        current_tokens = 0
        for element in reversed(sorted_elements):
            if element not in start_elements and current_tokens + element.effective_tokens <= end_tokens:
                end_elements.insert(0, element)
                current_tokens += element.effective_tokens
            else:
                break
        
        # Combine start and end
        result = start_elements + end_elements
        result.sort(key=lambda x: x.timestamp)
        
        return result
    
    def get_strategy_name(self) -> str:
        return "truncate_middle"


class SemanticCompressor(ContextCompressor):
    """Compresses based on semantic importance scores."""
    
    def compress(self, elements: List[ContextElement], target_tokens: int) -> List[ContextElement]:
        """Select elements based on importance scores."""
        # Sort by priority first, then by importance score
        sorted_elements = sorted(
            elements,
            key=lambda x: (x.priority.value, x.importance_score),
            reverse=True
        )
        
        result = []
        current_tokens = 0
        
        for element in sorted_elements:
            if current_tokens + element.effective_tokens <= target_tokens:
                result.append(element)
                current_tokens += element.effective_tokens
            else:
                # Try to compress this element if possible
                compressed_element = self._attempt_compression(element, target_tokens - current_tokens)
                if compressed_element:
                    result.append(compressed_element)
                    current_tokens += compressed_element.effective_tokens
                break
        
        # Restore chronological order
        result.sort(key=lambda x: x.timestamp)
        return result
    
    def _attempt_compression(self, element: ContextElement, available_tokens: int) -> Optional[ContextElement]:
        """Attempt to compress an element to fit available tokens."""
        if element.effective_tokens <= available_tokens:
            return element
        
        if element.compressed:
            return None  # Already compressed, can't compress further
        
        # Simple compression: truncate content
        compression_ratio = available_tokens / element.token_count
        if compression_ratio < 0.3:  # Don't compress too aggressively
            return None
        
        # Create compressed version
        compressed_content = element.content[:int(len(element.content) * compression_ratio)] + "..."
        compressed_element = ContextElement(
            element_id=element.element_id + "_compressed",
            content=compressed_content,
            turn_id=element.turn_id,
            role=element.role,
            timestamp=element.timestamp,
            token_count=available_tokens,
            priority=element.priority,
            importance_score=element.importance_score * 0.8,  # Slightly lower score for compressed
            compressed=True,
            compression_ratio=compression_ratio,
            metadata={**element.metadata, "original_tokens": element.token_count}
        )
        
        return compressed_element
    
    def get_strategy_name(self) -> str:
        return "semantic"


class RollingSummaryCompressor(ContextCompressor):
    """Maintains a rolling summary of the conversation."""
    
    def __init__(self, summary_ratio: float = 0.2):
        self.summary_ratio = summary_ratio
        self.current_summary = ""
    
    def compress(self, elements: List[ContextElement], target_tokens: int) -> List[ContextElement]:
        """Create rolling summary and keep recent elements."""
        if not elements:
            return []
        
        # Reserve tokens for summary
        summary_tokens = int(target_tokens * self.summary_ratio)
        remaining_tokens = target_tokens - summary_tokens
        
        # Keep most recent elements that fit
        sorted_elements = sorted(elements, key=lambda x: x.timestamp, reverse=True)
        recent_elements = []
        current_tokens = 0
        
        for element in sorted_elements:
            if current_tokens + element.effective_tokens <= remaining_tokens:
                recent_elements.append(element)
                current_tokens += element.effective_tokens
            else:
                break
        
        # Create summary from remaining elements
        summary_elements = [e for e in elements if e not in recent_elements]
        if summary_elements:
            summary_content = self._create_summary(summary_elements)
            summary_element = ContextElement(
                element_id="conversation_summary",
                content=summary_content,
                turn_id="summary",
                role="system",
                timestamp=min(e.timestamp for e in summary_elements),
                token_count=summary_tokens,
                priority=ContextPriority.HIGH,
                importance_score=1.0,
                compressed=True,
                metadata={"type": "rolling_summary", "summarized_elements": len(summary_elements)}
            )
            recent_elements.append(summary_element)
        
        # Restore chronological order
        recent_elements.sort(key=lambda x: x.timestamp)
        return recent_elements
    
    def _create_summary(self, elements: List[ContextElement]) -> str:
        """Create a summary of conversation elements."""
        if not elements:
            return ""
        
        # Simple summarization - in practice, could use an LLM
        user_messages = [e.content for e in elements if e.role == "user"]
        assistant_messages = [e.content for e in elements if e.role == "assistant"]
        
        summary_parts = []
        if user_messages:
            summary_parts.append(f"User discussed: {'; '.join(user_messages[:3])}")
        if assistant_messages:
            summary_parts.append(f"Assistant responded about: {'; '.join(assistant_messages[:3])}")
        
        return f"[SUMMARY: {'. '.join(summary_parts)}]"
    
    def get_strategy_name(self) -> str:
        return "rolling_summary"


class ContextOptimizer:
    """Optimizes context window usage and performance."""
    
    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "compression_time_ms": [],
            "compression_ratio": [],
            "token_utilization": [],
            "importance_preservation": []
        }
    
    def optimize_context(self, 
                        elements: List[ContextElement], 
                        max_tokens: int,
                        strategy: ContextCompressionStrategy = ContextCompressionStrategy.ADAPTIVE) -> Tuple[List[ContextElement], Dict[str, Any]]:
        """Optimize context elements for the given token limit."""
        start_time = time.time()
        original_tokens = sum(e.effective_tokens for e in elements)
        
        if original_tokens <= max_tokens:
            # No compression needed
            return elements, {
                "compression_needed": False, 
                "original_tokens": original_tokens,
                "final_tokens": original_tokens,
                "compression_ratio": 1.0,
                "elements_removed": 0,
                "token_utilization": original_tokens / max_tokens,
                "compression_time_ms": 0.0
            }
        
        # Select compressor based on strategy
        compressor = self._get_compressor(strategy, elements, max_tokens)
        
        # Perform compression
        optimized_elements = compressor.compress(elements, max_tokens)
        
        # Calculate metrics
        compression_time = (time.time() - start_time) * 1000
        final_tokens = sum(e.effective_tokens for e in optimized_elements)
        compression_ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0
        token_utilization = final_tokens / max_tokens
        importance_preservation = self._calculate_importance_preservation(elements, optimized_elements)
        
        # Record metrics
        self.performance_metrics["compression_time_ms"].append(compression_time)
        self.performance_metrics["compression_ratio"].append(compression_ratio)
        self.performance_metrics["token_utilization"].append(token_utilization)
        self.performance_metrics["importance_preservation"].append(importance_preservation)
        
        # Record optimization
        optimization_record = {
            "timestamp": time.time(),
            "strategy": compressor.get_strategy_name(),
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "compression_ratio": compression_ratio,
            "elements_removed": len(elements) - len(optimized_elements),
            "compression_time_ms": compression_time,
            "token_utilization": token_utilization,
            "importance_preservation": importance_preservation
        }
        
        self.optimization_history.append(optimization_record)
        
        return optimized_elements, optimization_record
    
    def _get_compressor(self, 
                       strategy: ContextCompressionStrategy, 
                       elements: List[ContextElement], 
                       max_tokens: int) -> ContextCompressor:
        """Get the appropriate compressor for the strategy."""
        if strategy == ContextCompressionStrategy.TRUNCATE_OLDEST:
            return TruncateOldestCompressor()
        elif strategy == ContextCompressionStrategy.TRUNCATE_MIDDLE:
            return TruncateMiddleCompressor()
        elif strategy == ContextCompressionStrategy.SEMANTIC_COMPRESSION:
            return SemanticCompressor()
        elif strategy == ContextCompressionStrategy.ROLLING_SUMMARY:
            return RollingSummaryCompressor()
        elif strategy == ContextCompressionStrategy.ADAPTIVE:
            return self._choose_adaptive_strategy(elements, max_tokens)
        else:
            return TruncateOldestCompressor()  # Default fallback
    
    def _choose_adaptive_strategy(self, elements: List[ContextElement], max_tokens: int) -> ContextCompressor:
        """Choose the best compression strategy based on context characteristics."""
        if not elements:
            return TruncateOldestCompressor()
        
        # Analyze context characteristics
        total_tokens = sum(e.effective_tokens for e in elements)
        avg_importance = sum(e.importance_score for e in elements) / len(elements)
        high_priority_ratio = len([e for e in elements if e.priority.value >= 4]) / len(elements)
        conversation_length = len(elements)
        
        # Decision logic for adaptive strategy
        if conversation_length > 20 and high_priority_ratio < 0.3:
            # Long conversation with mostly low-priority content -> rolling summary
            return RollingSummaryCompressor()
        elif high_priority_ratio > 0.5:
            # High proportion of important content -> semantic compression
            return SemanticCompressor()
        elif conversation_length < 10:
            # Short conversation -> truncate oldest
            return TruncateOldestCompressor()
        else:
            # Medium conversation -> truncate middle
            return TruncateMiddleCompressor()
    
    def _calculate_importance_preservation(self, 
                                         original: List[ContextElement], 
                                         optimized: List[ContextElement]) -> float:
        """Calculate how well importance scores were preserved."""
        if not original:
            return 1.0
        
        original_importance = sum(e.importance_score for e in original)
        preserved_importance = sum(e.importance_score for e in optimized)
        
        return preserved_importance / original_importance if original_importance > 0 else 1.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of optimization operations."""
        if not self.optimization_history:
            return {"optimizations_performed": 0}
        
        return {
            "optimizations_performed": len(self.optimization_history),
            "avg_compression_time_ms": sum(self.performance_metrics["compression_time_ms"]) / len(self.performance_metrics["compression_time_ms"]),
            "avg_compression_ratio": sum(self.performance_metrics["compression_ratio"]) / len(self.performance_metrics["compression_ratio"]),
            "avg_token_utilization": sum(self.performance_metrics["token_utilization"]) / len(self.performance_metrics["token_utilization"]),
            "avg_importance_preservation": sum(self.performance_metrics["importance_preservation"]) / len(self.performance_metrics["importance_preservation"]),
            "last_optimization": self.optimization_history[-1] if self.optimization_history else None
        }


class ContextWindowManager:
    """Main context window manager that coordinates all context management."""
    
    def __init__(self, 
                 max_tokens: int = 4096,
                 compression_strategy: ContextCompressionStrategy = ContextCompressionStrategy.ADAPTIVE,
                 auto_optimize: bool = True,
                 optimization_threshold: float = 0.9):
        """Initialize context window manager.
        
        Args:
            max_tokens: Maximum tokens in context window
            compression_strategy: Default compression strategy
            auto_optimize: Automatically optimize when threshold is reached
            optimization_threshold: Utilization threshold for auto-optimization (0.0-1.0)
        """
        self.max_tokens = max_tokens
        self.compression_strategy = compression_strategy
        self.auto_optimize = auto_optimize
        self.optimization_threshold = optimization_threshold
        
        # Core components
        self.context_window = ContextWindow(max_tokens=max_tokens)
        self.token_budget_manager = TokenBudgetManager(max_tokens)
        self.optimizer = ContextOptimizer()
        
        # State tracking
        self.turn_importance_calculator: Optional[Callable] = None
        self.priority_calculator: Optional[Callable] = None
        
        # Performance tracking
        self.management_metrics = {
            "elements_added": 0,
            "elements_removed": 0,
            "optimizations_triggered": 0,
            "budget_violations": 0,
        }
        
        logger.info(f"ContextWindowManager initialized with {max_tokens} max tokens")
    
    def add_turn(self, turn_state: Any) -> bool:
        """Add a conversation turn to the context window."""
        # Create context element from turn
        element = self._turn_to_context_element(turn_state)
        
        # Check if we need to optimize first
        if self.auto_optimize and self._should_optimize(element):
            self.optimize_context()
        
        # Try to add element
        if self.context_window.add_element(element):
            self.management_metrics["elements_added"] += 1
            logger.debug(f"Added turn {turn_state.turn_id} to context window")
            return True
        else:
            # Force optimization and try again
            self.optimize_context()
            if self.context_window.add_element(element):
                self.management_metrics["elements_added"] += 1
                logger.debug(f"Added turn {turn_state.turn_id} after optimization")
                return True
            else:
                logger.warning(f"Failed to add turn {turn_state.turn_id} to context window")
                return False
    
    def optimize_context(self) -> Dict[str, Any]:
        """Optimize the current context window."""
        elements = self.context_window.elements.copy()
        
        optimized_elements, optimization_record = self.optimizer.optimize_context(
            elements, 
            self.max_tokens,
            self.compression_strategy
        )
        
        # Update context window
        self.context_window.elements = optimized_elements
        self.context_window.current_tokens = sum(e.effective_tokens for e in optimized_elements)
        self.context_window.compression_applied = True
        self.context_window.optimization_history.append(optimization_record)
        
        self.management_metrics["optimizations_triggered"] += 1
        
        logger.info(f"Context optimized: {optimization_record['original_tokens']} -> {optimization_record['final_tokens']} tokens")
        return optimization_record
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get context window as OpenAI-compatible messages."""
        return self.context_window.to_messages()
    
    def get_context_elements(self) -> List[ContextElement]:
        """Get current context elements."""
        return self.context_window.elements.copy()
    
    def set_importance_calculator(self, calculator: Callable[[Any], float]) -> None:
        """Set function to calculate turn importance."""
        self.turn_importance_calculator = calculator
    
    def set_priority_calculator(self, calculator: Callable[[Any], ContextPriority]) -> None:
        """Set function to calculate turn priority."""
        self.priority_calculator = calculator
    
    def get_utilization(self) -> float:
        """Get current context window utilization percentage."""
        return self.context_window.get_utilization()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive context management metrics."""
        return {
            "context_window": {
                "max_tokens": self.max_tokens,
                "current_tokens": self.context_window.current_tokens,
                "utilization_percent": self.context_window.get_utilization(),
                "element_count": len(self.context_window.elements),
                "compression_applied": self.context_window.compression_applied,
            },
            "management_metrics": self.management_metrics,
            "optimization_performance": self.optimizer.get_performance_summary(),
            "token_budget": {
                "allocations": self.token_budget_manager.allocations,
                "utilization": self.token_budget_manager.get_utilization(),
            }
        }
    
    def _turn_to_context_element(self, turn_state: Any) -> ContextElement:
        """Convert a turn state to a context element."""
        # Calculate importance score
        importance_score = 1.0
        if self.turn_importance_calculator:
            try:
                importance_score = self.turn_importance_calculator(turn_state)
            except Exception as e:
                logger.warning(f"Error calculating importance: {e}")
        
        # Calculate priority
        priority = ContextPriority.MEDIUM
        if self.priority_calculator:
            try:
                priority = self.priority_calculator(turn_state)
            except Exception as e:
                logger.warning(f"Error calculating priority: {e}")
        
        return ContextElement(
            element_id=turn_state.turn_id,
            content=turn_state.content,
            turn_id=turn_state.turn_id,
            role=turn_state.role.value,
            timestamp=turn_state.timestamp,
            token_count=turn_state.token_count,
            priority=priority,
            importance_score=importance_score,
            metadata=turn_state.metadata.copy()
        )
    
    def _should_optimize(self, new_element: ContextElement) -> bool:
        """Check if optimization should be triggered."""
        projected_tokens = self.context_window.current_tokens + new_element.effective_tokens
        projected_utilization = projected_tokens / self.max_tokens
        
        return projected_utilization > self.optimization_threshold
    
    def __str__(self) -> str:
        return f"ContextWindowManager(tokens={self.context_window.current_tokens}/{self.max_tokens}, elements={len(self.context_window.elements)})"