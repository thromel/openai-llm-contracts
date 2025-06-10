"""Conversation memory system for maintaining context and knowledge across turns.

This module provides sophisticated memory management including semantic memory,
episodic memory, working memory, and memory retrieval systems.
"""

import time
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable
from enum import Enum, auto
import logging
import math
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in the conversation system."""
    SEMANTIC = "semantic"       # Facts, concepts, relationships
    EPISODIC = "episodic"       # Specific events, experiences
    WORKING = "working"         # Current context, active information
    PROCEDURAL = "procedural"   # How-to knowledge, processes
    EMOTIONAL = "emotional"     # Emotional context and sentiment


class MemoryPriority(Enum):
    """Priority levels for memory items."""
    CRITICAL = 5    # Must never be forgotten
    HIGH = 4        # Important to remember
    MEDIUM = 3      # Moderately important
    LOW = 2         # Less important
    TEMPORARY = 1   # Can be quickly forgotten


@dataclass
class MemoryItem:
    """Represents a single memory item."""
    memory_id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    created_at: float
    last_accessed: float
    access_count: int = 0
    importance_score: float = 1.0
    decay_rate: float = 0.1
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships
    related_memories: Set[str] = field(default_factory=set)
    source_turn_id: Optional[str] = None
    
    # Content analysis
    entities: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    sentiment: Optional[float] = None
    
    def get_current_strength(self, current_time: Optional[float] = None) -> float:
        """Calculate current memory strength based on decay."""
        current_time = current_time or time.time()
        time_elapsed = current_time - self.last_accessed
        
        # Memory strength formula: base_strength * decay_function * access_bonus
        base_strength = self.importance_score * self.priority.value
        decay_factor = math.exp(-self.decay_rate * time_elapsed / 3600)  # Decay per hour
        access_bonus = min(2.0, 1 + (self.access_count * 0.1))  # Bonus for frequent access
        
        return base_strength * decay_factor * access_bonus
    
    def access(self) -> None:
        """Record access to this memory item."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def add_relationship(self, memory_id: str) -> None:
        """Add relationship to another memory."""
        self.related_memories.add(memory_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "decay_rate": self.decay_rate,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "related_memories": list(self.related_memories),
            "source_turn_id": self.source_turn_id,
            "entities": list(self.entities),
            "keywords": list(self.keywords),
            "sentiment": self.sentiment,
            "current_strength": self.get_current_strength()
        }


class MemoryStore(ABC):
    """Abstract base class for memory storage systems."""
    
    @abstractmethod
    def store(self, memory_item: MemoryItem) -> bool:
        """Store a memory item."""
        pass
    
    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        pass
    
    @abstractmethod
    def search(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 10) -> List[MemoryItem]:
        """Search for memory items."""
        pass
    
    @abstractmethod
    def get_related(self, memory_id: str, limit: int = 5) -> List[MemoryItem]:
        """Get memories related to a specific memory."""
        pass
    
    @abstractmethod
    def cleanup_expired(self, threshold: float = 0.1) -> int:
        """Remove memories below strength threshold."""
        pass


class InMemoryStore(MemoryStore):
    """In-memory implementation of memory storage."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories: Dict[str, MemoryItem] = {}
        self.indices: Dict[str, Set[str]] = {
            "entities": defaultdict(set),
            "keywords": defaultdict(set),
            "tags": defaultdict(set),
            "type": defaultdict(set),
        }
    
    def store(self, memory_item: MemoryItem) -> bool:
        """Store a memory item."""
        # Check size limit
        if len(self.memories) >= self.max_size:
            self._evict_weakest_memories()
        
        self.memories[memory_item.memory_id] = memory_item
        self._update_indices(memory_item)
        
        logger.debug(f"Stored memory: {memory_item.memory_id}")
        return True
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
        return memory
    
    def search(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 10) -> List[MemoryItem]:
        """Search for memory items."""
        query_terms = set(query.lower().split())
        scored_memories: List[Tuple[float, MemoryItem]] = []
        
        for memory in self.memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            
            score = self._calculate_relevance_score(memory, query_terms)
            if score > 0:
                scored_memories.append((score, memory))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        results = [memory for _, memory in scored_memories[:limit]]
        
        # Update access for retrieved memories
        for memory in results:
            memory.access()
        
        return results
    
    def get_related(self, memory_id: str, limit: int = 5) -> List[MemoryItem]:
        """Get memories related to a specific memory."""
        memory = self.memories.get(memory_id)
        if not memory:
            return []
        
        related_memories = []
        
        # Direct relationships
        for related_id in memory.related_memories:
            if related_id in self.memories:
                related_memories.append(self.memories[related_id])
        
        # Similar entities/keywords
        if len(related_memories) < limit:
            for other_memory in self.memories.values():
                if other_memory.memory_id == memory_id:
                    continue
                
                # Calculate similarity based on entities and keywords
                similarity = self._calculate_similarity(memory, other_memory)
                if similarity > 0.3 and other_memory not in related_memories:
                    related_memories.append(other_memory)
                    
                    if len(related_memories) >= limit:
                        break
        
        # Update access for retrieved memories
        for related_memory in related_memories:
            related_memory.access()
        
        return related_memories[:limit]
    
    def cleanup_expired(self, threshold: float = 0.1) -> int:
        """Remove memories below strength threshold."""
        current_time = time.time()
        to_remove = []
        
        for memory_id, memory in self.memories.items():
            if (memory.priority != MemoryPriority.CRITICAL and 
                memory.get_current_strength(current_time) < threshold):
                to_remove.append(memory_id)
        
        # Remove weak memories
        for memory_id in to_remove:
            del self.memories[memory_id]
            self._remove_from_indices(memory_id)
        
        logger.info(f"Cleaned up {len(to_remove)} expired memories")
        return len(to_remove)
    
    def _calculate_relevance_score(self, memory: MemoryItem, query_terms: Set[str]) -> float:
        """Calculate relevance score for a memory against query terms."""
        score = 0.0
        
        # Content matching
        content_terms = set(memory.content.lower().split())
        content_overlap = len(query_terms & content_terms)
        score += content_overlap * 2.0
        
        # Entity matching
        entity_overlap = len(query_terms & memory.entities)
        score += entity_overlap * 1.5
        
        # Keyword matching
        keyword_overlap = len(query_terms & memory.keywords)
        score += keyword_overlap * 1.0
        
        # Tag matching
        tag_overlap = len(query_terms & memory.tags)
        score += tag_overlap * 0.5
        
        # Boost by memory strength
        score *= memory.get_current_strength()
        
        return score
    
    def _calculate_similarity(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """Calculate similarity between two memories."""
        # Entity similarity
        entity_similarity = len(memory1.entities & memory2.entities) / max(len(memory1.entities | memory2.entities), 1)
        
        # Keyword similarity
        keyword_similarity = len(memory1.keywords & memory2.keywords) / max(len(memory1.keywords | memory2.keywords), 1)
        
        # Tag similarity
        tag_similarity = len(memory1.tags & memory2.tags) / max(len(memory1.tags | memory2.tags), 1)
        
        # Type similarity
        type_similarity = 1.0 if memory1.memory_type == memory2.memory_type else 0.0
        
        # Weighted combination
        return (entity_similarity * 0.4 + 
                keyword_similarity * 0.3 + 
                tag_similarity * 0.2 + 
                type_similarity * 0.1)
    
    def _update_indices(self, memory: MemoryItem) -> None:
        """Update search indices for a memory."""
        for entity in memory.entities:
            self.indices["entities"][entity.lower()].add(memory.memory_id)
        
        for keyword in memory.keywords:
            self.indices["keywords"][keyword.lower()].add(memory.memory_id)
        
        for tag in memory.tags:
            self.indices["tags"][tag.lower()].add(memory.memory_id)
        
        self.indices["type"][memory.memory_type.value].add(memory.memory_id)
    
    def _remove_from_indices(self, memory_id: str) -> None:
        """Remove memory from all indices."""
        for index in self.indices.values():
            for memory_set in index.values():
                memory_set.discard(memory_id)
    
    def _evict_weakest_memories(self, count: int = 100) -> None:
        """Evict the weakest memories to make space."""
        current_time = time.time()
        
        # Sort memories by strength (weakest first)
        memories_by_strength = sorted(
            self.memories.items(),
            key=lambda x: x[1].get_current_strength(current_time)
        )
        
        # Remove weakest memories (but preserve critical ones)
        removed = 0
        for memory_id, memory in memories_by_strength:
            if memory.priority != MemoryPriority.CRITICAL and removed < count:
                del self.memories[memory_id]
                self._remove_from_indices(memory_id)
                removed += 1
        
        logger.info(f"Evicted {removed} weak memories to make space")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        current_time = time.time()
        
        type_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        total_strength = 0.0
        
        for memory in self.memories.values():
            type_counts[memory.memory_type.value] += 1
            priority_counts[memory.priority.value] += 1
            total_strength += memory.get_current_strength(current_time)
        
        return {
            "total_memories": len(self.memories),
            "max_size": self.max_size,
            "utilization": len(self.memories) / self.max_size,
            "types": dict(type_counts),
            "priorities": dict(priority_counts),
            "average_strength": total_strength / max(len(self.memories), 1),
            "index_sizes": {
                name: len(index) for name, index in self.indices.items()
            }
        }


class SemanticMemory:
    """Manages semantic memory - facts, concepts, and relationships."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.fact_patterns = [
            r'(\w+) is (\w+)',
            r'(\w+) has (\w+)',
            r'(\w+) can (\w+)',
            r'(\w+) costs \$?(\d+(?:\.\d+)?)',
            r'(\w+) is located in (\w+)',
        ]
        
        # Fact tracking
        self.facts: Dict[str, MemoryItem] = {}
        self.concepts: Dict[str, Set[str]] = defaultdict(set)  # concept -> related memories
        
    def extract_and_store_facts(self, turn: Any) -> List[MemoryItem]:
        """Extract facts from a turn and store them."""
        facts = self._extract_facts(turn.content)
        stored_memories = []
        
        for fact in facts:
            memory_id = f"fact_{hashlib.md5(fact.encode()).hexdigest()[:8]}"
            
            memory_item = MemoryItem(
                memory_id=memory_id,
                content=fact,
                memory_type=MemoryType.SEMANTIC,
                priority=MemoryPriority.HIGH,
                created_at=time.time(),
                last_accessed=time.time(),
                source_turn_id=turn.turn_id,
                entities=self._extract_entities(fact),
                keywords=self._extract_keywords(fact),
                tags={"fact"}
            )
            
            if self.memory_store.store(memory_item):
                self.facts[memory_id] = memory_item
                stored_memories.append(memory_item)
                self._update_concept_relationships(memory_item)
        
        return stored_memories
    
    def get_facts_about(self, entity: str) -> List[MemoryItem]:
        """Get all facts about a specific entity."""
        return self.memory_store.search(entity, MemoryType.SEMANTIC)
    
    def check_fact_consistency(self, new_fact: str) -> Optional[MemoryItem]:
        """Check if a new fact is consistent with existing facts."""
        # Simple consistency checking - could be more sophisticated
        new_entities = self._extract_entities(new_fact)
        
        for entity in new_entities:
            existing_facts = self.get_facts_about(entity)
            for fact_memory in existing_facts:
                if self._are_contradictory(new_fact, fact_memory.content):
                    return fact_memory
        
        return None
    
    def _extract_facts(self, content: str) -> List[str]:
        """Extract facts from content using patterns."""
        import re
        facts = []
        
        for pattern in self.fact_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    fact = f"{match[0]} {pattern.split('(')[1].split(')')[0]} {match[1]}"
                    facts.append(fact)
        
        return facts
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities from text."""
        import re
        # Simple entity extraction - capitalized words
        entities = set(re.findall(r'\b[A-Z][a-z]+\b', text))
        return entities
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = set(text.lower().split())
        return words - stop_words
    
    def _are_contradictory(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are contradictory."""
        # Simple contradiction detection
        return ("is not" in fact1 and "is" in fact2) or ("is not" in fact2 and "is" in fact1)
    
    def _update_concept_relationships(self, memory_item: MemoryItem) -> None:
        """Update concept relationship mappings."""
        for entity in memory_item.entities:
            self.concepts[entity.lower()].add(memory_item.memory_id)


class EpisodicMemory:
    """Manages episodic memory - specific events and experiences."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.episodes: Dict[str, MemoryItem] = {}
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date -> episode_ids
    
    def store_episode(self, turn: Any, context: Dict[str, Any]) -> MemoryItem:
        """Store an episodic memory from a turn."""
        episode_id = f"episode_{turn.turn_id}"
        
        # Create episode description
        episode_content = self._create_episode_description(turn, context)
        
        memory_item = MemoryItem(
            memory_id=episode_id,
            content=episode_content,
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.MEDIUM,
            created_at=turn.timestamp,
            last_accessed=time.time(),
            source_turn_id=turn.turn_id,
            entities=self._extract_entities(episode_content),
            keywords=self._extract_keywords(episode_content),
            tags={"episode", "experience"},
            metadata=context
        )
        
        # Store memory
        if self.memory_store.store(memory_item):
            self.episodes[episode_id] = memory_item
            
            # Update temporal index
            date_key = time.strftime("%Y-%m-%d", time.localtime(turn.timestamp))
            self.temporal_index[date_key].append(episode_id)
        
        return memory_item
    
    def get_recent_episodes(self, days: int = 7) -> List[MemoryItem]:
        """Get episodes from recent days."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_episodes = []
        
        for episode in self.episodes.values():
            if episode.created_at >= cutoff_time:
                recent_episodes.append(episode)
        
        # Sort by recency
        recent_episodes.sort(key=lambda x: x.created_at, reverse=True)
        return recent_episodes
    
    def get_episodes_about(self, topic: str) -> List[MemoryItem]:
        """Get episodes related to a specific topic."""
        return self.memory_store.search(topic, MemoryType.EPISODIC)
    
    def _create_episode_description(self, turn: Any, context: Dict[str, Any]) -> str:
        """Create a description of the episode."""
        role = turn.role.value
        content_summary = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
        
        description = f"{role.capitalize()} {context.get('action', 'said')}: {content_summary}"
        
        if context.get('topic'):
            description += f" (about {context['topic']})"
        
        return description
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities from text."""
        import re
        entities = set(re.findall(r'\b[A-Z][a-z]+\b', text))
        return entities
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = set(text.lower().split())
        return words - stop_words


class WorkingMemory:
    """Manages working memory - current context and active information."""
    
    def __init__(self, capacity: int = 7):  # Miller's magic number
        self.capacity = capacity
        self.active_items: deque = deque(maxlen=capacity)
        self.activation_levels: Dict[str, float] = {}
        
    def add_item(self, memory_item: MemoryItem) -> None:
        """Add item to working memory."""
        # Remove if already present
        self.remove_item(memory_item.memory_id)
        
        # Add to front
        self.active_items.appendleft(memory_item)
        self.activation_levels[memory_item.memory_id] = 1.0
        
        # Decay activation of other items
        self._decay_activation()
    
    def remove_item(self, memory_id: str) -> None:
        """Remove item from working memory."""
        self.active_items = deque([item for item in self.active_items if item.memory_id != memory_id], 
                                 maxlen=self.capacity)
        self.activation_levels.pop(memory_id, None)
    
    def get_active_items(self) -> List[MemoryItem]:
        """Get currently active items."""
        return list(self.active_items)
    
    def get_most_active(self, count: int = 3) -> List[MemoryItem]:
        """Get most active items."""
        sorted_items = sorted(
            self.active_items,
            key=lambda x: self.activation_levels.get(x.memory_id, 0),
            reverse=True
        )
        return sorted_items[:count]
    
    def refresh_item(self, memory_id: str) -> None:
        """Refresh activation of an item."""
        if memory_id in self.activation_levels:
            self.activation_levels[memory_id] = 1.0
    
    def _decay_activation(self, decay_rate: float = 0.1) -> None:
        """Decay activation levels of all items."""
        for memory_id in self.activation_levels:
            self.activation_levels[memory_id] *= (1 - decay_rate)


class MemoryRetrieval:
    """Handles memory retrieval with different strategies."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.retrieval_strategies = {
            "recency": self._retrieve_by_recency,
            "relevance": self._retrieve_by_relevance,
            "importance": self._retrieve_by_importance,
            "hybrid": self._retrieve_hybrid,
        }
    
    def retrieve(self, 
                query: str, 
                strategy: str = "hybrid",
                memory_type: Optional[MemoryType] = None,
                limit: int = 10) -> List[MemoryItem]:
        """Retrieve memories using specified strategy."""
        if strategy not in self.retrieval_strategies:
            strategy = "hybrid"
        
        return self.retrieval_strategies[strategy](query, memory_type, limit)
    
    def _retrieve_by_recency(self, query: str, memory_type: Optional[MemoryType], limit: int) -> List[MemoryItem]:
        """Retrieve memories by recency."""
        all_memories = self.memory_store.search(query, memory_type, limit * 2)
        
        # Sort by last accessed time
        all_memories.sort(key=lambda x: x.last_accessed, reverse=True)
        return all_memories[:limit]
    
    def _retrieve_by_relevance(self, query: str, memory_type: Optional[MemoryType], limit: int) -> List[MemoryItem]:
        """Retrieve memories by relevance."""
        return self.memory_store.search(query, memory_type, limit)
    
    def _retrieve_by_importance(self, query: str, memory_type: Optional[MemoryType], limit: int) -> List[MemoryItem]:
        """Retrieve memories by importance."""
        all_memories = self.memory_store.search(query, memory_type, limit * 2)
        
        # Sort by importance score
        all_memories.sort(key=lambda x: x.importance_score, reverse=True)
        return all_memories[:limit]
    
    def _retrieve_hybrid(self, query: str, memory_type: Optional[MemoryType], limit: int) -> List[MemoryItem]:
        """Retrieve memories using hybrid strategy."""
        all_memories = self.memory_store.search(query, memory_type, limit * 2)
        
        # Score by combination of factors
        scored_memories = []
        current_time = time.time()
        
        for memory in all_memories:
            # Combine multiple factors
            recency_score = 1.0 / (1.0 + (current_time - memory.last_accessed) / 3600)  # Recent = higher
            importance_score = memory.importance_score
            strength_score = memory.get_current_strength(current_time)
            access_score = math.log(1 + memory.access_count)
            
            # Weighted combination
            combined_score = (
                recency_score * 0.25 +
                importance_score * 0.25 +
                strength_score * 0.25 +
                access_score * 0.25
            )
            
            scored_memories.append((combined_score, memory))
        
        # Sort by combined score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]


class ConversationMemory:
    """Main conversation memory system that coordinates all memory types."""
    
    def __init__(self, 
                 memory_store: Optional[MemoryStore] = None,
                 working_memory_capacity: int = 7,
                 auto_cleanup: bool = True,
                 cleanup_interval: int = 100):  # cleanup every N turns
        """Initialize conversation memory system.
        
        Args:
            memory_store: Memory storage backend (defaults to InMemoryStore)
            working_memory_capacity: Size of working memory
            auto_cleanup: Automatically cleanup expired memories
            cleanup_interval: Number of turns between cleanup operations
        """
        self.memory_store = memory_store or InMemoryStore()
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # Memory subsystems
        self.semantic_memory = SemanticMemory(self.memory_store)
        self.episodic_memory = EpisodicMemory(self.memory_store)
        self.working_memory = WorkingMemory(working_memory_capacity)
        self.retrieval_system = MemoryRetrieval(self.memory_store)
        
        # Tracking
        self.turn_count = 0
        self.last_cleanup = 0
        self.memory_metrics = {
            "turns_processed": 0,
            "memories_created": 0,
            "facts_extracted": 0,
            "episodes_stored": 0,
            "cleanups_performed": 0,
        }
    
    def process_turn(self, turn: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a conversation turn through the memory system."""
        self.turn_count += 1
        self.memory_metrics["turns_processed"] += 1
        
        results = {
            "facts_extracted": [],
            "episode_stored": None,
            "working_memory_updated": False,
            "relevant_memories": []
        }
        
        # Extract and store semantic facts
        if turn.role.value == "assistant":  # Only extract facts from assistant responses
            facts = self.semantic_memory.extract_and_store_facts(turn)
            results["facts_extracted"] = facts
            self.memory_metrics["facts_extracted"] += len(facts)
            self.memory_metrics["memories_created"] += len(facts)
        
        # Store episodic memory
        episode = self.episodic_memory.store_episode(turn, context)
        results["episode_stored"] = episode
        self.memory_metrics["episodes_stored"] += 1
        self.memory_metrics["memories_created"] += 1
        
        # Update working memory with relevant items
        relevant_memories = self.get_relevant_memories(turn.content)
        results["relevant_memories"] = relevant_memories
        
        for memory in relevant_memories[:3]:  # Add top 3 to working memory
            self.working_memory.add_item(memory)
        
        results["working_memory_updated"] = True
        
        # Perform cleanup if needed
        if (self.auto_cleanup and 
            self.turn_count - self.last_cleanup >= self.cleanup_interval):
            self.cleanup_expired_memories()
        
        return results
    
    def get_relevant_memories(self, 
                            query: str, 
                            strategy: str = "hybrid",
                            limit: int = 10) -> List[MemoryItem]:
        """Get memories relevant to a query."""
        return self.retrieval_system.retrieve(query, strategy=strategy, limit=limit)
    
    def get_working_memory_context(self) -> List[MemoryItem]:
        """Get current working memory context."""
        return self.working_memory.get_active_items()
    
    def get_facts_about(self, entity: str) -> List[MemoryItem]:
        """Get all facts about an entity."""
        return self.semantic_memory.get_facts_about(entity)
    
    def get_recent_episodes(self, days: int = 7) -> List[MemoryItem]:
        """Get recent episodes."""
        return self.episodic_memory.get_recent_episodes(days)
    
    def check_fact_consistency(self, new_fact: str) -> Optional[MemoryItem]:
        """Check if a new fact is consistent with stored facts."""
        return self.semantic_memory.check_fact_consistency(new_fact)
    
    def cleanup_expired_memories(self, threshold: float = 0.1) -> int:
        """Clean up expired memories."""
        removed_count = self.memory_store.cleanup_expired(threshold)
        self.last_cleanup = self.turn_count
        self.memory_metrics["cleanups_performed"] += 1
        
        logger.info(f"Memory cleanup removed {removed_count} expired memories")
        return removed_count
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of conversation memory."""
        store_stats = {}
        if hasattr(self.memory_store, 'get_stats'):
            store_stats = self.memory_store.get_stats()
        
        return {
            "metrics": self.memory_metrics,
            "working_memory": {
                "active_items": len(self.working_memory.get_active_items()),
                "capacity": self.working_memory.capacity,
            },
            "storage": store_stats,
            "turn_count": self.turn_count,
            "last_cleanup": self.last_cleanup,
        }
    
    def export_memories(self, memory_type: Optional[MemoryType] = None, format: str = "json") -> str:
        """Export memories to various formats."""
        if format.lower() != "json":
            raise ValueError(f"Unsupported export format: {format}")
        
        # Get all memories of specified type
        all_memories = []
        if hasattr(self.memory_store, 'memories'):
            for memory in self.memory_store.memories.values():
                if memory_type is None or memory.memory_type == memory_type:
                    all_memories.append(memory.to_dict())
        
        return json.dumps(all_memories, indent=2, default=str)
    
    def reset(self) -> None:
        """Reset all memory systems."""
        if hasattr(self.memory_store, 'memories'):
            self.memory_store.memories.clear()
        
        self.working_memory.active_items.clear()
        self.working_memory.activation_levels.clear()
        
        self.turn_count = 0
        self.last_cleanup = 0
        self.memory_metrics = {
            "turns_processed": 0,
            "memories_created": 0,
            "facts_extracted": 0,
            "episodes_stored": 0,
            "cleanups_performed": 0,
        }
        
        logger.info("Conversation memory system reset")


# Convenience functions

def create_memory_system(max_memories: int = 10000, 
                        working_capacity: int = 7,
                        auto_cleanup: bool = True) -> ConversationMemory:
    """Create a conversation memory system with default configuration."""
    memory_store = InMemoryStore(max_size=max_memories)
    return ConversationMemory(
        memory_store=memory_store,
        working_memory_capacity=working_capacity,
        auto_cleanup=auto_cleanup
    )