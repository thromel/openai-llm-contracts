"""Conversation invariants tracking and validation.

This module provides invariant tracking across conversation turns to ensure
consistency in personality, facts, tone, and other conversation properties.
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set, Callable, Pattern
from enum import Enum, auto
import logging
import json

logger = logging.getLogger(__name__)


class InvariantSeverity(Enum):
    """Severity levels for invariant violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InvariantType(Enum):
    """Types of conversation invariants."""
    PERSONALITY = "personality"
    FACTUAL = "factual"
    TONE = "tone"
    TOPIC = "topic"
    MEMORY = "memory"
    RELATIONSHIP = "relationship"
    STYLE = "style"
    CONSTRAINT = "constraint"


@dataclass
class InvariantViolation:
    """Represents an invariant violation."""
    invariant_name: str
    invariant_type: InvariantType
    violation_message: str
    turn_id: str
    timestamp: float
    severity: InvariantSeverity
    expected_state: str
    actual_state: str
    context: Dict[str, Any] = field(default_factory=dict)
    auto_fix_suggestion: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "invariant_name": self.invariant_name,
            "invariant_type": self.invariant_type.value,
            "violation_message": self.violation_message,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "expected_state": self.expected_state,
            "actual_state": self.actual_state,
            "context": self.context,
            "auto_fix_suggestion": self.auto_fix_suggestion,
            "confidence": self.confidence,
        }


class ConversationInvariant(ABC):
    """Base class for conversation invariants."""
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 invariant_type: InvariantType,
                 severity: InvariantSeverity = InvariantSeverity.MEDIUM,
                 enabled: bool = True):
        self.name = name
        self.description = description
        self.invariant_type = invariant_type
        self.severity = severity
        self.enabled = enabled
        
        # State tracking
        self.violations: List[InvariantViolation] = []
        self.check_count = 0
        self.violation_count = 0
        self.last_check_time = 0.0
        
        # Configuration
        self.auto_fix_enabled = True
        self.confidence_threshold = 0.7
        
    @abstractmethod
    def check_turn(self, turn: Any, conversation_state: Any) -> Optional[InvariantViolation]:
        """Check if the invariant is violated for a given turn."""
        pass
    
    def initialize(self, conversation_state: Any) -> None:
        """Initialize invariant with conversation context."""
        pass
    
    def update_state(self, turn: Any, conversation_state: Any) -> None:
        """Update invariant state based on new turn."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the invariant."""
        return {
            "name": self.name,
            "type": self.invariant_type.value,
            "enabled": self.enabled,
            "check_count": self.check_count,
            "violation_count": self.violation_count,
            "violation_rate": self.violation_count / max(self.check_count, 1),
            "last_check_time": self.last_check_time,
            "recent_violations": len([v for v in self.violations if time.time() - v.timestamp < 300])  # Last 5 minutes
        }
    
    def reset(self) -> None:
        """Reset invariant state."""
        self.violations.clear()
        self.check_count = 0
        self.violation_count = 0
        self.last_check_time = 0.0


class PersonalityInvariant(ConversationInvariant):
    """Ensures consistent personality traits across the conversation."""
    
    def __init__(self, 
                 name: str = "personality_consistency",
                 personality_traits: Optional[Dict[str, str]] = None,
                 personality_description: str = "",
                 strict_mode: bool = False):
        super().__init__(
            name=name,
            description=f"Maintains consistent personality: {personality_description}",
            invariant_type=InvariantType.PERSONALITY,
            severity=InvariantSeverity.MEDIUM
        )
        
        # Personality configuration
        self.personality_traits = personality_traits or {}
        self.personality_description = personality_description
        self.strict_mode = strict_mode
        
        # Personality indicators
        self.positive_indicators: List[Pattern] = []
        self.negative_indicators: List[Pattern] = []
        self.trait_patterns: Dict[str, List[Pattern]] = {}
        
        # Initialize patterns from traits
        self._initialize_trait_patterns()
        
        # State tracking
        self.trait_history: Dict[str, List[float]] = {}  # Trait scores over time
        self.baseline_established = False
        self.baseline_traits: Dict[str, float] = {}
    
    def _initialize_trait_patterns(self) -> None:
        """Initialize pattern matching for personality traits."""
        # Common personality trait patterns
        trait_patterns = {
            "friendly": [
                r'\b(hello|hi|hey|good morning|good afternoon|wonderful|great|awesome)\b',
                r'\b(thank you|thanks|appreciate|helpful|kind)\b',
                r'[!]{1,2}(?![!])',  # Moderate exclamation use
            ],
            "formal": [
                r'\b(please|kindly|would you|could you|may I|thank you)\b',
                r'\b(certainly|indeed|however|furthermore|moreover)\b',
                r'[.]{1}(?![.])',  # Periods over exclamations
            ],
            "casual": [
                r'\b(yeah|yep|nope|gonna|wanna|kinda|sorta)\b',
                r'\b(cool|awesome|sweet|neat|nice)\b',
                r"'(ll|re|ve|d)\b",  # Contractions
            ],
            "helpful": [
                r'\b(help|assist|support|guide|explain|show|teach)\b',
                r'\b(let me|I can|I\'ll help|happy to)\b',
                r'\b(solution|answer|resolve|fix)\b',
            ],
            "professional": [
                r'\b(professional|business|enterprise|solution|strategy)\b',
                r'\b(analysis|assessment|evaluation|implementation)\b',
                r'\b(recommend|suggest|propose|advise)\b',
            ]
        }
        
        # Compile patterns for configured traits
        for trait, value in self.personality_traits.items():
            if trait.lower() in trait_patterns:
                self.trait_patterns[trait] = [
                    re.compile(pattern, re.IGNORECASE) 
                    for pattern in trait_patterns[trait.lower()]
                ]
    
    def check_turn(self, turn: Any, conversation_state: Any) -> Optional[InvariantViolation]:
        """Check personality consistency for a turn."""
        if not self.enabled or turn.role.value != "assistant":
            return None
        
        self.check_count += 1
        self.last_check_time = time.time()
        
        # Calculate personality scores for this turn
        turn_scores = self._calculate_personality_scores(turn.content)
        
        # Update trait history
        for trait, score in turn_scores.items():
            if trait not in self.trait_history:
                self.trait_history[trait] = []
            self.trait_history[trait].append(score)
        
        # Establish baseline if needed
        if not self.baseline_established and len(conversation_state.turns) >= 3:
            self._establish_baseline()
        
        # Check for violations if baseline is established
        if self.baseline_established:
            violation = self._check_personality_deviation(turn, turn_scores)
            if violation:
                self.violations.append(violation)
                self.violation_count += 1
                return violation
        
        return None
    
    def _calculate_personality_scores(self, content: str) -> Dict[str, float]:
        """Calculate personality trait scores for content."""
        scores = {}
        
        for trait, patterns in self.trait_patterns.items():
            score = 0.0
            word_count = len(content.split())
            
            for pattern in patterns:
                matches = len(pattern.findall(content))
                score += matches / max(word_count, 1)  # Normalize by word count
            
            scores[trait] = min(score, 1.0)  # Cap at 1.0
        
        return scores
    
    def _establish_baseline(self) -> None:
        """Establish baseline personality from early turns."""
        if not self.trait_history:
            return
        
        # Calculate average scores from first few turns
        for trait, scores in self.trait_history.items():
            if scores:
                self.baseline_traits[trait] = sum(scores) / len(scores)
        
        self.baseline_established = True
        logger.info(f"Personality baseline established: {self.baseline_traits}")
    
    def _check_personality_deviation(self, turn: Any, turn_scores: Dict[str, float]) -> Optional[InvariantViolation]:
        """Check if personality has deviated from baseline."""
        deviations = {}
        significant_deviations = []
        
        for trait, baseline_score in self.baseline_traits.items():
            current_score = turn_scores.get(trait, 0.0)
            deviation = abs(current_score - baseline_score)
            deviations[trait] = deviation
            
            # Threshold for significant deviation
            threshold = 0.3 if self.strict_mode else 0.5
            if deviation > threshold:
                significant_deviations.append((trait, baseline_score, current_score, deviation))
        
        if significant_deviations:
            # Create violation for most significant deviation
            trait, baseline, current, deviation = max(significant_deviations, key=lambda x: x[3])
            
            return InvariantViolation(
                invariant_name=self.name,
                invariant_type=self.invariant_type,
                violation_message=f"Personality trait '{trait}' deviated significantly: {baseline:.2f} -> {current:.2f}",
                turn_id=turn.turn_id,
                timestamp=turn.timestamp,
                severity=InvariantSeverity.HIGH if deviation > 0.7 else self.severity,
                expected_state=f"{trait}: ~{baseline:.2f}",
                actual_state=f"{trait}: {current:.2f}",
                context={"all_deviations": deviations, "baseline_traits": self.baseline_traits},
                auto_fix_suggestion=f"Adjust response tone to be more consistent with established {trait} level",
                confidence=min(deviation, 1.0)
            )
        
        return None


class FactualConsistencyInvariant(ConversationInvariant):
    """Ensures factual consistency across conversation turns."""
    
    def __init__(self, 
                 name: str = "factual_consistency",
                 fact_extraction_enabled: bool = True,
                 contradiction_detection_enabled: bool = True):
        super().__init__(
            name=name,
            description="Maintains factual consistency across conversation",
            invariant_type=InvariantType.FACTUAL,
            severity=InvariantSeverity.HIGH
        )
        
        self.fact_extraction_enabled = fact_extraction_enabled
        self.contradiction_detection_enabled = contradiction_detection_enabled
        
        # Fact tracking
        self.stated_facts: Dict[str, Dict[str, Any]] = {}  # fact_id -> fact_info
        self.fact_patterns: List[Pattern] = self._compile_fact_patterns()
        
        # Contradiction patterns
        self.contradiction_indicators = [
            re.compile(r'\b(actually|correction|mistake|wrong|incorrect|sorry)\b', re.IGNORECASE),
            re.compile(r'\b(not|never|no longer|used to be|previously)\b', re.IGNORECASE),
        ]
    
    def _compile_fact_patterns(self) -> List[Pattern]:
        """Compile patterns for fact extraction."""
        fact_patterns = [
            re.compile(r'\b(\w+) is (\w+)\b', re.IGNORECASE),  # X is Y
            re.compile(r'\b(\w+) was (\w+)\b', re.IGNORECASE),  # X was Y  
            re.compile(r'\b(\w+) has (\d+) (\w+)\b', re.IGNORECASE),  # X has N Y
            re.compile(r'\b(\w+) costs? \$?(\d+(?:\.\d+)?)\b', re.IGNORECASE),  # X costs Y
            re.compile(r'\bin (\d{4}), (\w+) (\w+)\b', re.IGNORECASE),  # In YEAR, X Y
        ]
        return fact_patterns
    
    def check_turn(self, turn: Any, conversation_state: Any) -> Optional[InvariantViolation]:
        """Check factual consistency for a turn."""
        if not self.enabled:
            return None
        
        self.check_count += 1
        self.last_check_time = time.time()
        
        # Extract new facts from this turn
        if self.fact_extraction_enabled:
            new_facts = self._extract_facts(turn.content, turn.turn_id)
            
            # Check for contradictions with existing facts
            if self.contradiction_detection_enabled:
                violation = self._check_contradictions(new_facts, turn)
                if violation:
                    self.violations.append(violation)
                    self.violation_count += 1
                    return violation
            
            # Store new facts
            for fact_id, fact_info in new_facts.items():
                self.stated_facts[fact_id] = fact_info
        
        return None
    
    def _extract_facts(self, content: str, turn_id: str) -> Dict[str, Dict[str, Any]]:
        """Extract facts from content."""
        facts = {}
        
        for pattern in self.fact_patterns:
            for match in pattern.finditer(content):
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].lower().strip()
                    predicate = " ".join(groups[1:]).lower().strip()
                    
                    fact_id = f"{subject}_{hash(predicate) % 10000}"
                    facts[fact_id] = {
                        "subject": subject,
                        "predicate": predicate,
                        "full_statement": match.group(0),
                        "turn_id": turn_id,
                        "timestamp": time.time(),
                        "pattern_type": pattern.pattern
                    }
        
        return facts
    
    def _check_contradictions(self, new_facts: Dict[str, Dict[str, Any]], turn: Any) -> Optional[InvariantViolation]:
        """Check for contradictions between new and existing facts."""
        for new_fact_id, new_fact in new_facts.items():
            subject = new_fact["subject"]
            
            # Look for existing facts about the same subject
            for existing_fact_id, existing_fact in self.stated_facts.items():
                if existing_fact["subject"] == subject:
                    # Check for contradictory predicates
                    if self._are_contradictory(new_fact["predicate"], existing_fact["predicate"]):
                        return InvariantViolation(
                            invariant_name=self.name,
                            invariant_type=self.invariant_type,
                            violation_message=f"Factual contradiction about '{subject}': previously stated '{existing_fact['predicate']}', now stating '{new_fact['predicate']}'",
                            turn_id=turn.turn_id,
                            timestamp=turn.timestamp,
                            severity=self.severity,
                            expected_state=f"{subject}: {existing_fact['predicate']}",
                            actual_state=f"{subject}: {new_fact['predicate']}",
                            context={
                                "existing_fact": existing_fact,
                                "new_fact": new_fact,
                                "previous_turn": existing_fact["turn_id"]
                            },
                            auto_fix_suggestion=f"Clarify or correct the statement about {subject}",
                            confidence=0.8
                        )
        
        return None
    
    def _are_contradictory(self, predicate1: str, predicate2: str) -> bool:
        """Check if two predicates are contradictory."""
        # Simple contradiction detection
        # In practice, this could use more sophisticated NLP
        
        # Direct contradiction words
        contradiction_pairs = [
            ("is", "is not"), ("was", "was not"), ("has", "has no"),
            ("can", "cannot"), ("will", "will not"), ("true", "false"),
            ("yes", "no"), ("alive", "dead"), ("open", "closed")
        ]
        
        for pos, neg in contradiction_pairs:
            if (pos in predicate1 and neg in predicate2) or (neg in predicate1 and pos in predicate2):
                return True
        
        # Numerical contradictions (simple)
        import re
        numbers1 = re.findall(r'\d+(?:\.\d+)?', predicate1)
        numbers2 = re.findall(r'\d+(?:\.\d+)?', predicate2)
        
        if numbers1 and numbers2:
            try:
                num1 = float(numbers1[0])
                num2 = float(numbers2[0])
                # Consider significantly different numbers as potential contradictions
                if abs(num1 - num2) / max(num1, num2, 1) > 0.5:
                    return True
            except ValueError:
                pass
        
        return False


class ToneInvariant(ConversationInvariant):
    """Ensures consistent tone throughout the conversation."""
    
    def __init__(self, 
                 name: str = "tone_consistency",
                 target_tone: str = "professional",
                 tone_indicators: Optional[Dict[str, List[str]]] = None):
        super().__init__(
            name=name,
            description=f"Maintains consistent {target_tone} tone",
            invariant_type=InvariantType.TONE,
            severity=InvariantSeverity.MEDIUM
        )
        
        self.target_tone = target_tone
        self.tone_indicators = tone_indicators or self._get_default_tone_indicators()
        
        # Compile tone patterns
        self.tone_patterns: Dict[str, List[Pattern]] = {}
        for tone, indicators in self.tone_indicators.items():
            self.tone_patterns[tone] = [
                re.compile(rf'\b{re.escape(indicator)}\b', re.IGNORECASE)
                for indicator in indicators
            ]
        
        # Tone tracking
        self.tone_scores_history: List[Dict[str, float]] = []
        self.baseline_tone_scores: Dict[str, float] = {}
        self.baseline_established = False
    
    def _get_default_tone_indicators(self) -> Dict[str, List[str]]:
        """Get default tone indicators."""
        return {
            "professional": ["professional", "business", "formal", "official", "corporate", "enterprise"],
            "casual": ["casual", "relaxed", "informal", "friendly", "chill", "easy"],
            "enthusiastic": ["exciting", "amazing", "awesome", "fantastic", "incredible", "wonderful"],
            "serious": ["serious", "important", "critical", "significant", "crucial", "vital"],
            "friendly": ["friendly", "warm", "welcoming", "kind", "nice", "pleasant"],
            "technical": ["technical", "specific", "detailed", "precise", "accurate", "systematic"]
        }
    
    def check_turn(self, turn: Any, conversation_state: Any) -> Optional[InvariantViolation]:
        """Check tone consistency for a turn."""
        if not self.enabled or turn.role.value != "assistant":
            return None
        
        self.check_count += 1
        self.last_check_time = time.time()
        
        # Calculate tone scores for this turn
        tone_scores = self._calculate_tone_scores(turn.content)
        self.tone_scores_history.append(tone_scores)
        
        # Establish baseline if needed
        if not self.baseline_established and len(self.tone_scores_history) >= 3:
            self._establish_tone_baseline()
        
        # Check for tone violations
        if self.baseline_established:
            violation = self._check_tone_consistency(turn, tone_scores)
            if violation:
                self.violations.append(violation)
                self.violation_count += 1
                return violation
        
        return None
    
    def _calculate_tone_scores(self, content: str) -> Dict[str, float]:
        """Calculate tone scores for content."""
        scores = {}
        word_count = len(content.split())
        
        for tone, patterns in self.tone_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(pattern.findall(content))
                score += matches
            
            # Normalize by content length
            scores[tone] = score / max(word_count, 1)
        
        return scores
    
    def _establish_tone_baseline(self) -> None:
        """Establish baseline tone from early turns."""
        if not self.tone_scores_history:
            return
        
        # Average scores across recent history
        for tone in self.tone_patterns.keys():
            scores = [turn_scores.get(tone, 0.0) for turn_scores in self.tone_scores_history]
            self.baseline_tone_scores[tone] = sum(scores) / len(scores)
        
        self.baseline_established = True
        logger.info(f"Tone baseline established: {self.baseline_tone_scores}")
    
    def _check_tone_consistency(self, turn: Any, tone_scores: Dict[str, float]) -> Optional[InvariantViolation]:
        """Check if tone has deviated from baseline."""
        if self.target_tone not in tone_scores:
            return None
        
        current_target_score = tone_scores[self.target_tone]
        baseline_target_score = self.baseline_tone_scores.get(self.target_tone, 0.0)
        
        # Check if target tone has decreased significantly
        if baseline_target_score > 0.1:  # Only if we had some baseline target tone
            deviation = baseline_target_score - current_target_score
            
            if deviation > 0.3:  # Significant decrease in target tone
                return InvariantViolation(
                    invariant_name=self.name,
                    invariant_type=self.invariant_type,
                    violation_message=f"Tone deviation: {self.target_tone} tone decreased from {baseline_target_score:.2f} to {current_target_score:.2f}",
                    turn_id=turn.turn_id,
                    timestamp=turn.timestamp,
                    severity=self.severity,
                    expected_state=f"{self.target_tone} tone: ~{baseline_target_score:.2f}",
                    actual_state=f"{self.target_tone} tone: {current_target_score:.2f}",
                    context={
                        "tone_scores": tone_scores,
                        "baseline_scores": self.baseline_tone_scores,
                        "target_tone": self.target_tone
                    },
                    auto_fix_suggestion=f"Increase {self.target_tone} tone in response",
                    confidence=min(deviation, 1.0)
                )
        
        return None


class TopicBoundaryInvariant(ConversationInvariant):
    """Ensures conversation stays within defined topic boundaries."""
    
    def __init__(self, 
                 name: str = "topic_boundaries",
                 allowed_topics: Optional[List[str]] = None,
                 forbidden_topics: Optional[List[str]] = None,
                 topic_detection_threshold: float = 0.3):
        super().__init__(
            name=name,
            description="Maintains conversation within topic boundaries",
            invariant_type=InvariantType.TOPIC,
            severity=InvariantSeverity.MEDIUM
        )
        
        self.allowed_topics = allowed_topics or []
        self.forbidden_topics = forbidden_topics or []
        self.topic_detection_threshold = topic_detection_threshold
        
        # Compile topic patterns
        self.allowed_patterns = [
            re.compile(rf'\b{re.escape(topic)}\b', re.IGNORECASE)
            for topic in self.allowed_topics
        ]
        self.forbidden_patterns = [
            re.compile(rf'\b{re.escape(topic)}\b', re.IGNORECASE)
            for topic in self.forbidden_topics
        ]
        
        # Topic tracking
        self.detected_topics: Dict[str, List[str]] = {}  # turn_id -> topics
        self.topic_drift_score = 0.0
    
    def check_turn(self, turn: Any, conversation_state: Any) -> Optional[InvariantViolation]:
        """Check topic boundaries for a turn."""
        if not self.enabled:
            return None
        
        self.check_count += 1
        self.last_check_time = time.time()
        
        # Detect topics in this turn
        detected_allowed = self._detect_topics(turn.content, self.allowed_patterns)
        detected_forbidden = self._detect_topics(turn.content, self.forbidden_patterns)
        
        self.detected_topics[turn.turn_id] = detected_allowed + detected_forbidden
        
        # Check for forbidden topics
        if detected_forbidden:
            violation = InvariantViolation(
                invariant_name=self.name,
                invariant_type=self.invariant_type,
                violation_message=f"Forbidden topic(s) detected: {', '.join(detected_forbidden)}",
                turn_id=turn.turn_id,
                timestamp=turn.timestamp,
                severity=InvariantSeverity.HIGH,
                expected_state="No forbidden topics",
                actual_state=f"Contains: {', '.join(detected_forbidden)}",
                context={
                    "detected_forbidden": detected_forbidden,
                    "forbidden_topics": self.forbidden_topics
                },
                auto_fix_suggestion="Redirect conversation away from forbidden topics",
                confidence=0.9
            )
            
            self.violations.append(violation)
            self.violation_count += 1
            return violation
        
        # Check for topic drift if allowed topics are specified
        if self.allowed_topics and not detected_allowed:
            self.topic_drift_score += 0.1
            
            if self.topic_drift_score > self.topic_detection_threshold:
                violation = InvariantViolation(
                    invariant_name=self.name,
                    invariant_type=self.invariant_type,
                    violation_message=f"Topic drift detected: conversation moving away from allowed topics",
                    turn_id=turn.turn_id,
                    timestamp=turn.timestamp,
                    severity=self.severity,
                    expected_state=f"Related to: {', '.join(self.allowed_topics)}",
                    actual_state="Off-topic content",
                    context={
                        "allowed_topics": self.allowed_topics,
                        "drift_score": self.topic_drift_score
                    },
                    auto_fix_suggestion="Steer conversation back to allowed topics",
                    confidence=self.topic_drift_score
                )
                
                self.violations.append(violation)
                self.violation_count += 1
                return violation
        else:
            # Reset drift score if we're back on topic
            self.topic_drift_score = max(0, self.topic_drift_score - 0.05)
        
        return None
    
    def _detect_topics(self, content: str, patterns: List[Pattern]) -> List[str]:
        """Detect topics in content using patterns."""
        detected = []
        for i, pattern in enumerate(patterns):
            if pattern.search(content):
                if i < len(self.allowed_topics):
                    detected.append(self.allowed_topics[i])
                elif i - len(self.allowed_topics) < len(self.forbidden_topics):
                    detected.append(self.forbidden_topics[i - len(self.allowed_topics)])
        return detected


class MemoryInvariant(ConversationInvariant):
    """Ensures consistency with conversation memory and context."""
    
    def __init__(self, 
                 name: str = "memory_consistency",
                 memory_window_size: int = 10,
                 reference_detection_enabled: bool = True):
        super().__init__(
            name=name,
            description="Maintains consistency with conversation memory",
            invariant_type=InvariantType.MEMORY,
            severity=InvariantSeverity.MEDIUM
        )
        
        self.memory_window_size = memory_window_size
        self.reference_detection_enabled = reference_detection_enabled
        
        # Memory tracking
        self.mentioned_entities: Set[str] = set()
        self.entity_contexts: Dict[str, List[str]] = {}
        
        # Reference patterns
        self.reference_patterns = [
            re.compile(r'\b(that|this|it|they|them|those|these)\b', re.IGNORECASE),
            re.compile(r'\b(as I mentioned|like I said|previously|earlier|before)\b', re.IGNORECASE),
        ]
    
    def check_turn(self, turn: Any, conversation_state: Any) -> Optional[InvariantViolation]:
        """Check memory consistency for a turn."""
        if not self.enabled:
            return None
        
        self.check_count += 1
        self.last_check_time = time.time()
        
        # Check for unclear references
        if self.reference_detection_enabled:
            unclear_refs = self._detect_unclear_references(turn, conversation_state)
            if unclear_refs:
                violation = InvariantViolation(
                    invariant_name=self.name,
                    invariant_type=self.invariant_type,
                    violation_message=f"Unclear reference(s) detected: {', '.join(unclear_refs)}",
                    turn_id=turn.turn_id,
                    timestamp=turn.timestamp,
                    severity=self.severity,
                    expected_state="Clear references to previous context",
                    actual_state=f"Unclear: {', '.join(unclear_refs)}",
                    context={
                        "unclear_references": unclear_refs,
                        "recent_entities": list(self.mentioned_entities)
                    },
                    auto_fix_suggestion="Clarify references to previous conversation elements",
                    confidence=0.7
                )
                
                self.violations.append(violation)
                self.violation_count += 1
                return violation
        
        # Update entity tracking
        self._update_entity_tracking(turn)
        
        return None
    
    def _detect_unclear_references(self, turn: Any, conversation_state: Any) -> List[str]:
        """Detect unclear references in the turn."""
        unclear_refs = []
        
        # Get recent context
        recent_turns = conversation_state.get_recent_turns(self.memory_window_size)
        recent_content = " ".join([t.content for t in recent_turns])
        
        for pattern in self.reference_patterns:
            matches = pattern.findall(turn.content)
            for match in matches:
                # Check if the reference has clear antecedent in recent context
                if not self._has_clear_antecedent(match, recent_content):
                    unclear_refs.append(match)
        
        return unclear_refs
    
    def _has_clear_antecedent(self, reference: str, context: str) -> bool:
        """Check if a reference has a clear antecedent in context."""
        # Simple heuristic: if the context mentions specific entities,
        # pronouns are more likely to be clear
        
        # Count specific nouns/entities in context
        import re
        entities = re.findall(r'\b[A-Z][a-z]+\b', context)  # Capitalized words
        nouns = re.findall(r'\b\w+(?:ing|ed|tion|ness|ity|ment)\b', context)  # Common noun patterns
        
        entity_count = len(set(entities + nouns))
        
        # References are clearer when there are fewer potential antecedents
        return entity_count <= 3
    
    def _update_entity_tracking(self, turn: Any) -> None:
        """Update tracking of mentioned entities."""
        import re
        
        # Extract potential entities (capitalized words, specific patterns)
        entities = re.findall(r'\b[A-Z][a-z]+\b', turn.content)
        
        for entity in entities:
            self.mentioned_entities.add(entity)
            if entity not in self.entity_contexts:
                self.entity_contexts[entity] = []
            self.entity_contexts[entity].append(turn.content)


class InvariantTracker:
    """Tracks multiple conversation invariants across turns."""
    
    def __init__(self):
        self.invariants: List[ConversationInvariant] = []
        self.global_violations: List[InvariantViolation] = []
        self.tracking_metrics = {
            "turns_processed": 0,
            "invariants_checked": 0,
            "violations_detected": 0,
            "auto_fixes_suggested": 0,
        }
    
    def add_invariant(self, invariant: ConversationInvariant) -> None:
        """Add an invariant to track."""
        self.invariants.append(invariant)
        logger.info(f"Added invariant: {invariant.name}")
    
    def check_turn(self, turn: Any, conversation_state: Any) -> List[InvariantViolation]:
        """Check all invariants for a turn."""
        violations = []
        self.tracking_metrics["turns_processed"] += 1
        
        for invariant in self.invariants:
            if invariant.enabled:
                try:
                    violation = invariant.check_turn(turn, conversation_state)
                    self.tracking_metrics["invariants_checked"] += 1
                    
                    if violation:
                        violations.append(violation)
                        self.global_violations.append(violation)
                        self.tracking_metrics["violations_detected"] += 1
                        
                        if violation.auto_fix_suggestion:
                            self.tracking_metrics["auto_fixes_suggested"] += 1
                    
                    # Update invariant state
                    invariant.update_state(turn, conversation_state)
                    
                except Exception as e:
                    logger.error(f"Error checking invariant {invariant.name}: {e}")
        
        return violations
    
    def get_invariant_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all invariants."""
        return {
            invariant.name: invariant.get_status()
            for invariant in self.invariants
        }
    
    def get_violations(self, 
                      invariant_name: Optional[str] = None,
                      severity: Optional[InvariantSeverity] = None,
                      recent_only: bool = False) -> List[InvariantViolation]:
        """Get violations with optional filtering."""
        violations = self.global_violations
        
        if invariant_name:
            violations = [v for v in violations if v.invariant_name == invariant_name]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        if recent_only:
            cutoff_time = time.time() - 300  # Last 5 minutes
            violations = [v for v in violations if v.timestamp > cutoff_time]
        
        return violations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of invariant tracking."""
        return {
            "tracking_metrics": self.tracking_metrics,
            "total_invariants": len(self.invariants),
            "enabled_invariants": len([i for i in self.invariants if i.enabled]),
            "total_violations": len(self.global_violations),
            "recent_violations": len([v for v in self.global_violations if time.time() - v.timestamp < 300]),
            "violations_by_severity": {
                severity.value: len([v for v in self.global_violations if v.severity == severity])
                for severity in InvariantSeverity
            },
            "violations_by_type": {
                inv_type.value: len([v for v in self.global_violations if v.invariant_type == inv_type])
                for inv_type in InvariantType
            }
        }
    
    def reset_all(self) -> None:
        """Reset all invariants and tracking state."""
        for invariant in self.invariants:
            invariant.reset()
        
        self.global_violations.clear()
        self.tracking_metrics = {
            "turns_processed": 0,
            "invariants_checked": 0,
            "violations_detected": 0,
            "auto_fixes_suggested": 0,
        }
    
    def export_violations(self, format: str = "json") -> str:
        """Export violations to various formats."""
        if format.lower() == "json":
            return json.dumps([v.to_dict() for v in self.global_violations], indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Convenience functions for creating common invariants

def create_personality_invariant(personality_type: str = "helpful") -> PersonalityInvariant:
    """Create a personality invariant for common personality types."""
    personality_configs = {
        "helpful": {
            "traits": {"helpful": "high", "friendly": "medium", "professional": "medium"},
            "description": "Helpful, friendly, and professional assistant"
        },
        "formal": {
            "traits": {"formal": "high", "professional": "high", "casual": "low"},
            "description": "Formal and professional communication style"
        },
        "casual": {
            "traits": {"casual": "high", "friendly": "high", "formal": "low"},
            "description": "Casual and friendly communication style"
        }
    }
    
    config = personality_configs.get(personality_type, personality_configs["helpful"])
    
    return PersonalityInvariant(
        name=f"personality_{personality_type}",
        personality_traits=config["traits"],
        personality_description=config["description"]
    )

def create_topic_boundary_invariant(allowed_topics: List[str], forbidden_topics: Optional[List[str]] = None) -> TopicBoundaryInvariant:
    """Create a topic boundary invariant."""
    return TopicBoundaryInvariant(
        name="topic_boundaries",
        allowed_topics=allowed_topics,
        forbidden_topics=forbidden_topics or []
    )

def create_professional_tone_invariant() -> ToneInvariant:
    """Create a professional tone invariant."""
    return ToneInvariant(
        name="professional_tone",
        target_tone="professional"
    )