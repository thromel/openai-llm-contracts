"""Temporal contract system for conversation validation.

This module implements temporal logic contracts that can enforce properties
over time across multiple turns in a conversation.
"""

import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Set
from enum import Enum, auto
import logging

from ..contracts.base import ContractBase, ValidationResult

logger = logging.getLogger(__name__)


class TemporalOperator(Enum):
    """Temporal logic operators."""
    ALWAYS = "always"          # □ - Property must always hold
    EVENTUALLY = "eventually"   # ◇ - Property must eventually hold
    NEXT = "next"              # X - Property must hold in next state
    UNTIL = "until"            # U - Property A holds until property B holds
    SINCE = "since"            # S - Property A has held since property B held
    WITHIN = "within"          # Property must hold within N turns/time


@dataclass
class TemporalViolation:
    """Represents a temporal contract violation."""
    contract_name: str
    operator: TemporalOperator
    message: str
    turn_id: str
    timestamp: float
    expected_condition: str
    actual_state: str
    severity: str = "medium"
    auto_fix_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalContract(ContractBase):
    """Base class for temporal contracts."""
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 operator: TemporalOperator,
                 condition: Union[str, Callable],
                 window_size: Optional[int] = None,
                 time_window_seconds: Optional[float] = None):
        """Initialize temporal contract.
        
        Args:
            name: Contract name
            description: Contract description  
            operator: Temporal operator to use
            condition: Condition to evaluate (string pattern or callable)
            window_size: Number of turns to consider (for bounded operators)
            time_window_seconds: Time window in seconds (for time-bounded operators)
        """
        super().__init__(name, description)
        self.operator = operator
        self.condition = condition
        self.window_size = window_size
        self.time_window_seconds = time_window_seconds
        
        # State tracking
        self.evaluation_history: List[Dict[str, Any]] = []
        self.violations: List[TemporalViolation] = []
        self.last_satisfied_turn: Optional[str] = None
        self.last_satisfied_time: Optional[float] = None
        
        # Configuration
        self.auto_reset = True  # Reset state when condition is satisfied
        self.strict_mode = False  # Strict evaluation vs. lenient
        
    def validate_turn(self, turn: Any, conversation_state: Any) -> ValidationResult:
        """Validate a single turn against the temporal contract."""
        try:
            # Evaluate condition for current turn
            current_satisfaction = self._evaluate_condition(turn, conversation_state)
            
            # Record evaluation
            self._record_evaluation(turn, current_satisfaction, conversation_state)
            
            # Check temporal property
            violation = self._check_temporal_property(turn, conversation_state, current_satisfaction)
            
            if violation:
                self.violations.append(violation)
                return ValidationResult(
                    is_valid=False,
                    message=violation.message,
                    auto_fix_suggestion=violation.auto_fix_suggestion
                )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"Error evaluating temporal contract {self.name}: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"Temporal contract evaluation error: {str(e)}"
            )
    
    def _evaluate_condition(self, turn: Any, conversation_state: Any) -> bool:
        """Evaluate the condition for the current turn."""
        if isinstance(self.condition, str):
            # String pattern matching
            return bool(re.search(self.condition, turn.content, re.IGNORECASE))
        elif callable(self.condition):
            # Callable condition
            return bool(self.condition(turn, conversation_state))
        else:
            return False
    
    def _record_evaluation(self, turn: Any, satisfied: bool, conversation_state: Any) -> None:
        """Record the evaluation result."""
        evaluation = {
            "turn_id": turn.turn_id,
            "timestamp": turn.timestamp,
            "satisfied": satisfied,
            "turn_content": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content,
            "conversation_turn_count": conversation_state.turn_count
        }
        
        self.evaluation_history.append(evaluation)
        
        if satisfied:
            self.last_satisfied_turn = turn.turn_id
            self.last_satisfied_time = turn.timestamp
    
    def _check_temporal_property(self, turn: Any, conversation_state: Any, current_satisfaction: bool) -> Optional[TemporalViolation]:
        """Check if the temporal property is violated."""
        # This is overridden by specific temporal contract implementations
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the temporal contract."""
        return {
            "name": self.name,
            "operator": self.operator.value,
            "evaluations": len(self.evaluation_history),
            "violations": len(self.violations),
            "last_satisfied_turn": self.last_satisfied_turn,
            "last_satisfied_time": self.last_satisfied_time,
            "currently_satisfied": self.evaluation_history[-1]["satisfied"] if self.evaluation_history else None
        }
    
    def reset(self) -> None:
        """Reset contract state."""
        self.evaluation_history.clear()
        self.violations.clear()
        self.last_satisfied_turn = None
        self.last_satisfied_time = None


class AlwaysContract(TemporalContract):
    """Contract that enforces a property must always hold (□P)."""
    
    def __init__(self, name: str, condition: Union[str, Callable], description: str = ""):
        super().__init__(
            name=name,
            description=description or f"Property must always hold: {condition}",
            operator=TemporalOperator.ALWAYS,
            condition=condition
        )
    
    def _check_temporal_property(self, turn: Any, conversation_state: Any, current_satisfaction: bool) -> Optional[TemporalViolation]:
        """For ALWAYS, any false evaluation is a violation."""
        if not current_satisfaction:
            return TemporalViolation(
                contract_name=self.name,
                operator=self.operator,
                message=f"ALWAYS property violated: condition failed at turn {turn.turn_id}",
                turn_id=turn.turn_id,
                timestamp=turn.timestamp,
                expected_condition=str(self.condition),
                actual_state=f"Condition not satisfied in: {turn.content[:100]}",
                severity="high"
            )
        return None


class EventuallyContract(TemporalContract):
    """Contract that enforces a property must eventually hold (◇P)."""
    
    def __init__(self, 
                 name: str, 
                 condition: Union[str, Callable], 
                 window_size: int = 10,
                 description: str = ""):
        super().__init__(
            name=name,
            description=description or f"Property must eventually hold: {condition}",
            operator=TemporalOperator.EVENTUALLY,
            condition=condition,
            window_size=window_size
        )
        self.satisfaction_required = True  # Start requiring satisfaction
    
    def _check_temporal_property(self, turn: Any, conversation_state: Any, current_satisfaction: bool) -> Optional[TemporalViolation]:
        """For EVENTUALLY, check if condition is satisfied within window."""
        if current_satisfaction:
            # Condition satisfied, reset requirement
            self.satisfaction_required = False
            return None
        
        # Check if we've exceeded the window without satisfaction
        if len(self.evaluation_history) >= self.window_size:
            recent_evaluations = self.evaluation_history[-self.window_size:]
            if not any(eval_["satisfied"] for eval_ in recent_evaluations):
                return TemporalViolation(
                    contract_name=self.name,
                    operator=self.operator,
                    message=f"EVENTUALLY property violated: condition not satisfied within {self.window_size} turns",
                    turn_id=turn.turn_id,
                    timestamp=turn.timestamp,
                    expected_condition=str(self.condition),
                    actual_state=f"Not satisfied in last {self.window_size} turns",
                    severity="medium",
                    auto_fix_suggestion=f"Consider addressing: {str(self.condition)}"
                )
        
        return None


class NextContract(TemporalContract):
    """Contract that enforces a property must hold in the next turn (XP)."""
    
    def __init__(self, 
                 name: str, 
                 trigger_condition: Union[str, Callable],
                 next_condition: Union[str, Callable],
                 description: str = ""):
        super().__init__(
            name=name,
            description=description or f"After {trigger_condition}, next turn must satisfy {next_condition}",
            operator=TemporalOperator.NEXT,
            condition=trigger_condition
        )
        self.next_condition = next_condition
        self.trigger_turn_id = None
        self.expecting_next = False
    
    def _check_temporal_property(self, turn: Any, conversation_state: Any, current_satisfaction: bool) -> Optional[TemporalViolation]:
        """For NEXT, check if next condition holds after trigger."""
        if self.expecting_next:
            # We're in the "next" turn after trigger
            next_satisfied = self._evaluate_next_condition(turn, conversation_state)
            self.expecting_next = False  # Reset state
            
            if not next_satisfied:
                return TemporalViolation(
                    contract_name=self.name,
                    operator=self.operator,
                    message=f"NEXT property violated: expected condition not met in turn following {self.trigger_turn_id}",
                    turn_id=turn.turn_id,
                    timestamp=turn.timestamp,
                    expected_condition=str(self.next_condition),
                    actual_state=f"Next condition not satisfied: {turn.content[:100]}",
                    severity="medium"
                )
        
        elif current_satisfaction:
            # Trigger condition satisfied, expect next condition in next turn
            self.trigger_turn_id = turn.turn_id
            self.expecting_next = True
        
        return None
    
    def _evaluate_next_condition(self, turn: Any, conversation_state: Any) -> bool:
        """Evaluate the next condition."""
        if isinstance(self.next_condition, str):
            return bool(re.search(self.next_condition, turn.content, re.IGNORECASE))
        elif callable(self.next_condition):
            return bool(self.next_condition(turn, conversation_state))
        return False


class WithinContract(TemporalContract):
    """Contract that enforces a property must hold within N turns or time."""
    
    def __init__(self, 
                 name: str, 
                 condition: Union[str, Callable],
                 window_size: Optional[int] = None,
                 time_window_seconds: Optional[float] = None,
                 description: str = ""):
        if not window_size and not time_window_seconds:
            raise ValueError("Must specify either window_size or time_window_seconds")
            
        super().__init__(
            name=name,
            description=description or f"Property must hold within specified window: {condition}",
            operator=TemporalOperator.WITHIN,
            condition=condition,
            window_size=window_size,
            time_window_seconds=time_window_seconds
        )
        self.start_time = time.time()
        self.start_turn = 0
    
    def _check_temporal_property(self, turn: Any, conversation_state: Any, current_satisfaction: bool) -> Optional[TemporalViolation]:
        """For WITHIN, check if condition is satisfied within the specified window."""
        if current_satisfaction:
            return None  # Condition satisfied
        
        # Check turn-based window
        if self.window_size:
            turns_elapsed = conversation_state.turn_count - self.start_turn
            if turns_elapsed >= self.window_size:
                return TemporalViolation(
                    contract_name=self.name,
                    operator=self.operator,
                    message=f"WITHIN property violated: condition not satisfied within {self.window_size} turns",
                    turn_id=turn.turn_id,
                    timestamp=turn.timestamp,
                    expected_condition=str(self.condition),
                    actual_state=f"Not satisfied within {turns_elapsed} turns",
                    severity="medium"
                )
        
        # Check time-based window
        if self.time_window_seconds:
            time_elapsed = time.time() - self.start_time
            if time_elapsed >= self.time_window_seconds:
                return TemporalViolation(
                    contract_name=self.name,
                    operator=self.operator,
                    message=f"WITHIN property violated: condition not satisfied within {self.time_window_seconds} seconds",
                    turn_id=turn.turn_id,
                    timestamp=turn.timestamp,
                    expected_condition=str(self.condition),
                    actual_state=f"Not satisfied within {time_elapsed:.1f} seconds",
                    severity="medium"
                )
        
        return None


class UntilContract(TemporalContract):
    """Contract that enforces property A holds until property B holds (A U B)."""
    
    def __init__(self, 
                 name: str,
                 condition_a: Union[str, Callable],
                 condition_b: Union[str, Callable],
                 description: str = ""):
        super().__init__(
            name=name,
            description=description or f"Condition A must hold until condition B: {condition_a} U {condition_b}",
            operator=TemporalOperator.UNTIL,
            condition=condition_a
        )
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.condition_b_satisfied = False
    
    def _check_temporal_property(self, turn: Any, conversation_state: Any, current_satisfaction: bool) -> Optional[TemporalViolation]:
        """For UNTIL, check that A holds until B is satisfied."""
        # Check if condition B is satisfied
        condition_b_satisfied = self._evaluate_condition_b(turn, conversation_state)
        
        if condition_b_satisfied:
            self.condition_b_satisfied = True
            return None  # B satisfied, UNTIL contract fulfilled
        
        if self.condition_b_satisfied:
            return None  # B was already satisfied, contract fulfilled
        
        # B not satisfied yet, so A must hold
        if not current_satisfaction:
            return TemporalViolation(
                contract_name=self.name,
                operator=self.operator,
                message=f"UNTIL property violated: condition A failed before condition B was satisfied",
                turn_id=turn.turn_id,
                timestamp=turn.timestamp,
                expected_condition=f"A: {self.condition_a} until B: {self.condition_b}",
                actual_state=f"Condition A not satisfied: {turn.content[:100]}",
                severity="medium"
            )
        
        return None
    
    def _evaluate_condition_b(self, turn: Any, conversation_state: Any) -> bool:
        """Evaluate condition B."""
        if isinstance(self.condition_b, str):
            return bool(re.search(self.condition_b, turn.content, re.IGNORECASE))
        elif callable(self.condition_b):
            return bool(self.condition_b(turn, conversation_state))
        return False


class SinceContract(TemporalContract):
    """Contract that enforces property A has held since property B held (A S B)."""
    
    def __init__(self, 
                 name: str,
                 condition_a: Union[str, Callable],
                 condition_b: Union[str, Callable],
                 description: str = ""):
        super().__init__(
            name=name,
            description=description or f"Condition A must hold since condition B: {condition_a} S {condition_b}",
            operator=TemporalOperator.SINCE,
            condition=condition_a
        )
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.condition_b_occurred = False
        self.b_occurrence_turn = None
    
    def _check_temporal_property(self, turn: Any, conversation_state: Any, current_satisfaction: bool) -> Optional[TemporalViolation]:
        """For SINCE, check that A has held since B occurred."""
        # Check if condition B occurs
        condition_b_satisfied = self._evaluate_condition_b(turn, conversation_state)
        
        if condition_b_satisfied:
            self.condition_b_occurred = True
            self.b_occurrence_turn = turn.turn_id
        
        # If B has never occurred, SINCE is trivially satisfied
        if not self.condition_b_occurred:
            return None
        
        # B has occurred, so A must hold from that point forward
        if not current_satisfaction:
            return TemporalViolation(
                contract_name=self.name,
                operator=self.operator,
                message=f"SINCE property violated: condition A failed since condition B occurred at turn {self.b_occurrence_turn}",
                turn_id=turn.turn_id,
                timestamp=turn.timestamp,
                expected_condition=f"A: {self.condition_a} since B: {self.condition_b}",
                actual_state=f"Condition A not satisfied: {turn.content[:100]}",
                severity="medium"
            )
        
        return None
    
    def _evaluate_condition_b(self, turn: Any, conversation_state: Any) -> bool:
        """Evaluate condition B."""
        if isinstance(self.condition_b, str):
            return bool(re.search(self.condition_b, turn.content, re.IGNORECASE))
        elif callable(self.condition_b):
            return bool(self.condition_b(turn, conversation_state))
        return False


class TemporalValidator:
    """Validates multiple temporal contracts across conversation turns."""
    
    def __init__(self):
        self.contracts: List[TemporalContract] = []
        self.global_violations: List[TemporalViolation] = []
        
    def add_contract(self, contract: TemporalContract) -> None:
        """Add a temporal contract."""
        self.contracts.append(contract)
        logger.info(f"Added temporal contract: {contract.name}")
    
    def validate_turn(self, turn: Any, conversation_state: Any) -> List[ValidationResult]:
        """Validate a turn against all temporal contracts."""
        results = []
        
        for contract in self.contracts:
            try:
                result = contract.validate_turn(turn, conversation_state)
                results.append(result)
                
                # Track global violations
                if not result.is_valid and contract.violations:
                    latest_violation = contract.violations[-1]
                    self.global_violations.append(latest_violation)
                    
            except Exception as e:
                logger.error(f"Error validating temporal contract {contract.name}: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Temporal validation error: {str(e)}"
                ))
        
        return results
    
    def get_contract_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all contracts."""
        return {
            contract.name: contract.get_status()
            for contract in self.contracts
        }
    
    def get_violations(self, contract_name: Optional[str] = None) -> List[TemporalViolation]:
        """Get violations, optionally filtered by contract name."""
        if contract_name:
            return [v for v in self.global_violations if v.contract_name == contract_name]
        return self.global_violations
    
    def reset_all_contracts(self) -> None:
        """Reset all contracts."""
        for contract in self.contracts:
            contract.reset()
        self.global_violations.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of temporal validation."""
        total_evaluations = sum(len(c.evaluation_history) for c in self.contracts)
        total_violations = len(self.global_violations)
        
        return {
            "total_contracts": len(self.contracts),
            "total_evaluations": total_evaluations,
            "total_violations": total_violations,
            "contracts": [
                {
                    "name": contract.name,
                    "operator": contract.operator.value,
                    "evaluations": len(contract.evaluation_history),
                    "violations": len(contract.violations),
                    "status": "satisfied" if not contract.violations else "violated"
                }
                for contract in self.contracts
            ]
        }


# Convenience functions for creating common temporal contracts

def create_always_contract(name: str, pattern: str, description: str = "") -> AlwaysContract:
    """Create an ALWAYS contract with pattern matching."""
    return AlwaysContract(name, pattern, description)

def create_eventually_contract(name: str, pattern: str, window_size: int = 10, description: str = "") -> EventuallyContract:
    """Create an EVENTUALLY contract with pattern matching."""
    return EventuallyContract(name, pattern, window_size, description)

def create_politeness_contract() -> AlwaysContract:
    """Create a contract ensuring politeness in all responses."""
    def is_polite(turn, conversation_state):
        impolite_patterns = [
            r'\bstupid\b', r'\bidiot\b', r'\bshut up\b', 
            r'\bgo away\b', r'\byou\'re wrong\b'
        ]
        content_lower = turn.content.lower()
        return not any(re.search(pattern, content_lower) for pattern in impolite_patterns)
    
    return AlwaysContract(
        name="politeness_always",
        condition=is_polite,
        description="Responses must always be polite and respectful"
    )

def create_topic_consistency_contract(topic: str, window_size: int = 5) -> EventuallyContract:
    """Create a contract ensuring topic relevance."""
    return EventuallyContract(
        name=f"topic_consistency_{topic}",
        condition=lambda turn, state: topic.lower() in turn.content.lower(),
        window_size=window_size,
        description=f"Conversation must stay relevant to topic: {topic}"
    )

def create_question_answer_contract() -> NextContract:
    """Create a contract ensuring questions are followed by answers."""
    return NextContract(
        name="question_answer_flow",
        trigger_condition=lambda turn, state: turn.role.value == "user" and "?" in turn.content,
        next_condition=lambda turn, state: turn.role.value == "assistant",
        description="User questions must be followed by assistant responses"
    )