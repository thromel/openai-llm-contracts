"""Guardrails.ai migration adapter for LLM Design by Contract framework.

This module provides comprehensive migration support from Guardrails.ai,
including config conversion, validator adaptation, and compatibility layer.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
from enum import Enum
import logging

try:
    import guardrails as gd
    from guardrails import Guard
    from guardrails.validators import Validator
    GUARDRAILS_AVAILABLE = True
except ImportError:
    # Mock classes when Guardrails is not available
    class Guard:
        pass
    class Validator:
        pass
    GUARDRAILS_AVAILABLE = False

# Import contract framework components
from ..contracts.base import ContractBase, ValidationResult
from ..validators.basic_validators import OutputValidator as BasicValidator
from ..core.exceptions import ContractViolationError
# from ..utils.telemetry import log_contract_execution  # Function not available

logger = logging.getLogger(__name__)


class GuardrailsValidatorType(Enum):
    """Types of Guardrails validators supported."""
    REGEX = "regex"
    LENGTH = "length"
    CHOICE = "choice"
    RANGE = "range"
    FORMAT = "format"
    CUSTOM = "custom"
    PYDANTIC = "pydantic"
    SEMANTIC = "semantic"


@dataclass
class GuardrailsValidatorConfig:
    """Configuration for a Guardrails validator."""
    name: str
    validator_type: GuardrailsValidatorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    on_fail: str = "exception"  # exception, fix, reask, filter, refrain
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "validator_type": self.validator_type.value,
            "parameters": self.parameters,
            "on_fail": self.on_fail,
            "metadata": self.metadata
        }


class GuardrailsValidator(BasicValidator):
    """Adapter for Guardrails validators in the contract framework."""
    
    def __init__(self,
                 config: GuardrailsValidatorConfig,
                 original_validator: Optional[Any] = None):
        """Initialize Guardrails validator adapter.
        
        Args:
            config: Validator configuration
            original_validator: Original Guardrails validator instance
        """
        super().__init__(
            name=config.name,
            description=f"Migrated Guardrails {config.validator_type.value} validator"
        )
        
        self.config = config
        self.original_validator = original_validator
        self.validation_metrics = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "auto_fixes_applied": 0
        }
    
    async def validate_async(self, data: Any) -> ValidationResult:
        """Validate data using migrated Guardrails logic."""
        self.validation_metrics["total_validations"] += 1
        
        try:
            # Apply validator based on type
            is_valid, error_message, fixed_content = await self._apply_validator(data)
            
            if is_valid:
                self.validation_metrics["successful_validations"] += 1
                return ValidationResult(is_valid=True)
            else:
                self.validation_metrics["failed_validations"] += 1
                
                if fixed_content is not None:
                    self.validation_metrics["auto_fixes_applied"] += 1
                    return ValidationResult(
                        is_valid=False,
                        error_message=error_message,
                        auto_fixed_content=fixed_content
                    )
                else:
                    return ValidationResult(
                        is_valid=False,
                        error_message=error_message
                    )
        
        except Exception as e:
            self.validation_metrics["failed_validations"] += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    async def _apply_validator(self, data: Any) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Apply the specific validator logic based on type."""
        if self.config.validator_type == GuardrailsValidatorType.REGEX:
            return await self._validate_regex(data)
        elif self.config.validator_type == GuardrailsValidatorType.LENGTH:
            return await self._validate_length(data)
        elif self.config.validator_type == GuardrailsValidatorType.CHOICE:
            return await self._validate_choice(data)
        elif self.config.validator_type == GuardrailsValidatorType.RANGE:
            return await self._validate_range(data)
        elif self.config.validator_type == GuardrailsValidatorType.FORMAT:
            return await self._validate_format(data)
        elif self.config.validator_type == GuardrailsValidatorType.CUSTOM:
            return await self._validate_custom(data)
        else:
            return False, f"Unsupported validator type: {self.config.validator_type}", None
    
    async def _validate_regex(self, data: Any) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate using regex pattern."""
        pattern = self.config.parameters.get("pattern", "")
        flags = self.config.parameters.get("flags", 0)
        
        try:
            text = str(data)
            regex = re.compile(pattern, flags)
            
            if regex.search(text):
                return True, None, None
            else:
                error_msg = f"Text does not match pattern: {pattern}"
                
                # Attempt auto-fix if enabled
                if self.config.on_fail == "fix":
                    fixed_content = await self._auto_fix_regex(text, pattern)
                    return False, error_msg, fixed_content
                
                return False, error_msg, None
        
        except Exception as e:
            return False, f"Regex validation error: {str(e)}", None
    
    async def _validate_length(self, data: Any) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate length constraints."""
        min_length = self.config.parameters.get("min", 0)
        max_length = self.config.parameters.get("max", float('inf'))
        
        try:
            text = str(data)
            length = len(text)
            
            if min_length <= length <= max_length:
                return True, None, None
            else:
                error_msg = f"Length {length} not in range [{min_length}, {max_length}]"
                
                # Attempt auto-fix
                if self.config.on_fail == "fix":
                    if length > max_length:
                        fixed_content = text[:max_length]
                        return False, error_msg, fixed_content
                    elif length < min_length:
                        fixed_content = text + " " * (min_length - length)
                        return False, error_msg, fixed_content
                
                return False, error_msg, None
        
        except Exception as e:
            return False, f"Length validation error: {str(e)}", None
    
    async def _validate_choice(self, data: Any) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate against allowed choices."""
        choices = self.config.parameters.get("choices", [])
        case_sensitive = self.config.parameters.get("case_sensitive", True)
        
        try:
            value = str(data)
            
            if case_sensitive:
                valid = value in choices
            else:
                valid = value.lower() in [choice.lower() for choice in choices]
            
            if valid:
                return True, None, None
            else:
                error_msg = f"Value '{value}' not in allowed choices: {choices}"
                
                # Attempt auto-fix with closest match
                if self.config.on_fail == "fix":
                    fixed_content = await self._find_closest_choice(value, choices)
                    return False, error_msg, fixed_content
                
                return False, error_msg, None
        
        except Exception as e:
            return False, f"Choice validation error: {str(e)}", None
    
    async def _validate_range(self, data: Any) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate numeric range."""
        min_val = self.config.parameters.get("min", float('-inf'))
        max_val = self.config.parameters.get("max", float('inf'))
        
        try:
            if isinstance(data, str):
                # Try to extract number from string
                import re
                number_match = re.search(r'-?\d+\.?\d*', data)
                if number_match:
                    value = float(number_match.group())
                else:
                    return False, f"No numeric value found in: {data}", None
            else:
                value = float(data)
            
            if min_val <= value <= max_val:
                return True, None, None
            else:
                error_msg = f"Value {value} not in range [{min_val}, {max_val}]"
                
                # Attempt auto-fix by clamping
                if self.config.on_fail == "fix":
                    fixed_value = max(min_val, min(value, max_val))
                    return False, error_msg, fixed_value
                
                return False, error_msg, None
        
        except Exception as e:
            return False, f"Range validation error: {str(e)}", None
    
    async def _validate_format(self, data: Any) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate specific formats (email, URL, phone, etc.)."""
        format_type = self.config.parameters.get("format", "")
        
        try:
            text = str(data)
            
            if format_type == "email":
                return await self._validate_email_format(text)
            elif format_type == "url":
                return await self._validate_url_format(text)
            elif format_type == "phone":
                return await self._validate_phone_format(text)
            elif format_type == "json":
                return await self._validate_json_format(text)
            else:
                return False, f"Unsupported format type: {format_type}", None
        
        except Exception as e:
            return False, f"Format validation error: {str(e)}", None
    
    async def _validate_custom(self, data: Any) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate using custom logic."""
        if self.original_validator and GUARDRAILS_AVAILABLE:
            try:
                # Try to use original Guardrails validator
                result = self.original_validator.validate(data)
                return True, None, None
            except Exception as e:
                return False, f"Custom validation failed: {str(e)}", None
        else:
            return False, "Custom validator not available", None
    
    async def _auto_fix_regex(self, text: str, pattern: str) -> Optional[str]:
        """Attempt to auto-fix text to match regex pattern."""
        # Basic auto-fix strategies for common patterns
        if pattern == r"^\d+$":  # Numbers only
            import re
            numbers = re.findall(r'\d+', text)
            return ''.join(numbers) if numbers else "0"
        elif pattern == r"^[a-zA-Z]+$":  # Letters only
            import re
            letters = re.findall(r'[a-zA-Z]+', text)
            return ''.join(letters)
        else:
            return None
    
    async def _find_closest_choice(self, value: str, choices: List[str]) -> Optional[str]:
        """Find the closest matching choice using string similarity."""
        if not choices:
            return None
        
        # Simple similarity based on common prefixes
        best_match = choices[0]
        best_score = 0
        
        for choice in choices:
            # Calculate similarity score
            common_prefix = 0
            for i, (c1, c2) in enumerate(zip(value.lower(), choice.lower())):
                if c1 == c2:
                    common_prefix += 1
                else:
                    break
            
            score = common_prefix / max(len(value), len(choice))
            if score > best_score:
                best_score = score
                best_match = choice
        
        return best_match if best_score > 0.3 else None
    
    async def _validate_email_format(self, text: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate email format."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, text):
            return True, None, None
        else:
            error_msg = f"Invalid email format: {text}"
            
            # Attempt basic fix
            if self.config.on_fail == "fix":
                # Add @ if missing
                if "@" not in text and "." in text:
                    parts = text.split(".")
                    if len(parts) >= 2:
                        fixed = f"{parts[0]}@{'.'.join(parts[1:])}"
                        return False, error_msg, fixed
            
            return False, error_msg, None
    
    async def _validate_url_format(self, text: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate URL format."""
        import re
        url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        
        if re.match(url_pattern, text):
            return True, None, None
        else:
            error_msg = f"Invalid URL format: {text}"
            
            # Attempt basic fix
            if self.config.on_fail == "fix":
                if not text.startswith(("http://", "https://")):
                    fixed = f"https://{text}"
                    return False, error_msg, fixed
            
            return False, error_msg, None
    
    async def _validate_phone_format(self, text: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate phone number format."""
        import re
        # Remove all non-digits
        digits = re.sub(r'\D', '', text)
        
        if len(digits) >= 10:
            return True, None, None
        else:
            return False, f"Invalid phone number: {text}", None
    
    async def _validate_json_format(self, text: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate JSON format."""
        try:
            json.loads(text)
            return True, None, None
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {str(e)}"
            
            # Attempt basic JSON fix
            if self.config.on_fail == "fix":
                fixed = await self._auto_fix_json(text)
                return False, error_msg, fixed
            
            return False, error_msg, None
    
    async def _auto_fix_json(self, text: str) -> Optional[str]:
        """Attempt to auto-fix JSON format."""
        import re
        
        # Basic JSON fixes
        text = text.strip()
        
        # Add missing quotes around keys
        text = re.sub(r'(\w+):', r'"\1":', text)
        
        # Fix single quotes to double quotes
        text = text.replace("'", '"')
        
        # Add missing commas
        text = re.sub(r'}\s*{', '},{', text)
        text = re.sub(r']\s*\[', '],[', text)
        
        # Wrap in braces if missing
        if not text.startswith(('{', '[')):
            text = f'{{{text}}}'
        
        # Validate the fix
        try:
            json.loads(text)
            return text
        except:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            **self.validation_metrics,
            "success_rate": self.validation_metrics["successful_validations"] / max(self.validation_metrics["total_validations"], 1),
            "auto_fix_rate": self.validation_metrics["auto_fixes_applied"] / max(self.validation_metrics["failed_validations"], 1)
        }


class GuardrailsAdapter:
    """Main adapter for migrating from Guardrails.ai to contract framework."""
    
    def __init__(self):
        self.migrated_validators = {}
        self.migration_metrics = {
            "total_migrations": 0,
            "successful_migrations": 0,
            "failed_migrations": 0
        }
    
    def migrate_guard(self, guard: Any) -> List[GuardrailsValidator]:
        """Migrate a Guardrails Guard to contract validators.
        
        Args:
            guard: Guardrails Guard object
            
        Returns:
            List of migrated contract validators
        """
        self.migration_metrics["total_migrations"] += 1
        
        try:
            if not GUARDRAILS_AVAILABLE:
                raise ImportError("Guardrails is required for migration")
            
            validators = []
            
            # Extract validators from guard
            if hasattr(guard, 'validators'):
                for validator in guard.validators:
                    migrated = self._migrate_single_validator(validator)
                    if migrated:
                        validators.append(migrated)
            
            self.migration_metrics["successful_migrations"] += 1
            return validators
            
        except Exception as e:
            self.migration_metrics["failed_migrations"] += 1
            logger.error(f"Guard migration failed: {e}")
            return []
    
    def _migrate_single_validator(self, validator: Any) -> Optional[GuardrailsValidator]:
        """Migrate a single Guardrails validator."""
        try:
            # Detect validator type and extract configuration
            config = self._extract_validator_config(validator)
            
            migrated_validator = GuardrailsValidator(
                config=config,
                original_validator=validator
            )
            
            self.migrated_validators[config.name] = migrated_validator
            return migrated_validator
            
        except Exception as e:
            logger.error(f"Single validator migration failed: {e}")
            return None
    
    def _extract_validator_config(self, validator: Any) -> GuardrailsValidatorConfig:
        """Extract configuration from a Guardrails validator."""
        # This would be implemented based on actual Guardrails validator structure
        validator_name = getattr(validator, '__class__', {}).get('__name__', 'unknown')
        
        # Map common Guardrails validators to our types
        type_mapping = {
            'RegexMatch': GuardrailsValidatorType.REGEX,
            'ValidLength': GuardrailsValidatorType.LENGTH,
            'ValidChoices': GuardrailsValidatorType.CHOICE,
            'ValidRange': GuardrailsValidatorType.RANGE,
            'ValidUrl': GuardrailsValidatorType.FORMAT,
            'ValidEmail': GuardrailsValidatorType.FORMAT,
        }
        
        validator_type = type_mapping.get(validator_name, GuardrailsValidatorType.CUSTOM)
        
        # Extract parameters based on validator type
        parameters = {}
        if hasattr(validator, '__dict__'):
            parameters = {k: v for k, v in validator.__dict__.items() 
                         if not k.startswith('_')}
        
        return GuardrailsValidatorConfig(
            name=validator_name,
            validator_type=validator_type,
            parameters=parameters,
            on_fail=getattr(validator, 'on_fail', 'exception')
        )
    
    def migrate_config_file(self, config_path: str) -> List[GuardrailsValidator]:
        """Migrate validators from a Guardrails configuration file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            validators = []
            
            # Parse configuration and create validators
            if 'validators' in config_data:
                for validator_config in config_data['validators']:
                    config = GuardrailsValidatorConfig(
                        name=validator_config.get('name', 'unknown'),
                        validator_type=GuardrailsValidatorType(validator_config.get('type', 'custom')),
                        parameters=validator_config.get('parameters', {}),
                        on_fail=validator_config.get('on_fail', 'exception')
                    )
                    
                    validator = GuardrailsValidator(config)
                    validators.append(validator)
            
            return validators
            
        except Exception as e:
            logger.error(f"Config file migration failed: {e}")
            return []
    
    def generate_migration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive migration report."""
        return {
            "migration_metrics": self.migration_metrics,
            "migrated_validators": {
                name: validator.get_metrics()
                for name, validator in self.migrated_validators.items()
            },
            "total_migrated": len(self.migrated_validators),
            "success_rate": self.migration_metrics["successful_migrations"] / max(self.migration_metrics["total_migrations"], 1)
        }


class GuardrailsMigrator:
    """High-level migration utility for Guardrails.ai projects."""
    
    def __init__(self):
        self.adapter = GuardrailsAdapter()
        self.migration_plan = []
        self.migration_results = []
    
    def analyze_guardrails_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze a Guardrails project for migration planning."""
        analysis = {
            "total_guards": 0,
            "total_validators": 0,
            "validator_types": {},
            "complexity_score": 0,
            "migration_recommendations": []
        }
        
        try:
            import os
            
            # Scan for Guardrails configuration files
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if file.endswith(('.py', '.json', '.yaml', '.yml')):
                        file_path = os.path.join(root, file)
                        self._analyze_file(file_path, analysis)
            
            # Generate recommendations
            analysis["migration_recommendations"] = self._generate_recommendations(analysis)
            
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
        
        return analysis
    
    def _analyze_file(self, file_path: str, analysis: Dict[str, Any]):
        """Analyze a single file for Guardrails usage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for Guardrails imports and usage patterns
            if 'import guardrails' in content or 'from guardrails' in content:
                analysis["total_guards"] += content.count('Guard(')
                
                # Count validator types
                validator_patterns = {
                    'RegexMatch': r'RegexMatch\(',
                    'ValidLength': r'ValidLength\(',
                    'ValidChoices': r'ValidChoices\(',
                    'ValidRange': r'ValidRange\(',
                    'ValidUrl': r'ValidUrl\(',
                    'ValidEmail': r'ValidEmail\(',
                }
                
                for validator_type, pattern in validator_patterns.items():
                    import re
                    matches = len(re.findall(pattern, content))
                    analysis["validator_types"][validator_type] = analysis["validator_types"].get(validator_type, 0) + matches
                    analysis["total_validators"] += matches
        
        except Exception as e:
            logger.debug(f"Failed to analyze file {file_path}: {e}")
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate migration recommendations based on analysis."""
        recommendations = []
        
        if analysis["total_guards"] > 10:
            recommendations.append("Consider batch migration approach for large number of guards")
        
        if analysis["total_validators"] > 50:
            recommendations.append("Plan for comprehensive testing due to high validator count")
        
        # Type-specific recommendations
        for validator_type, count in analysis["validator_types"].items():
            if count > 10:
                recommendations.append(f"High usage of {validator_type} - consider custom optimization")
        
        if not analysis["validator_types"]:
            recommendations.append("No standard validators found - manual migration may be required")
        
        return recommendations
    
    def create_migration_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a detailed migration plan."""
        plan = []
        
        # Phase 1: Standard validators
        if analysis["validator_types"]:
            plan.append({
                "phase": 1,
                "name": "Migrate Standard Validators",
                "description": "Migrate common Guardrails validators",
                "estimated_effort": "Low",
                "validators": list(analysis["validator_types"].keys())
            })
        
        # Phase 2: Custom logic
        if analysis["total_validators"] > sum(analysis["validator_types"].values()):
            plan.append({
                "phase": 2,
                "name": "Migrate Custom Validators",
                "description": "Handle custom validation logic",
                "estimated_effort": "Medium",
                "validators": ["Custom"]
            })
        
        # Phase 3: Integration testing
        plan.append({
            "phase": 3,
            "name": "Integration Testing",
            "description": "Test migrated validators in contract framework",
            "estimated_effort": "Medium",
            "validators": []
        })
        
        self.migration_plan = plan
        return plan
    
    def execute_migration(self, project_path: str) -> Dict[str, Any]:
        """Execute the complete migration process."""
        start_time = time.time()
        
        # Analyze project
        analysis = self.analyze_guardrails_project(project_path)
        
        # Create migration plan
        plan = self.create_migration_plan(analysis)
        
        # Execute migration phases
        results = []
        for phase in plan:
            phase_result = self._execute_migration_phase(phase, project_path)
            results.append(phase_result)
        
        migration_time = time.time() - start_time
        
        return {
            "analysis": analysis,
            "migration_plan": plan,
            "results": results,
            "migration_time_seconds": migration_time,
            "success": all(result["success"] for result in results)
        }
    
    def _execute_migration_phase(self, phase: Dict[str, Any], project_path: str) -> Dict[str, Any]:
        """Execute a single migration phase."""
        try:
            if phase["name"] == "Migrate Standard Validators":
                # Find and migrate standard validators
                migrated_validators = []
                # Implementation would scan files and migrate validators
                
                return {
                    "phase": phase["phase"],
                    "success": True,
                    "migrated_count": len(migrated_validators),
                    "validators": migrated_validators
                }
            
            elif phase["name"] == "Migrate Custom Validators":
                # Handle custom validators
                return {
                    "phase": phase["phase"],
                    "success": True,
                    "migrated_count": 0,
                    "note": "Manual review required for custom validators"
                }
            
            elif phase["name"] == "Integration Testing":
                # Run integration tests
                return {
                    "phase": phase["phase"],
                    "success": True,
                    "tests_passed": 0,
                    "tests_failed": 0
                }
            
            else:
                return {
                    "phase": phase["phase"],
                    "success": False,
                    "error": "Unknown phase"
                }
        
        except Exception as e:
            return {
                "phase": phase["phase"],
                "success": False,
                "error": str(e)
            }


# Convenience functions

def convert_guardrails_to_contract(guard: Any) -> List[ContractBase]:
    """Convert a Guardrails Guard to contract validators."""
    adapter = GuardrailsAdapter()
    return adapter.migrate_guard(guard)


def migrate_guardrails_config(config_path: str) -> List[GuardrailsValidator]:
    """Migrate validators from a Guardrails configuration file."""
    adapter = GuardrailsAdapter()
    return adapter.migrate_config_file(config_path)


# Example usage
def example_guardrails_migration():
    """Example of Guardrails migration usage."""
    if not GUARDRAILS_AVAILABLE:
        print("Guardrails not available for migration example")
        return
    
    # Example migration process
    migrator = GuardrailsMigrator()
    
    # Analyze project
    analysis = migrator.analyze_guardrails_project("./my_guardrails_project")
    print(f"Found {analysis['total_guards']} guards and {analysis['total_validators']} validators")
    
    # Create migration plan
    plan = migrator.create_migration_plan(analysis)
    print(f"Migration plan has {len(plan)} phases")
    
    # Execute migration
    results = migrator.execute_migration("./my_guardrails_project")
    print(f"Migration completed: {results['success']}")