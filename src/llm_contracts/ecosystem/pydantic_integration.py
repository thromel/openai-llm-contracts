"""Pydantic integration for LLM Design by Contract framework.

This module provides comprehensive integration with Pydantic models,
including automatic contract generation from models and validation.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type, get_type_hints, get_origin, get_args
from enum import Enum
import logging

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError, Field
    from pydantic.fields import FieldInfo
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Mock classes when Pydantic is not available
    class BaseModel:
        pass
    class PydanticValidationError(Exception):
        pass
    class Field:
        pass
    class FieldInfo:
        pass
    PYDANTIC_AVAILABLE = False

# Import contract framework components
from ..contracts.base import ContractBase, ValidationResult
from ..validators.basic_validators import OutputValidator as BasicValidator
from ..core.exceptions import ContractViolationError
# from ..utils.telemetry import log_contract_execution  # Function not available

logger = logging.getLogger(__name__)


class PydanticContractType(Enum):
    """Types of contracts that can be generated from Pydantic models."""
    MODEL_VALIDATION = "model_validation"
    FIELD_VALIDATION = "field_validation"
    TYPE_VALIDATION = "type_validation"
    CONSTRAINT_VALIDATION = "constraint_validation"
    CUSTOM_VALIDATION = "custom_validation"


@dataclass
class PydanticFieldConstraint:
    """Represents a constraint extracted from a Pydantic field."""
    field_name: str
    constraint_type: str
    constraint_value: Any
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_name": self.field_name,
            "constraint_type": self.constraint_type,
            "constraint_value": self.constraint_value,
            "error_message": self.error_message
        }


class PydanticContract(ContractBase):
    """Contract implementation based on Pydantic models."""
    
    def __init__(self,
                 model_class: Type[BaseModel],
                 contract_name: Optional[str] = None,
                 strict_validation: bool = True,
                 auto_fix_enabled: bool = True):
        """Initialize Pydantic contract.
        
        Args:
            model_class: Pydantic model class to base contract on
            contract_name: Name for the contract
            strict_validation: Use strict validation mode
            auto_fix_enabled: Enable automatic fixing of validation errors
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is required for Pydantic integration")
        
        if not issubclass(model_class, BaseModel):
            raise ValueError("model_class must be a Pydantic BaseModel")
        
        super().__init__(
            contract_type="pydantic_model",
            contract_name=contract_name or f"pydantic_{model_class.__name__}",
            description=f"Contract based on Pydantic model {model_class.__name__}"
        )
        
        self.model_class = model_class
        self.strict_validation = strict_validation
        self.auto_fix_enabled = auto_fix_enabled
        
        # Extract constraints from model
        self.field_constraints = self._extract_field_constraints()
        
        # Validation metrics
        self.validation_metrics = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "auto_fixes_applied": 0,
            "validation_errors": []
        }
    
    async def validate_async(self, data: Any) -> ValidationResult:
        """Validate data against Pydantic model."""
        self.validation_metrics["total_validations"] += 1
        
        try:
            # Attempt to parse data with Pydantic model
            if isinstance(data, dict):
                model_instance = self.model_class(**data)
            elif isinstance(data, str):
                # Try to parse as JSON first
                try:
                    json_data = json.loads(data)
                    model_instance = self.model_class(**json_data)
                except json.JSONDecodeError:
                    # Treat as single field if model has only one field
                    fields = list(self.model_class.__fields__.keys())
                    if len(fields) == 1:
                        model_instance = self.model_class(**{fields[0]: data})
                    else:
                        raise ValidationError("Cannot parse string data for multi-field model")
            else:
                # Try direct instantiation
                model_instance = self.model_class(data)
            
            self.validation_metrics["successful_validations"] += 1
            
            # Return validated data as dict
            return ValidationResult(
                is_valid=True,
                validated_content=model_instance.dict()
            )
            
        except PydanticValidationError as e:
            self.validation_metrics["failed_validations"] += 1
            self.validation_metrics["validation_errors"].append(str(e))
            
            # Attempt auto-fix if enabled
            if self.auto_fix_enabled:
                fixed_data = await self._auto_fix_validation_error(data, e)
                if fixed_data is not None:
                    self.validation_metrics["auto_fixes_applied"] += 1
                    return ValidationResult(
                        is_valid=False,
                        error_message=str(e),
                        auto_fixed_content=fixed_data
                    )
            
            return ValidationResult(
                is_valid=False,
                error_message=f"Pydantic validation failed: {str(e)}"
            )
        
        except Exception as e:
            self.validation_metrics["failed_validations"] += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def _extract_field_constraints(self) -> List[PydanticFieldConstraint]:
        """Extract constraints from Pydantic model fields."""
        constraints = []
        
        if not hasattr(self.model_class, '__fields__'):
            return constraints
        
        for field_name, field_info in self.model_class.__fields__.items():
            # Extract type constraints
            field_type = field_info.type_
            if field_type:
                constraints.append(PydanticFieldConstraint(
                    field_name=field_name,
                    constraint_type="type",
                    constraint_value=field_type,
                    error_message=f"Field {field_name} must be of type {field_type}"
                ))
            
            # Extract field constraints from Field() definition
            if hasattr(field_info, 'field_info') and field_info.field_info:
                field_constraints = field_info.field_info
                
                # Min/max length constraints
                if hasattr(field_constraints, 'min_length') and field_constraints.min_length is not None:
                    constraints.append(PydanticFieldConstraint(
                        field_name=field_name,
                        constraint_type="min_length",
                        constraint_value=field_constraints.min_length
                    ))
                
                if hasattr(field_constraints, 'max_length') and field_constraints.max_length is not None:
                    constraints.append(PydanticFieldConstraint(
                        field_name=field_name,
                        constraint_type="max_length",
                        constraint_value=field_constraints.max_length
                    ))
                
                # Min/max value constraints
                if hasattr(field_constraints, 'ge') and field_constraints.ge is not None:
                    constraints.append(PydanticFieldConstraint(
                        field_name=field_name,
                        constraint_type="min_value",
                        constraint_value=field_constraints.ge
                    ))
                
                if hasattr(field_constraints, 'le') and field_constraints.le is not None:
                    constraints.append(PydanticFieldConstraint(
                        field_name=field_name,
                        constraint_type="max_value",
                        constraint_value=field_constraints.le
                    ))
                
                # Regex constraints
                if hasattr(field_constraints, 'regex') and field_constraints.regex is not None:
                    constraints.append(PydanticFieldConstraint(
                        field_name=field_name,
                        constraint_type="regex",
                        constraint_value=field_constraints.regex
                    ))
        
        return constraints
    
    async def _auto_fix_validation_error(self, data: Any, error: PydanticValidationError) -> Optional[Dict[str, Any]]:
        """Attempt to auto-fix Pydantic validation errors."""
        if not isinstance(data, dict):
            return None
        
        fixed_data = data.copy()
        errors = error.errors()
        
        for error_info in errors:
            field_path = error_info.get('loc', [])
            error_type = error_info.get('type', '')
            error_msg = error_info.get('msg', '')
            
            if not field_path:
                continue
            
            field_name = field_path[0] if len(field_path) == 1 else '.'.join(str(p) for p in field_path)
            
            # Apply specific fixes based on error type
            if error_type == 'missing':
                # Add missing required fields with default values
                fixed_data[field_name] = await self._get_default_value_for_field(field_name)
            
            elif error_type == 'value_error.str.regex':
                # Try to fix regex validation errors
                if field_name in fixed_data:
                    fixed_value = await self._fix_regex_validation(fixed_data[field_name], field_name)
                    if fixed_value is not None:
                        fixed_data[field_name] = fixed_value
            
            elif error_type in ('value_error.number.not_ge', 'value_error.number.not_le'):
                # Fix numeric range errors
                if field_name in fixed_data:
                    fixed_value = await self._fix_numeric_range(fixed_data[field_name], field_name, error_type)
                    if fixed_value is not None:
                        fixed_data[field_name] = fixed_value
            
            elif error_type in ('value_error.any_str.min_length', 'value_error.any_str.max_length'):
                # Fix string length errors
                if field_name in fixed_data:
                    fixed_value = await self._fix_string_length(fixed_data[field_name], field_name, error_type)
                    if fixed_value is not None:
                        fixed_data[field_name] = fixed_value
            
            elif error_type == 'type_error':
                # Try type conversion
                if field_name in fixed_data:
                    fixed_value = await self._fix_type_error(fixed_data[field_name], field_name)
                    if fixed_value is not None:
                        fixed_data[field_name] = fixed_value
        
        # Validate the fixed data
        try:
            self.model_class(**fixed_data)
            return fixed_data
        except PydanticValidationError:
            return None
    
    async def _get_default_value_for_field(self, field_name: str) -> Any:
        """Get appropriate default value for a missing field."""
        if not hasattr(self.model_class, '__fields__'):
            return None
        
        field_info = self.model_class.__fields__.get(field_name)
        if not field_info:
            return None
        
        # Return field default if available
        if hasattr(field_info, 'default') and field_info.default is not None:
            return field_info.default
        
        # Generate appropriate default based on type
        field_type = field_info.type_
        
        if field_type == str:
            return ""
        elif field_type == int:
            return 0
        elif field_type == float:
            return 0.0
        elif field_type == bool:
            return False
        elif field_type == list:
            return []
        elif field_type == dict:
            return {}
        else:
            return None
    
    async def _fix_regex_validation(self, value: str, field_name: str) -> Optional[str]:
        """Attempt to fix regex validation errors."""
        # Find the regex constraint for this field
        regex_constraint = None
        for constraint in self.field_constraints:
            if constraint.field_name == field_name and constraint.constraint_type == "regex":
                regex_constraint = constraint.constraint_value
                break
        
        if not regex_constraint:
            return None
        
        # Apply basic regex fixes
        import re
        
        # Common regex patterns and fixes
        if str(regex_constraint) == r"^\d+$":  # Numbers only
            # Extract digits from string
            digits = re.findall(r'\d', value)
            return ''.join(digits) if digits else "0"
        
        elif str(regex_constraint) == r"^[a-zA-Z]+$":  # Letters only
            # Extract letters from string
            letters = re.findall(r'[a-zA-Z]', value)
            return ''.join(letters) if letters else "text"
        
        elif "email" in str(regex_constraint).lower():
            # Basic email fix
            if "@" not in value and "." in value:
                parts = value.split(".")
                return f"{parts[0]}@{'.'.join(parts[1:])}"
        
        return None
    
    async def _fix_numeric_range(self, value: Any, field_name: str, error_type: str) -> Optional[Union[int, float]]:
        """Fix numeric range validation errors."""
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return None
        
        # Find min/max constraints for this field
        min_constraint = None
        max_constraint = None
        
        for constraint in self.field_constraints:
            if constraint.field_name == field_name:
                if constraint.constraint_type == "min_value":
                    min_constraint = constraint.constraint_value
                elif constraint.constraint_type == "max_value":
                    max_constraint = constraint.constraint_value
        
        # Apply range clamping
        if error_type == 'value_error.number.not_ge' and min_constraint is not None:
            return max(numeric_value, min_constraint)
        elif error_type == 'value_error.number.not_le' and max_constraint is not None:
            return min(numeric_value, max_constraint)
        
        return None
    
    async def _fix_string_length(self, value: str, field_name: str, error_type: str) -> Optional[str]:
        """Fix string length validation errors."""
        # Find length constraints for this field
        min_length = None
        max_length = None
        
        for constraint in self.field_constraints:
            if constraint.field_name == field_name:
                if constraint.constraint_type == "min_length":
                    min_length = constraint.constraint_value
                elif constraint.constraint_type == "max_length":
                    max_length = constraint.constraint_value
        
        # Apply length fixes
        if error_type == 'value_error.any_str.min_length' and min_length is not None:
            if len(value) < min_length:
                return value + " " * (min_length - len(value))
        elif error_type == 'value_error.any_str.max_length' and max_length is not None:
            if len(value) > max_length:
                return value[:max_length]
        
        return None
    
    async def _fix_type_error(self, value: Any, field_name: str) -> Optional[Any]:
        """Attempt to fix type conversion errors."""
        # Find the expected type for this field
        expected_type = None
        for constraint in self.field_constraints:
            if constraint.field_name == field_name and constraint.constraint_type == "type":
                expected_type = constraint.constraint_value
                break
        
        if not expected_type:
            return None
        
        # Attempt type conversion
        try:
            if expected_type == str:
                return str(value)
            elif expected_type == int:
                if isinstance(value, str):
                    # Extract first number from string
                    import re
                    numbers = re.findall(r'-?\d+', value)
                    return int(numbers[0]) if numbers else 0
                return int(float(value))
            elif expected_type == float:
                if isinstance(value, str):
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', value)
                    return float(numbers[0]) if numbers else 0.0
                return float(value)
            elif expected_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema representation of the Pydantic model."""
        if hasattr(self.model_class, 'schema'):
            return self.model_class.schema()
        else:
            # Fallback for older Pydantic versions
            return {"type": "object", "properties": {}}
    
    def get_constraints_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all field constraints."""
        return [constraint.to_dict() for constraint in self.field_constraints]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            **self.validation_metrics,
            "success_rate": self.validation_metrics["successful_validations"] / max(self.validation_metrics["total_validations"], 1),
            "auto_fix_rate": self.validation_metrics["auto_fixes_applied"] / max(self.validation_metrics["failed_validations"], 1),
            "constraint_count": len(self.field_constraints)
        }


class PydanticValidator(BasicValidator):
    """Validator that uses Pydantic models for validation."""
    
    def __init__(self,
                 model_class: Type[BaseModel],
                 validator_name: Optional[str] = None):
        """Initialize Pydantic validator.
        
        Args:
            model_class: Pydantic model class to use for validation
            validator_name: Name for the validator
        """
        super().__init__(
            name=validator_name or f"pydantic_{model_class.__name__}",
            description=f"Validator based on Pydantic model {model_class.__name__}"
        )
        
        self.contract = PydanticContract(model_class)
    
    async def validate_async(self, data: Any) -> ValidationResult:
        """Validate data using Pydantic contract."""
        return await self.contract.validate_async(data)


class ModelBasedContract(ContractBase):
    """Contract that combines multiple Pydantic models."""
    
    def __init__(self,
                 models: Dict[str, Type[BaseModel]],
                 contract_name: Optional[str] = None,
                 validation_mode: str = "all"):  # "all", "any", "sequential"
        """Initialize model-based contract.
        
        Args:
            models: Dictionary of model names to Pydantic model classes
            contract_name: Name for the contract
            validation_mode: How to combine multiple models ("all", "any", "sequential")
        """
        super().__init__(
            contract_type="model_based",
            contract_name=contract_name or "multi_model_contract",
            description=f"Contract based on {len(models)} Pydantic models"
        )
        
        self.models = models
        self.validation_mode = validation_mode
        
        # Create individual contracts for each model
        self.model_contracts = {
            name: PydanticContract(model_class, contract_name=f"{self.contract_name}_{name}")
            for name, model_class in models.items()
        }
        
        self.validation_metrics = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "model_results": {name: {"success": 0, "failure": 0} for name in models.keys()}
        }
    
    async def validate_async(self, data: Any) -> ValidationResult:
        """Validate data against multiple models."""
        self.validation_metrics["total_validations"] += 1
        
        results = {}
        for model_name, contract in self.model_contracts.items():
            result = await contract.validate_async(data)
            results[model_name] = result
            
            if result.is_valid:
                self.validation_metrics["model_results"][model_name]["success"] += 1
            else:
                self.validation_metrics["model_results"][model_name]["failure"] += 1
        
        # Combine results based on validation mode
        if self.validation_mode == "all":
            # All models must pass
            all_valid = all(result.is_valid for result in results.values())
            if all_valid:
                self.validation_metrics["successful_validations"] += 1
                return ValidationResult(is_valid=True)
            else:
                self.validation_metrics["failed_validations"] += 1
                failed_models = [name for name, result in results.items() if not result.is_valid]
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Models failed validation: {failed_models}"
                )
        
        elif self.validation_mode == "any":
            # Any model can pass
            any_valid = any(result.is_valid for result in results.values())
            if any_valid:
                self.validation_metrics["successful_validations"] += 1
                return ValidationResult(is_valid=True)
            else:
                self.validation_metrics["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    error_message="No models passed validation"
                )
        
        elif self.validation_mode == "sequential":
            # Try models in order until one passes
            for model_name, result in results.items():
                if result.is_valid:
                    self.validation_metrics["successful_validations"] += 1
                    return ValidationResult(is_valid=True)
            
            self.validation_metrics["failed_validations"] += 1
            return ValidationResult(
                is_valid=False,
                error_message="No models passed sequential validation"
            )
        
        else:
            raise ValueError(f"Unknown validation mode: {self.validation_mode}")
    
    def get_model_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all models."""
        return {
            name: contract.get_schema()
            for name, contract in self.model_contracts.items()
        }


# Convenience functions

def create_pydantic_contract(model_class: Type[BaseModel],
                           contract_name: Optional[str] = None,
                           **kwargs) -> PydanticContract:
    """Create a Pydantic contract from a model class."""
    return PydanticContract(
        model_class=model_class,
        contract_name=contract_name,
        **kwargs
    )


def pydantic_to_contract_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to contract schema format."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for schema conversion")
    
    contract = PydanticContract(model_class)
    
    return {
        "model_name": model_class.__name__,
        "schema": contract.get_schema(),
        "constraints": contract.get_constraints_summary(),
        "contract_type": "pydantic_model"
    }


def create_multi_model_contract(models: Dict[str, Type[BaseModel]],
                              validation_mode: str = "all",
                              contract_name: Optional[str] = None) -> ModelBasedContract:
    """Create a contract that validates against multiple Pydantic models."""
    return ModelBasedContract(
        models=models,
        validation_mode=validation_mode,
        contract_name=contract_name
    )


# Example usage and utility functions

def example_pydantic_integration():
    """Example of Pydantic integration usage."""
    if not PYDANTIC_AVAILABLE:
        print("Pydantic not available for example")
        return
    
    # Example Pydantic model
    class UserModel(BaseModel):
        name: str = Field(min_length=1, max_length=50)
        age: int = Field(ge=0, le=120)
        email: str = Field(regex=r'^[^@]+@[^@]+\.[^@]+$')
        active: bool = True
    
    # Create contract from model
    contract = create_pydantic_contract(UserModel)
    
    # Test data
    test_data = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "active": True
    }
    
    # Validate (this would be async in real usage)
    print(f"Created contract for {UserModel.__name__}")
    print(f"Schema: {contract.get_schema()}")
    print(f"Constraints: {len(contract.get_constraints_summary())}")


def extract_pydantic_models_from_code(code: str) -> List[str]:
    """Extract Pydantic model definitions from Python code."""
    import ast
    import re
    
    models = []
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class inherits from BaseModel
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'BaseModel':
                        models.append(node.name)
                    elif isinstance(base, ast.Attribute) and base.attr == 'BaseModel':
                        models.append(node.name)
    
    except SyntaxError:
        # Fallback to regex if AST parsing fails
        pattern = r'class\s+(\w+)\s*\([^)]*BaseModel[^)]*\)\s*:'
        matches = re.findall(pattern, code)
        models.extend(matches)
    
    return models


def generate_contract_from_json_schema(schema: Dict[str, Any]) -> str:
    """Generate a Pydantic model and contract from JSON schema."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for schema generation")
    
    # This is a simplified implementation
    # In practice, you might use libraries like datamodel-code-generator
    
    model_name = schema.get('title', 'GeneratedModel')
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    # Generate Pydantic model code
    model_code = f"""
from pydantic import BaseModel, Field
from typing import Optional

class {model_name}(BaseModel):
"""
    
    for prop_name, prop_info in properties.items():
        prop_type = prop_info.get('type', 'str')
        is_required = prop_name in required
        
        # Map JSON Schema types to Python types
        type_mapping = {
            'string': 'str',
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'array': 'List',
            'object': 'Dict'
        }
        
        python_type = type_mapping.get(prop_type, 'str')
        if not is_required:
            python_type = f"Optional[{python_type}]"
        
        default_value = " = None" if not is_required else ""
        model_code += f"    {prop_name}: {python_type}{default_value}\n"
    
    return model_code.strip()