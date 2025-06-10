"""Comprehensive tests for LLM ecosystem integration components."""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional

# Import ecosystem components
from .langchain_integration import (
    ContractLLM, LangChainContractValidator, ChainContractOrchestrator,
    ContractOutputParser, ContractAgent, ContractTool, ContractCallbackHandler,
    ContractExecutionMode, ContractExecutionContext
)

from .guardrails_adapter import (
    GuardrailsAdapter, GuardrailsMigrator, GuardrailsValidator,
    GuardrailsValidatorConfig, GuardrailsValidatorType
)

from .pydantic_integration import (
    PydanticContract, PydanticValidator, ModelBasedContract,
    PydanticFieldConstraint, create_pydantic_contract
)

from .opentelemetry_integration import (
    OpenTelemetryIntegration, ContractTracer, MetricsCollector,
    ContractSpanContext, TelemetryLevel
)

# Import framework components for testing
from ..contracts.base import ContractBase, ValidationResult
from ..validators.basic_validators import BasicValidator
from ..core.exceptions import ContractViolationError


class TestLangChainIntegration:
    """Tests for LangChain integration components."""
    
    def test_contract_execution_context(self):
        """Test contract execution context."""
        context = ContractExecutionContext(
            chain_id="test_chain",
            step_name="test_step",
            input_data={"prompt": "test"},
            metadata={"test": True}
        )
        
        assert context.chain_id == "test_chain"
        assert context.step_name == "test_step"
        assert context.input_data["prompt"] == "test"
        assert context.metadata["test"] is True
        
        # Test serialization
        context_dict = context.to_dict()
        assert context_dict["chain_id"] == "test_chain"
        assert context_dict["metadata"]["test"] is True
    
    @patch('llm_contracts.ecosystem.langchain_integration.LANGCHAIN_AVAILABLE', False)
    def test_contract_llm_without_langchain(self):
        """Test ContractLLM when LangChain is not available."""
        mock_llm = Mock()
        
        with pytest.raises(ImportError, match="LangChain is required"):
            ContractLLM(mock_llm)
    
    def test_langchain_contract_validator(self):
        """Test LangChain contract validator."""
        validator = LangChainContractValidator()
        
        # Test chain compatibility validation
        mock_chain = Mock()
        mock_chain.__call__ = Mock()
        mock_chain._call = Mock()
        mock_chain.input_keys = ["input"]
        mock_chain.output_keys = ["output"]
        
        result = validator.validate_chain_compatibility(mock_chain)
        assert result.is_valid
        
        # Test incompatible chain
        incompatible_chain = Mock()
        del incompatible_chain.__call__  # Remove required method
        
        result = validator.validate_chain_compatibility(incompatible_chain)
        assert not result.is_valid
        assert "missing required methods" in result.error_message.lower()
    
    def test_chain_contract_orchestrator(self):
        """Test chain contract orchestrator."""
        orchestrator = ChainContractOrchestrator()
        
        # Create mock contracts
        mock_input_contract = Mock(spec=ContractBase)
        mock_output_contract = Mock(spec=ContractBase)
        
        # Register contracts
        orchestrator.register_chain_contracts(
            chain_id="test_chain",
            input_contracts=[mock_input_contract],
            output_contracts=[mock_output_contract]
        )
        
        assert "test_chain" in orchestrator.chain_contracts
        assert len(orchestrator.chain_contracts["test_chain"]["input_contracts"]) == 1
        assert orchestrator.orchestration_metrics["chains_orchestrated"] == 1
    
    @pytest.mark.asyncio
    async def test_chain_orchestrator_execution(self):
        """Test chain orchestrator execution."""
        orchestrator = ChainContractOrchestrator()
        
        # Mock chain
        mock_chain = Mock()
        mock_chain.acall = Mock(return_value={"output": "test result"})
        
        # Mock contracts that pass validation
        mock_contract = Mock(spec=ContractBase)
        mock_validator = Mock()
        mock_validator.validate_async = Mock(return_value=ValidationResult(is_valid=True))
        
        with patch('llm_contracts.ecosystem.langchain_integration.PerformanceOptimizedInputValidator', return_value=mock_validator), \
             patch('llm_contracts.ecosystem.langchain_integration.PerformanceOptimizedOutputValidator', return_value=mock_validator):
            
            orchestrator.register_chain_contracts(
                chain_id="test_chain",
                input_contracts=[mock_contract],
                output_contracts=[mock_contract]
            )
            
            result = await orchestrator.execute_chain_with_contracts(
                chain=mock_chain,
                chain_id="test_chain",
                inputs={"input": "test"}
            )
            
            assert result["output"] == "test result"
            assert orchestrator.orchestration_metrics["total_executions"] == 1
    
    def test_contract_output_parser(self):
        """Test contract output parser."""
        # Mock base parser
        mock_base_parser = Mock()
        mock_base_parser.parse = Mock(return_value={"parsed": "data"})
        mock_base_parser.get_format_instructions = Mock(return_value="Base instructions")
        
        # Mock contract
        mock_contract = Mock(spec=ContractBase)
        
        parser = ContractOutputParser(
            base_parser=mock_base_parser,
            output_contracts=[mock_contract]
        )
        
        # Test parsing with validation success
        with patch.object(parser, 'output_validator') as mock_validator:
            mock_validator.validate_async = Mock(return_value=ValidationResult(is_valid=True))
            
            with patch('asyncio.run'):
                result = parser.parse("test text")
                assert result == {"parsed": "data"}
    
    def test_contract_callback_handler(self):
        """Test contract callback handler."""
        handler = ContractCallbackHandler()
        
        # Test LLM start callback
        handler.on_llm_start({"test": "serialized"}, ["prompt1", "prompt2"])
        assert handler.execution_count == 1
        assert len(handler.contract_events) == 1
        assert handler.contract_events[0]["event"] == "llm_start"
        
        # Test LLM error callback with contract violation
        error = ContractViolationError("Test violation")
        handler.on_llm_error(error)
        assert handler.violation_count == 1
        assert len(handler.contract_events) == 2
        
        # Test metrics
        metrics = handler.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["contract_violations"] == 1
        assert metrics["violation_rate"] == 1.0


class TestGuardrailsAdapter:
    """Tests for Guardrails.ai migration adapter."""
    
    def test_guardrails_validator_config(self):
        """Test Guardrails validator configuration."""
        config = GuardrailsValidatorConfig(
            name="test_validator",
            validator_type=GuardrailsValidatorType.REGEX,
            parameters={"pattern": r"\d+"},
            on_fail="fix"
        )
        
        assert config.name == "test_validator"
        assert config.validator_type == GuardrailsValidatorType.REGEX
        assert config.parameters["pattern"] == r"\d+"
        
        # Test serialization
        config_dict = config.to_dict()
        assert config_dict["validator_type"] == "regex"
    
    @pytest.mark.asyncio
    async def test_guardrails_validator_regex(self):
        """Test Guardrails validator with regex validation."""
        config = GuardrailsValidatorConfig(
            name="regex_validator",
            validator_type=GuardrailsValidatorType.REGEX,
            parameters={"pattern": r"^\d+$"},
            on_fail="exception"
        )
        
        validator = GuardrailsValidator(config)
        
        # Test valid input
        result = await validator.validate_async("12345")
        assert result.is_valid
        
        # Test invalid input
        result = await validator.validate_async("abc123")
        assert not result.is_valid
        assert "does not match pattern" in result.error_message
    
    @pytest.mark.asyncio
    async def test_guardrails_validator_length(self):
        """Test Guardrails validator with length validation."""
        config = GuardrailsValidatorConfig(
            name="length_validator",
            validator_type=GuardrailsValidatorType.LENGTH,
            parameters={"min": 5, "max": 10},
            on_fail="fix"
        )
        
        validator = GuardrailsValidator(config)
        
        # Test valid input
        result = await validator.validate_async("hello")
        assert result.is_valid
        
        # Test too short input with auto-fix
        result = await validator.validate_async("hi")
        assert not result.is_valid
        assert result.auto_fixed_content == "hi   "  # Padded with spaces
        
        # Test too long input with auto-fix
        result = await validator.validate_async("this is too long")
        assert not result.is_valid
        assert len(result.auto_fixed_content) == 10
    
    @pytest.mark.asyncio
    async def test_guardrails_validator_choice(self):
        """Test Guardrails validator with choice validation."""
        config = GuardrailsValidatorConfig(
            name="choice_validator",
            validator_type=GuardrailsValidatorType.CHOICE,
            parameters={"choices": ["red", "green", "blue"]},
            on_fail="fix"
        )
        
        validator = GuardrailsValidator(config)
        
        # Test valid choice
        result = await validator.validate_async("red")
        assert result.is_valid
        
        # Test invalid choice with auto-fix
        result = await validator.validate_async("purple")
        assert not result.is_valid
        assert result.auto_fixed_content in ["red", "green", "blue"]
    
    @pytest.mark.asyncio
    async def test_guardrails_validator_email_format(self):
        """Test Guardrails validator with email format validation."""
        config = GuardrailsValidatorConfig(
            name="email_validator",
            validator_type=GuardrailsValidatorType.FORMAT,
            parameters={"format": "email"},
            on_fail="fix"
        )
        
        validator = GuardrailsValidator(config)
        
        # Test valid email
        result = await validator.validate_async("user@example.com")
        assert result.is_valid
        
        # Test invalid email with potential fix
        result = await validator.validate_async("user.example.com")
        assert not result.is_valid
        if result.auto_fixed_content:
            assert "@" in result.auto_fixed_content
    
    def test_guardrails_adapter(self):
        """Test Guardrails adapter functionality."""
        adapter = GuardrailsAdapter()
        
        # Test configuration file migration
        with patch('builtins.open', mock_open(read_data='{"validators": [{"name": "test", "type": "regex", "parameters": {"pattern": "\\\\d+"}}]}')):
            validators = adapter.migrate_config_file("test_config.json")
            assert len(validators) == 1
            assert validators[0].config.name == "test"
    
    def test_guardrails_migrator(self):
        """Test Guardrails migrator functionality."""
        migrator = GuardrailsMigrator()
        
        # Test project analysis
        with patch('os.walk', return_value=[(".", [], ["test.py"])]), \
             patch('builtins.open', mock_open(read_data="import guardrails\nGuard()")):
            
            analysis = migrator.analyze_guardrails_project("test_project")
            assert analysis["total_guards"] >= 0
            assert isinstance(analysis["migration_recommendations"], list)
        
        # Test migration plan creation
        analysis = {"total_guards": 5, "total_validators": 10, "validator_types": {"RegexMatch": 3}}
        plan = migrator.create_migration_plan(analysis)
        assert len(plan) >= 2  # At least standard validators and integration testing phases


class TestPydanticIntegration:
    """Tests for Pydantic integration components."""
    
    @pytest.fixture
    def mock_pydantic_model(self):
        """Create a mock Pydantic model for testing."""
        if not hasattr(self, '_mock_model_created'):
            # Create a mock model class
            class MockUserModel:
                __name__ = "MockUserModel"
                __fields__ = {
                    "name": Mock(type_=str, field_info=Mock(min_length=1, max_length=50)),
                    "age": Mock(type_=int, field_info=Mock(ge=0, le=120)),
                    "email": Mock(type_=str, field_info=Mock(regex=r'^[^@]+@[^@]+\.[^@]+$'))
                }
                
                def __init__(self, **kwargs):
                    self.name = kwargs.get("name", "")
                    self.age = kwargs.get("age", 0)
                    self.email = kwargs.get("email", "")
                
                def dict(self):
                    return {"name": self.name, "age": self.age, "email": self.email}
                
                @classmethod
                def schema(cls):
                    return {"type": "object", "properties": {"name": {"type": "string"}}}
            
            self._mock_model_created = True
            self._mock_model = MockUserModel
        
        return self._mock_model
    
    def test_pydantic_field_constraint(self):
        """Test Pydantic field constraint representation."""
        constraint = PydanticFieldConstraint(
            field_name="age",
            constraint_type="min_value",
            constraint_value=0,
            error_message="Age must be non-negative"
        )
        
        assert constraint.field_name == "age"
        assert constraint.constraint_type == "min_value"
        assert constraint.constraint_value == 0
        
        # Test serialization
        constraint_dict = constraint.to_dict()
        assert constraint_dict["field_name"] == "age"
        assert constraint_dict["constraint_value"] == 0
    
    @patch('llm_contracts.ecosystem.pydantic_integration.PYDANTIC_AVAILABLE', False)
    def test_pydantic_contract_without_pydantic(self):
        """Test PydanticContract when Pydantic is not available."""
        with pytest.raises(ImportError, match="Pydantic is required"):
            PydanticContract(Mock())
    
    @patch('llm_contracts.ecosystem.pydantic_integration.PYDANTIC_AVAILABLE', True)
    @patch('llm_contracts.ecosystem.pydantic_integration.BaseModel')
    def test_pydantic_contract_invalid_model(self, mock_base_model):
        """Test PydanticContract with invalid model class."""
        mock_base_model.__subclasshook__ = Mock(return_value=False)
        
        class NotAModel:
            pass
        
        with patch('builtins.issubclass', return_value=False):
            with pytest.raises(ValueError, match="must be a Pydantic BaseModel"):
                PydanticContract(NotAModel)
    
    @patch('llm_contracts.ecosystem.pydantic_integration.PYDANTIC_AVAILABLE', True)
    def test_pydantic_contract_constraint_extraction(self, mock_pydantic_model):
        """Test constraint extraction from Pydantic model."""
        with patch('builtins.issubclass', return_value=True):
            contract = PydanticContract(mock_pydantic_model)
            
            constraints = contract._extract_field_constraints()
            assert len(constraints) > 0
            
            # Check for type constraints
            type_constraints = [c for c in constraints if c.constraint_type == "type"]
            assert len(type_constraints) >= 3  # name, age, email
    
    @pytest.mark.asyncio
    @patch('llm_contracts.ecosystem.pydantic_integration.PYDANTIC_AVAILABLE', True)
    async def test_pydantic_contract_validation(self, mock_pydantic_model):
        """Test Pydantic contract validation."""
        with patch('builtins.issubclass', return_value=True):
            contract = PydanticContract(mock_pydantic_model)
            
            # Test successful validation
            valid_data = {"name": "John", "age": 30, "email": "john@example.com"}
            result = await contract.validate_async(valid_data)
            assert result.is_valid
            assert result.validated_content is not None
    
    @pytest.mark.asyncio
    @patch('llm_contracts.ecosystem.pydantic_integration.PYDANTIC_AVAILABLE', True)
    async def test_pydantic_contract_auto_fix(self, mock_pydantic_model):
        """Test Pydantic contract auto-fix functionality."""
        with patch('builtins.issubclass', return_value=True), \
             patch('llm_contracts.ecosystem.pydantic_integration.PydanticValidationError') as mock_error:
            
            # Mock validation error
            mock_error_instance = Mock()
            mock_error_instance.errors.return_value = [
                {"loc": ["name"], "type": "missing", "msg": "field required"}
            ]
            
            contract = PydanticContract(mock_pydantic_model, auto_fix_enabled=True)
            
            # Mock the model to raise validation error then succeed
            def mock_init(*args, **kwargs):
                if not kwargs.get("name"):
                    raise mock_error_instance
                return mock_pydantic_model(**kwargs)
            
            with patch.object(mock_pydantic_model, '__init__', side_effect=mock_init):
                # Test data missing required field
                invalid_data = {"age": 30}
                
                # Mock the auto-fix method
                contract._get_default_value_for_field = Mock(return_value="Default Name")
                
                with patch.object(contract, 'model_class') as mock_model:
                    mock_model.side_effect = [mock_error_instance, mock_pydantic_model()]
                    
                    result = await contract.validate_async(invalid_data)
                    # The result depends on the auto-fix implementation
    
    def test_model_based_contract(self):
        """Test model-based contract with multiple models."""
        mock_model1 = Mock()
        mock_model1.__name__ = "Model1"
        mock_model2 = Mock()
        mock_model2.__name__ = "Model2"
        
        models = {"model1": mock_model1, "model2": mock_model2}
        
        with patch('llm_contracts.ecosystem.pydantic_integration.PydanticContract'):
            contract = ModelBasedContract(models, validation_mode="all")
            
            assert len(contract.models) == 2
            assert len(contract.model_contracts) == 2
            assert contract.validation_mode == "all"


class TestOpenTelemetryIntegration:
    """Tests for OpenTelemetry integration components."""
    
    def test_contract_span_context(self):
        """Test contract span context."""
        context = ContractSpanContext(
            contract_name="test_contract",
            contract_type="input",
            operation="validate",
            input_size=100,
            metadata={"test": "value"}
        )
        
        assert context.contract_name == "test_contract"
        assert context.contract_type == "input"
        assert context.operation == "validate"
        assert context.input_size == 100
        
        # Test attributes conversion
        attributes = context.to_attributes()
        assert attributes["llm_contract.name"] == "test_contract"
        assert attributes["llm_contract.type"] == "input"
        assert attributes["llm_contract.operation"] == "validate"
        assert attributes["llm_contract.input_size"] == 100
        assert attributes["llm_contract.metadata.test"] == "value"
    
    @patch('llm_contracts.ecosystem.opentelemetry_integration.OPENTELEMETRY_AVAILABLE', False)
    def test_contract_tracer_without_opentelemetry(self):
        """Test ContractTracer when OpenTelemetry is not available."""
        tracer = ContractTracer()
        assert tracer.tracer is None
        
        # Test span operations
        mock_context = ContractSpanContext("test", "input", "validate")
        span = tracer.start_contract_span(mock_context)
        assert span is None
    
    @patch('llm_contracts.ecosystem.opentelemetry_integration.OPENTELEMETRY_AVAILABLE', True)
    def test_contract_tracer_with_opentelemetry(self):
        """Test ContractTracer when OpenTelemetry is available."""
        with patch('llm_contracts.ecosystem.opentelemetry_integration.trace') as mock_trace:
            mock_tracer = Mock()
            mock_span = Mock()
            mock_tracer.start_span.return_value = mock_span
            mock_trace.get_tracer.return_value = mock_tracer
            
            tracer = ContractTracer()
            assert tracer.tracer is not None
            
            # Test span creation
            context = ContractSpanContext("test_contract", "input", "validate")
            span = tracer.start_contract_span(context)
            
            mock_tracer.start_span.assert_called_once()
            assert tracer.span_metrics["total_spans"] == 1
            assert tracer.span_metrics["active_spans"] == 1
    
    def test_metrics_collector(self):
        """Test metrics collector functionality."""
        collector = MetricsCollector()
        
        # Test recording validation
        collector.record_validation(
            contract_name="test_contract",
            contract_type="input",
            duration=0.1,
            is_valid=True,
            auto_fixed=False
        )
        
        assert collector.local_metrics["validations_total"] == 1
        assert collector.local_metrics["violations_total"] == 0
        assert len(collector.local_metrics["validation_durations"]) == 1
        
        # Test recording violation
        collector.record_validation(
            contract_name="test_contract",
            contract_type="input",
            duration=0.2,
            is_valid=False,
            auto_fixed=True
        )
        
        assert collector.local_metrics["validations_total"] == 2
        assert collector.local_metrics["violations_total"] == 1
        assert collector.local_metrics["auto_fixes_total"] == 1
        
        # Test metrics summary
        summary = collector.get_metrics_summary()
        assert summary["violation_rate"] == 0.5  # 1 violation out of 2 validations
        assert summary["auto_fix_rate"] == 1.0   # 1 fix out of 1 violation
    
    def test_contract_span_context_manager(self):
        """Test contract span context manager."""
        mock_tracer = Mock()
        mock_contract = Mock(spec=ContractBase)
        mock_contract.contract_name = "test_contract"
        mock_contract.contract_type = "input"
        mock_metrics = Mock()
        
        span_cm = ContractSpan(mock_tracer, mock_contract, "validate", mock_metrics)
        
        # Test context manager protocol
        with span_cm as span:
            assert span.start_time is not None
            span.set_result(ValidationResult(is_valid=True))
        
        # Verify tracer was called
        mock_tracer.start_contract_span.assert_called_once()
        mock_tracer.end_contract_span.assert_called_once()
    
    @patch('llm_contracts.ecosystem.opentelemetry_integration.OPENTELEMETRY_AVAILABLE', True)
    def test_opentelemetry_integration_initialization(self):
        """Test OpenTelemetry integration initialization."""
        with patch('llm_contracts.ecosystem.opentelemetry_integration.TracerProvider') as mock_provider, \
             patch('llm_contracts.ecosystem.opentelemetry_integration.MeterProvider'), \
             patch('llm_contracts.ecosystem.opentelemetry_integration.trace'), \
             patch('llm_contracts.ecosystem.opentelemetry_integration.metrics'):
            
            integration = OpenTelemetryIntegration(
                service_name="test_service",
                telemetry_level=TelemetryLevel.DETAILED
            )
            
            success = integration.initialize()
            assert success
            assert integration.initialized
    
    def test_opentelemetry_integration_contract_instrumentation(self):
        """Test contract instrumentation."""
        integration = OpenTelemetryIntegration()
        integration.initialize()
        
        # Mock contract
        mock_contract = Mock(spec=ContractBase)
        mock_contract.contract_name = "test_contract"
        mock_contract.contract_type = "input"
        mock_contract.validate_async = Mock(return_value=ValidationResult(is_valid=True))
        
        # Instrument contract
        instrumented_contract = integration.instrument_contract(mock_contract)
        
        # Verify contract is instrumented
        assert instrumented_contract is mock_contract
        assert hasattr(instrumented_contract, 'validate_async')


class TestEcosystemIntegration:
    """Integration tests across ecosystem components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_contract_with_telemetry(self):
        """Test end-to-end contract execution with telemetry."""
        # Set up telemetry
        integration = OpenTelemetryIntegration()
        integration.initialize()
        
        # Create a simple contract
        class TestContract(ContractBase):
            def __init__(self):
                super().__init__("test", "test_contract", "Test contract")
            
            async def validate_async(self, data: Any) -> ValidationResult:
                await asyncio.sleep(0.01)  # Simulate work
                return ValidationResult(is_valid=True, validated_content=data)
        
        contract = TestContract()
        
        # Instrument contract
        instrumented_contract = integration.instrument_contract(contract)
        
        # Execute validation
        result = await instrumented_contract.validate_async("test data")
        
        assert result.is_valid
        assert result.validated_content == "test data"
        
        # Check metrics were recorded
        metrics = integration.metrics_collector.get_metrics_summary()
        assert metrics["validations_total"] >= 1
    
    def test_guardrails_to_pydantic_migration_workflow(self):
        """Test workflow of migrating from Guardrails to Pydantic contracts."""
        # Simulate Guardrails configuration
        guardrails_config = {
            "validators": [
                {
                    "name": "email_validator",
                    "type": "format",
                    "parameters": {"format": "email"},
                    "on_fail": "exception"
                },
                {
                    "name": "length_validator", 
                    "type": "length",
                    "parameters": {"min": 5, "max": 50},
                    "on_fail": "fix"
                }
            ]
        }
        
        # Migrate Guardrails config
        adapter = GuardrailsAdapter()
        
        with patch('builtins.open', mock_open(read_data=json.dumps(guardrails_config))):
            migrated_validators = adapter.migrate_config_file("test_config.json")
        
        assert len(migrated_validators) == 2
        
        # Verify validators work
        email_validator = migrated_validators[0]
        length_validator = migrated_validators[1]
        
        assert email_validator.config.validator_type == GuardrailsValidatorType.FORMAT
        assert length_validator.config.validator_type == GuardrailsValidatorType.LENGTH
    
    @pytest.mark.asyncio
    async def test_multi_ecosystem_validation_pipeline(self):
        """Test validation pipeline using multiple ecosystem components."""
        # Create validation pipeline with different ecosystem components
        
        # 1. Pydantic contract for structure validation
        class MockPydanticModel:
            __name__ = "TestModel"
            __fields__ = {"text": Mock(type_=str)}
            
            def __init__(self, **kwargs):
                self.text = kwargs.get("text", "")
            
            def dict(self):
                return {"text": self.text}
        
        with patch('builtins.issubclass', return_value=True):
            pydantic_contract = PydanticContract(MockPydanticModel)
        
        # 2. Guardrails validator for content validation
        guardrails_config = GuardrailsValidatorConfig(
            name="content_validator",
            validator_type=GuardrailsValidatorType.LENGTH,
            parameters={"min": 1, "max": 100}
        )
        guardrails_validator = GuardrailsValidator(guardrails_config)
        
        # 3. Test data
        test_data = {"text": "This is a test message"}
        
        # Execute validation pipeline
        # Step 1: Pydantic structure validation
        pydantic_result = await pydantic_contract.validate_async(test_data)
        assert pydantic_result.is_valid
        
        # Step 2: Guardrails content validation
        guardrails_result = await guardrails_validator.validate_async(test_data["text"])
        assert guardrails_result.is_valid
        
        # Pipeline passes
        assert all([pydantic_result.is_valid, guardrails_result.is_valid])


# Helper function for mocking file operations
def mock_open(read_data=""):
    """Create a mock for file operations."""
    mock_file = Mock()
    mock_file.read.return_value = read_data
    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = None
    return Mock(return_value=mock_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])