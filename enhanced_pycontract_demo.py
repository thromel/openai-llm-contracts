#!/usr/bin/env python3
"""
Enhanced PyContract-style syntax demonstration

This shows the full range of PyContract-style syntax for complex LLM contracts,
including advanced features like decorators, LLMCL integration, and composition.
"""

from typing import Dict, Any, List
import asyncio
import json

# Import our complex PyContract implementation
from complex_pycontract_syntax import PyContractFactory, ComplexPyContractParser

# Decorator for applying multiple complex contracts
def complex_pycontract(**constraints):
    """Apply multiple PyContract-style constraints to a function."""
    def decorator(func):
        contracts = {}
        for name, constraint in constraints.items():
            try:
                contracts[name] = PyContractFactory.create_contract(name, constraint)
            except Exception as e:
                print(f"Warning: Failed to create contract {name}: {e}")
        
        async def async_wrapper(*args, **kwargs):
            # Pre-execution validation
            for name, contract in contracts.items():
                if hasattr(contract, '_get_contract_type') and contract._get_contract_type().name in ['INPUT', 'SECURITY']:
                    result = contract.validate(kwargs)
                    if not result.is_valid:
                        raise ValueError(f"Pre-condition failed ({name}): {result.message}")
            
            # Execute function
            response = await func(*args, **kwargs)
            
            # Post-execution validation
            for name, contract in contracts.items():
                if hasattr(contract, '_get_contract_type') and contract._get_contract_type().name in ['OUTPUT', 'PERFORMANCE', 'TEMPORAL']:
                    result = contract.validate(response, context={'start_time': kwargs.get('_start_time')})
                    if not result.is_valid and contract.is_required:
                        raise ValueError(f"Post-condition failed ({name}): {result.message}")
            
            return response
        
        def sync_wrapper(*args, **kwargs):
            # Simplified sync version
            response = func(*args, **kwargs)
            return response
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class PyContractComposer:
    """Compose multiple PyContract constraints into complex validation rules."""
    
    def __init__(self):
        self.contracts = {}
    
    def add_constraint(self, name: str, constraint: str):
        """Add a PyContract constraint."""
        self.contracts[name] = PyContractFactory.create_contract(name, constraint)
        return self
    
    def add_composite_constraint(self, name: str, constraint_type: str, sub_constraints: List[str]):
        """Add a composite constraint that combines multiple constraints."""
        # Example: "all_of", "any_of", "none_of"
        if constraint_type == "all_of":
            self.contracts[name] = AllOfContract(name, [
                PyContractFactory.create_contract(f"{name}_{i}", constraint)
                for i, constraint in enumerate(sub_constraints)
            ])
        elif constraint_type == "any_of":
            self.contracts[name] = AnyOfContract(name, [
                PyContractFactory.create_contract(f"{name}_{i}", constraint)
                for i, constraint in enumerate(sub_constraints)
            ])
        return self
    
    def validate_all(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate against all constraints."""
        results = {}
        for name, contract in self.contracts.items():
            result = contract.validate(data, context)
            results[name] = {
                'valid': result.is_valid,
                'message': result.message,
                'auto_fix': result.auto_fix_suggestion
            }
        return results


class AllOfContract:
    """Contract that requires all sub-contracts to pass."""
    
    def __init__(self, name: str, contracts: List[Any]):
        self.name = name
        self.contracts = contracts
        self.is_required = True
    
    def validate(self, data: Any, context: Dict[str, Any] = None):
        from src.llm_contracts.core.interfaces import ValidationResult
        
        for contract in self.contracts:
            result = contract.validate(data, context)
            if not result.is_valid:
                return ValidationResult(
                    False,
                    f"Composite validation failed: {result.message}",
                    auto_fix_suggestion=result.auto_fix_suggestion
                )
        
        return ValidationResult(True, "All composite constraints satisfied")


class AnyOfContract:
    """Contract that requires at least one sub-contract to pass."""
    
    def __init__(self, name: str, contracts: List[Any]):
        self.name = name
        self.contracts = contracts
        self.is_required = True
    
    def validate(self, data: Any, context: Dict[str, Any] = None):
        from src.llm_contracts.core.interfaces import ValidationResult
        
        messages = []
        for contract in self.contracts:
            result = contract.validate(data, context)
            if result.is_valid:
                return ValidationResult(True, f"Alternative constraint satisfied: {result.message}")
            messages.append(result.message)
        
        return ValidationResult(
            False,
            f"No alternative constraints satisfied: {'; '.join(messages)}"
        )


def demo_advanced_pycontract_syntax():
    """Demonstrate advanced PyContract syntax features."""
    
    print("=== Advanced PyContract Syntax Demo ===\n")
    
    # 1. Composite constraints
    print("1. Composite Constraints:")
    composer = PyContractComposer()
    
    # Security constraints composition
    composer.add_composite_constraint(
        "comprehensive_security",
        "all_of",
        [
            "regex_pattern:(?i)(injection|exploit),message:Injection detected",
            "regex_pattern:(?i)(password|secret|key),message:Sensitive data detected",
            "regex_pattern:<script.*?>,message:XSS pattern detected"
        ]
    )
    
    # Performance constraints alternatives
    composer.add_composite_constraint(
        "flexible_performance",
        "any_of",
        [
            "response_time:<=3s,message:Fast response required",
            "cost_limit:$0.10/request,message:Low cost alternative",
            # Note: fixing the parsing issue by using <= instead of <
        ]
    )
    
    test_data = "Hello, how can I help you today?"
    results = composer.validate_all(test_data)
    
    for name, result in results.items():
        status = "✓" if result['valid'] else "✗"
        print(f"  {status} {name}: {result['message']}")
    
    print("\n2. Enhanced LLMCL-style Syntax:")
    
    # Show proposed enhanced syntax
    enhanced_llmcl = '''
    # PyContract-enhanced LLMCL syntax
    contract ComprehensiveValidation {
        # Parameter constraints with PyContract syntax
        param temperature: float[0:2], message="Temperature out of range"
        param top_p: float[0:1], message="top_p out of range"
        param max_tokens: int(0:4096], message="Token limit exceeded"
        
        # Security constraints
        security content_filter: regex_pattern="(?i)(injection|exploit)",
                                 auto_fix="sanitize"
        
        # Format constraints
        output json_format: schema={required: [status, data]},
                           auto_fix="wrap_object"
        
        # Performance constraints
        performance response_time: <=5s, alert_at=3s
        
        # Temporal constraints with window
        temporal always: len(response) > 0, window=10turns
        temporal eventually: json_valid(response), window=5turns
        
        # Budget constraints
        budget cost_limit: $100/month, alert_at=80%
        budget token_quota: 100000tokens/day, alert_at=90%
        
        # Reliability constraints
        reliability circuit_breaker: failure_threshold=5,
                                   timeout=30s,
                                   recovery_timeout=60s
        
        # Composite constraints
        require all_of(security.content_filter, output.json_format)
            message: "Both security and format requirements must be met"
        
        require any_of(performance.response_time, budget.cost_limit)
            message: "Either fast response or low cost required"
        
        # Cross-parameter constraints
        require not (exists(temperature) and exists(top_p))
            message: "Use either temperature or top_p, not both"
        
        # Domain-specific constraints
        domain medical {
            require contains(response, "consult a healthcare professional")
                message: "Medical disclaimer required"
        }
        
        domain financial {
            require contains(response, "not financial advice")
                message: "Financial disclaimer required"
        }
    }
    '''
    
    print(enhanced_llmcl)
    
    print("\n3. Decorator Usage with Complex Constraints:")
    
    # Example function with complex PyContract constraints
    @complex_pycontract(
        input_security='regex_pattern:(?i)(injection|exploit),message:Security threat',
        output_format='json_schema:required=[status],message:Status required',
        performance='response_time:<=2s,message:Response too slow',
        budget='cost_limit:$0.05/request,message:Request too expensive'
    )
    async def secure_llm_call(prompt: str, **params):
        """Simulated secure LLM call with comprehensive validation."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate response
        return {
            "status": "success",
            "response": f"Processed: {prompt}",
            "tokens_used": 50,
            "cost": 0.02
        }
    
    print("Function decorated with complex PyContract constraints:")
    print("@complex_pycontract(")
    print("    input_security='regex_pattern:(?i)(injection|exploit)',")
    print("    output_format='json_schema:required=[status]',")
    print("    performance='response_time:<=2s',")
    print("    budget='cost_limit:$0.05/request'")
    print(")")
    
    print("\n4. Integration Patterns:")
    
    # Show how to integrate with existing systems
    integration_examples = {
        "LangChain Integration": '''
        from langchain import LLMChain
        
        @complex_pycontract(
            input_validation='regex_pattern:(?i)(harmful|dangerous)',
            output_safety='content_policy:no_violence',
            performance='response_time:<=10s'
        )
        def safe_langchain_call(chain: LLMChain, input_data: dict):
            return chain.run(input_data)
        ''',
        
        "OpenAI Provider": '''
        provider = OpenAIProvider(api_key="...")
        
        # Add PyContract-style validations
        provider.add_input_contract(
            PyContractFactory.create_contract(
                "params", "temperature:float[0:2],top_p:float[0:1]"
            )
        )
        
        provider.add_output_contract(
            PyContractFactory.create_contract(
                "format", "json_schema:required=[choices]"
            )
        )
        ''',
        
        "Streaming Validation": '''
        @complex_pycontract(
            streaming_safety='regex_pattern:(?i)(stop|halt),action=terminate',
            streaming_format='json_chunks:valid_syntax',
            streaming_performance='chunk_rate:>=10/s'
        )
        async def validated_stream(prompt: str):
            async for chunk in llm.stream(prompt):
                yield chunk
        '''
    }
    
    for title, example in integration_examples.items():
        print(f"\n{title}:")
        print(example)


def demo_real_world_scenarios():
    """Demonstrate real-world PyContract usage scenarios."""
    
    print("\n=== Real-World PyContract Scenarios ===\n")
    
    scenarios = {
        "Healthcare Chatbot": {
            "constraints": {
                "medical_safety": 'regex_pattern:(?i)(diagnose|prescribe|medical advice),message:Medical disclaimer required,auto_fix=add_disclaimer',
                "privacy_protection": 'regex_pattern:(?i)(ssn|social security|dob),message:PII detected,auto_fix=redact',
                "response_quality": 'json_schema:required=[answer,confidence,sources],message:Complete response required',
                "liability_protection": 'temporal_always:contains(response,"consult healthcare professional"),window=conversation'
            },
            "description": "Ensures medical chatbot responses include disclaimers and protect patient privacy"
        },
        
        "Financial Advisory": {
            "constraints": {
                "financial_disclaimer": 'regex_pattern:investment|trading|financial advice,message:Disclaimer required,auto_fix=add_disclaimer',
                "risk_assessment": 'json_schema:required=[risk_level,disclaimers],message:Risk assessment required',
                "cost_control": 'cost_limit:$5/conversation,alert_at=80%,message=Conversation cost limit',
                "performance": 'response_time:<=3s,message=Response time too slow for trading context'
            },
            "description": "Ensures financial advice includes proper disclaimers and risk assessments"
        },
        
        "Content Moderation": {
            "constraints": {
                "toxicity_filter": 'regex_pattern:(?i)(hate|violence|harassment),message:Toxic content detected,auto_fix=moderate',
                "age_appropriate": 'content_policy:family_friendly,message=Content not age appropriate',
                "spam_detection": 'regex_pattern:(?i)(click here|buy now|limited time),message=Spam detected',
                "quality_threshold": 'temporal_eventually:quality_score>0.8,window=5turns,message=Quality improvement needed'
            },
            "description": "Comprehensive content moderation for user-generated content"
        },
        
        "Educational Assistant": {
            "constraints": {
                "accuracy_validation": 'json_schema:required=[answer,sources,confidence],message=Academic sources required',
                "plagiarism_check": 'regex_pattern:exact_match_threshold<0.9,message=Potential plagiarism',
                "learning_progression": 'temporal_always:difficulty_appropriate(context.grade_level),window=session',
                "engagement_tracking": 'temporal_eventually:student_interaction>0,window=10min,message=Student engagement required'
            },
            "description": "Ensures educational content is accurate, original, and appropriately challenging"
        }
    }
    
    for scenario_name, scenario in scenarios.items():
        print(f"{scenario_name}:")
        print(f"  Description: {scenario['description']}")
        print("  Constraints:")
        
        for constraint_name, constraint_spec in scenario['constraints'].items():
            print(f"    - {constraint_name}: {constraint_spec}")
        
        print()


if __name__ == "__main__":
    demo_advanced_pycontract_syntax()
    demo_real_world_scenarios()
    
    print("\n" + "=" * 80)
    print("Enhanced PyContract System Summary:")
    print("- Supports all existing complex contract types")
    print("- Provides concise, readable constraint syntax")
    print("- Enables composition of multiple constraints")
    print("- Integrates with existing LLMCL and decorator systems")
    print("- Offers real-world scenario patterns")
    print("- Maintains compatibility with current framework")
    print("=" * 80)