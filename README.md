# OpenAI LLM Contracts

A comprehensive reliability layer for Large Language Model APIs, implementing Design by Contract principles with a focus on OpenAI's API.

## Overview

The LLM Design by Contract Framework addresses critical reliability issues in production LLM applications:

- **~60% invalid input rate** in production LLM applications
- **~20% output format compliance problems**
- **Lack of temporal consistency** across multi-turn conversations
- **Missing safety and content policy enforcement**

This framework provides a **transparent compatibility layer** that intercepts, validates, and ensures compliance for all LLM API interactions while maintaining 100% SDK compatibility.

## Installation

```bash
pip install openai-llm-contracts
```

For additional features:
```bash
# LangChain integration
pip install "openai-llm-contracts[langchain]"

# Guardrails.ai migration support
pip install "openai-llm-contracts[guardrails]"

# OpenTelemetry observability
pip install "openai-llm-contracts[telemetry]"

# Development tools
pip install "openai-llm-contracts[dev]"
```

## Quick Start

### Basic Usage

```python
from openai import OpenAI
from llm_contracts.providers import ImprovedOpenAIProvider
from llm_contracts.contracts.base import PromptLengthContract, JSONFormatContract

# Drop-in replacement for OpenAI client
client = ImprovedOpenAIProvider(api_key="your-api-key")

# Add contracts
client.add_input_contract(PromptLengthContract(max_tokens=4000))
client.add_output_contract(JSONFormatContract(schema={
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}))

# Use exactly like OpenAI SDK
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Generate a person profile"}]
)
```

### Using LLMCL (LLM Contract Language)

```python
from llm_contracts.language import llmcl_contract

@llmcl_contract("""
contract PersonGenerator {
    require prompt_length < 1000 "Prompt too long"
    ensure is_json(output) "Output must be JSON"
    ensure has_fields(output, ["name", "age"]) "Missing required fields"
}
""")
def generate_person(client, prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
```

### Streaming with Validation

```python
# Streaming responses with real-time validation
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
# Validation happens in real-time during streaming
```

### Multi-turn Conversations

```python
from llm_contracts.conversation import ConversationStateManager

# Create conversation manager
conversation = ConversationStateManager()
client.set_conversation_manager(conversation)

# Add temporal contracts
from llm_contracts.conversation.temporal_contracts import ConsistencyContract
client.add_temporal_contract(ConsistencyContract())

# Have a conversation
response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "My name is Alice"}]
)

response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's my name?"}]
)
# Temporal contracts ensure consistency across turns
```

## Key Features

### 1. **100% OpenAI SDK Compatibility**
- Drop-in replacement for OpenAI client
- All parameters and methods work exactly the same
- No code changes required for migration

### 2. **Comprehensive Contract Types**
- Input validation (length, format, content)
- Output validation (structure, format, completeness)
- Temporal contracts (consistency across conversations)
- Semantic contracts (meaning preservation)
- Performance contracts (latency, token usage)
- Security contracts (injection detection, PII filtering)

### 3. **Production-Ready Features**
- Circuit breaker for graceful degradation
- Async-first design with concurrent validation
- Streaming response validation
- Auto-remediation with intelligent retry
- Comprehensive metrics and observability

### 4. **Developer Experience**
- Contract testing framework
- Debugging and profiling tools
- A/B testing support
- Gradual rollout capabilities
- IDE integration support

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [LLMCL Language Reference](docs/LLMCL_LANGUAGE_REFERENCE.md)
- [API Reference](docs/LLMCL_API_REFERENCE_AND_EXAMPLES.md)
- [Performance & Best Practices](docs/LLMCL_PERFORMANCE_OBSERVABILITY_AND_BEST_PRACTICES.md)
- [Contracts & Temporal Logic](docs/LLMCL_CONTRACTS_AND_TEMPORAL_LOGIC.md)
- [Conflict Resolution](docs/LLMCL_CONFLICT_RESOLUTION_AND_AUTO_REMEDIATION.md)

## Examples

See the [examples](examples/) directory for more comprehensive examples:
- [Basic Usage](examples/simple_demo.py)
- [LLMCL Demo](examples/llmcl_demo.py)
- [Streaming Validation](examples/streaming_validation_demo.py)
- [Conversation State](examples/conversation_state_demo.py)
- [A/B Testing](examples/ab_testing_demo.py)
- [Ecosystem Integration](examples/ecosystem_integration_demo.py)

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This framework builds upon the excellent work of the OpenAI Python SDK and integrates with various tools in the LLM ecosystem including LangChain, Guardrails.ai, and Pydantic.