"""
OpenAI token counting utilities using tiktoken.

This module provides accurate token counting using OpenAI's official tiktoken library,
which is the same tokenizer used by OpenAI's models.
"""

import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache
import tiktoken

logger = logging.getLogger(__name__)


# Model to encoding mapping (from OpenAI's documentation)
MODEL_TO_ENCODING = {
    # GPT-4 models
    "gpt-4": "cl100k_base",
    "gpt-4-32k": "cl100k_base", 
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-vision-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    
    # GPT-3.5 models
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    
    # Embedding models
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base", 
    "text-embedding-3-large": "cl100k_base",
    
    # Legacy models
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base", 
    "babbage": "r50k_base",
    "ada": "r50k_base",
}


@lru_cache(maxsize=128)
def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a specific model.
    
    Args:
        model: OpenAI model name
        
    Returns:
        tiktoken.Encoding object
    """
    encoding_name = MODEL_TO_ENCODING.get(model, "cl100k_base")
    
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding {encoding_name} for model {model}: {e}")
        # Fallback to cl100k_base (GPT-4/3.5-turbo default)
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using OpenAI's tiktoken.
    
    Args:
        text: Text to count tokens for
        model: OpenAI model name (determines tokenizer)
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    try:
        encoding = get_encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed for model {model}: {e}")
        # Fallback to rough estimation
        return max(1, len(text) // 4)


def count_tokens_from_messages(messages: List[Dict[str, Any]], model: str = "gpt-4") -> int:
    """Count tokens from OpenAI messages format.
    
    This includes the token overhead from message formatting.
    Based on OpenAI's cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    
    Args:
        messages: List of message dictionaries in OpenAI format
        model: OpenAI model name
        
    Returns:
        Total token count including message formatting overhead
    """
    if not messages:
        return 0
    
    try:
        encoding = get_encoding_for_model(model)
    except Exception as e:
        logger.warning(f"Failed to get encoding for {model}: {e}")
        # Fallback calculation
        total = 0
        for message in messages:
            for value in message.values():
                total += max(1, len(str(value)) // 4)
            total += 4  # Message formatting overhead
        return total
    
    # Token overhead varies by model family
    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|\n
        tokens_per_name = -1    # if there's a name, the role is omitted
    elif model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        # Default to gpt-4 values for unknown models
        tokens_per_message = 3
        tokens_per_name = 1
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def encode_text(text: str, model: str = "gpt-4") -> List[int]:
    """Encode text to token IDs.
    
    Args:
        text: Text to encode
        model: OpenAI model name
        
    Returns:
        List of token IDs
    """
    if not text:
        return []
    
    encoding = get_encoding_for_model(model)
    return encoding.encode(text)


def decode_tokens(tokens: List[int], model: str = "gpt-4") -> str:
    """Decode token IDs back to text.
    
    Args:
        tokens: List of token IDs
        model: OpenAI model name
        
    Returns:
        Decoded text
    """
    if not tokens:
        return ""
    
    encoding = get_encoding_for_model(model)
    return encoding.decode(tokens)


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate tokens in text (alias for count_tokens for backwards compatibility).
    
    Args:
        text: Text to count tokens for
        model: OpenAI model name
        
    Returns:
        Number of tokens
    """
    return count_tokens(text, model)


def get_model_context_limit(model: str) -> int:
    """Get the context window limit for a model.
    
    Args:
        model: OpenAI model name
        
    Returns:
        Context limit in tokens
    """
    # Context limits from OpenAI documentation
    context_limits = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-vision-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4-1106-preview": 128000,
        "gpt-4-0613": 8192,
        "gpt-4-32k-0613": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-instruct": 4096,
        "gpt-3.5-turbo-0125": 16384,
        "gpt-3.5-turbo-1106": 16384,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-16k-0613": 16384,
    }
    
    return context_limits.get(model, 4096)  # Default to 4096 if unknown


def is_within_context_limit(text: str, model: str = "gpt-4") -> bool:
    """Check if text is within the model's context limit.
    
    Args:
        text: Text to check
        model: OpenAI model name
        
    Returns:
        True if within limit, False otherwise
    """
    token_count = count_tokens(text, model)
    context_limit = get_model_context_limit(model)
    return token_count <= context_limit


def truncate_to_limit(text: str, model: str = "gpt-4", reserve_tokens: int = 0) -> str:
    """Truncate text to fit within model's context limit.
    
    Args:
        text: Text to truncate
        model: OpenAI model name
        reserve_tokens: Number of tokens to reserve (for response, etc.)
        
    Returns:
        Truncated text that fits within context limit
    """
    if not text:
        return text
    
    context_limit = get_model_context_limit(model)
    available_tokens = context_limit - reserve_tokens
    
    if available_tokens <= 0:
        return ""
    
    current_tokens = count_tokens(text, model)
    if current_tokens <= available_tokens:
        return text
    
    # Binary search to find the right truncation point
    encoding = get_encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= available_tokens:
        return text
    
    # Truncate tokens and decode back
    truncated_tokens = tokens[:available_tokens]
    return encoding.decode(truncated_tokens)


def clear_encoding_cache():
    """Clear the encoding cache."""
    get_encoding_for_model.cache_clear()
    logger.info("Encoding cache cleared")