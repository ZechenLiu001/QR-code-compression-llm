"""JSON restore task data generation"""

import json
import random
from typing import Dict, Any


def generate_json_sample(
    target_tokens: int,
    field_count: int = 10,
    seed: int = 42,
    nesting_depth: int = 2,
) -> Dict[str, Any]:
    """Generate JSON restore task sample
    
    Task: Model must output complete JSON string (not just a single field)
    
    Args:
        target_tokens: Target token count for context
        field_count: Number of fields in JSON
        seed: Random seed
        nesting_depth: Maximum nesting depth
        
    Returns:
        Dictionary with context, query, answer, metadata
    """
    random.seed(seed)
    
    # Generate JSON structure
    json_obj = _generate_json_object(field_count, nesting_depth, seed)
    context_json = json.dumps(json_obj, ensure_ascii=False, indent=2)
    
    # Estimate current token count (rough estimate: ~4 chars per token)
    current_chars = len(context_json)
    current_tokens_est = current_chars // 4
    
    # Pad if needed to reach target tokens
    if current_tokens_est < target_tokens:
        padding_needed = (target_tokens - current_tokens_est) * 4
        padding_text = _generate_padding_text(padding_needed, seed + 1000)
        # Add padding as a comment-like field or additional fields
        json_obj["_padding"] = padding_text
        context_json = json.dumps(json_obj, ensure_ascii=False, indent=2)
    
    # Query
    query = "请完整还原上述 JSON 内容。要求：只输出严格合法 JSON，不要输出任何解释或额外文字。"
    
    # Answer (ground truth)
    answer = json.dumps(json_obj, ensure_ascii=False, separators=(',', ':'))
    
    return {
        "context": context_json,
        "query": query,
        "answer": answer,
        "metadata": {
            "field_count": len(json_obj),
            "nesting_depth": nesting_depth,
            "total_chars": len(context_json),
        },
    }


def _generate_json_object(field_count: int, max_depth: int, seed: int) -> Dict[str, Any]:
    """Generate a random JSON object"""
    random.seed(seed)
    obj = {}
    
    field_names = [
        "name", "id", "value", "count", "price", "status", "type", "category",
        "description", "timestamp", "location", "author", "title", "content",
        "email", "phone", "address", "city", "country", "zipcode",
    ]
    
    for i in range(field_count):
        field_name = random.choice(field_names) + str(i)
        if max_depth > 0 and random.random() < 0.3:
            # Nested object
            obj[field_name] = _generate_json_object(
                random.randint(2, 5), max_depth - 1, seed + i
            )
        elif random.random() < 0.5:
            # String value
            obj[field_name] = _generate_random_string(random.randint(5, 30), seed + i)
        elif random.random() < 0.7:
            # Number
            obj[field_name] = random.randint(1, 10000)
        else:
            # Boolean
            obj[field_name] = random.choice([True, False])
    
    return obj


def _generate_random_string(length: int, seed: int) -> str:
    """Generate random string"""
    random.seed(seed)
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(chars) for _ in range(length))


def _generate_padding_text(length: int, seed: int) -> str:
    """Generate padding text to reach target token count"""
    random.seed(seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
        "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
    ]
    text = []
    current_len = 0
    while current_len < length:
        word = random.choice(words)
        text.append(word)
        current_len += len(word) + 1
    return " ".join(text[:length // 10])  # Rough length control
