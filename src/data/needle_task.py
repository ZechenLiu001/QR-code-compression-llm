"""Needle-in-haystack task data generation"""

import random
from typing import Dict, Any


def generate_needle_sample(
    target_tokens: int,
    needle_position: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate needle-in-haystack task sample
    
    Args:
        target_tokens: Target token count for context
        needle_position: Relative position of needle (0.0-1.0)
        seed: Random seed
        
    Returns:
        Dictionary with context, query, answer, metadata
    """
    random.seed(seed)
    
    # Generate needle (secret code)
    needle = _generate_needle(seed)
    
    # Generate haystack text
    haystack = _generate_haystack(target_tokens, needle, needle_position, seed)
    
    # Query
    query = "文中提到的秘密代码是什么？只输出代码本身，不要解释。"
    
    return {
        "context": haystack,
        "query": query,
        "answer": needle,
        "metadata": {
            "needle_position": needle_position,
            "needle_length": len(needle),
            "haystack_length": len(haystack),
        },
    }


def _generate_needle(seed: int) -> str:
    """Generate a unique needle (secret code)"""
    random.seed(seed)
    # Generate a unique code like "SECRET-ABC123-XYZ789"
    prefix = random.choice(["SECRET", "CODE", "KEY", "TOKEN", "PASS"])
    middle = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))
    suffix = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))
    return f"{prefix}-{middle}-{suffix}"


def _generate_haystack(target_tokens: int, needle: str, position: float, seed: int) -> str:
    """Generate haystack text with needle inserted at position"""
    random.seed(seed)
    
    # Rough estimate: ~4 chars per token
    target_chars = target_tokens * 4
    needle_chars = len(needle)
    haystack_chars = target_chars - needle_chars - 50  # Leave room for insertion text
    
    # Generate filler text
    paragraphs = _generate_paragraphs(haystack_chars, seed)
    
    # Insert needle at position
    insert_pos = int(len(paragraphs) * position)
    
    # Create insertion context
    insertion_text = f"\n\n重要信息：秘密代码是 {needle}。请妥善保管。\n\n"
    
    haystack = paragraphs[:insert_pos] + insertion_text + paragraphs[insert_pos:]
    
    return haystack


def _generate_paragraphs(target_chars: int, seed: int) -> str:
    """Generate filler paragraphs"""
    random.seed(seed)
    
    sentences = [
        "这是一个关于技术发展的段落。",
        "人工智能正在改变我们的生活方式。",
        "机器学习算法需要大量的训练数据。",
        "深度学习模型在图像识别方面表现出色。",
        "自然语言处理技术不断取得突破。",
        "计算机视觉应用越来越广泛。",
        "数据科学帮助我们理解复杂现象。",
        "云计算提供了强大的计算资源。",
        "边缘计算降低了延迟和带宽需求。",
        "区块链技术保证了数据的安全性。",
        "物联网连接了各种智能设备。",
        "5G网络提供了高速的数据传输。",
        "量子计算有望解决复杂优化问题。",
        "生物信息学推动了医学研究进展。",
        "自动化系统提高了生产效率。",
    ]
    
    text = []
    current_len = 0
    
    while current_len < target_chars:
        sentence = random.choice(sentences)
        text.append(sentence)
        current_len += len(sentence) + 1
        
        # Add paragraph break occasionally
        if random.random() < 0.3:
            text.append("\n\n")
            current_len += 2
    
    return " ".join(text)
