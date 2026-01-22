"""Text context codec - direct text encoding"""

from typing import Union, List, Dict, Any
from .base import BaseCodec


class TextCodec(BaseCodec):
    """Direct text encoding (no transformation)"""
    
    def encode(self, text: str) -> str:
        """Return text as-is"""
        return text
    
    def decode(self, encoded: str) -> str:
        """Return text as-is"""
        return encoded
    
    def get_token_cost(
        self,
        encoded: str,
        processor=None,
    ) -> Dict[str, Any]:
        """Get token cost for text
        
        Args:
            encoded: Text string
            processor: Optional processor for token counting
            
        Returns:
            Token cost dictionary
        """
        if processor is not None and hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
            tokens = tokenizer.encode(encoded, add_special_tokens=False)
            text_tokens = len(tokens)
        else:
            # Rough estimate: ~4 chars per token
            text_tokens = len(encoded) // 4
        
        return {
            "text_tokens": text_tokens,
            "visual_tokens": 0,
            "visual_tokens_source": "n/a",
            "patch_count": 0,
            "image_size": None,
            "total_tokens": text_tokens,
        }
