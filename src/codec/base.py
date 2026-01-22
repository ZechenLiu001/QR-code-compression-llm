"""Base codec interface"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from PIL import Image


class BaseCodec(ABC):
    """Base class for context codecs"""
    
    @abstractmethod
    def encode(self, text: str) -> Union[str, Image.Image, List[Image.Image]]:
        """Encode text to context carrier
        
        Args:
            text: Input text string
            
        Returns:
            Encoded context (text string, single image, or list of images)
        """
        pass
    
    @abstractmethod
    def decode(self, encoded) -> str:
        """Decode context carrier back to text
        
        Used for validation and external baseline comparison
        
        Args:
            encoded: Encoded context (text, image, or list of images)
            
        Returns:
            Decoded text string
        """
        pass
    
    @abstractmethod
    def get_token_cost(
        self,
        encoded: Union[str, Image.Image, List[Image.Image]],
        processor=None,
    ) -> Dict[str, Any]:
        """Get token cost estimate
        
        Priority: processor output -> input_ids -> proxy estimate
        
        Args:
            encoded: Encoded context
            processor: Optional processor for precise token counting
            
        Returns:
            Dictionary with token cost information
        """
        pass
