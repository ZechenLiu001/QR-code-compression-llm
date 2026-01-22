"""Base data generator class"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDataGenerator(ABC):
    """Base class for data generators"""
    
    @abstractmethod
    def generate_sample(self, target_tokens: int, seed: int, **kwargs) -> Dict[str, Any]:
        """Generate a single sample
        
        Args:
            target_tokens: Target number of tokens for context
            seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with context, query, answer, metadata
        """
        pass
