"""Length bucket utilities"""

from typing import List, Tuple
import random


class LengthBucket:
    """Manages length buckets for data generation"""
    
    def __init__(self, buckets: List[int]):
        """Initialize length buckets
        
        Args:
            buckets: List of target token counts (e.g., [1024, 2048, 4096])
        """
        self.buckets = sorted(buckets)
    
    def get_bucket(self, target_tokens: int) -> int:
        """Get the bucket for a target token count
        
        Args:
            target_tokens: Target token count
            
        Returns:
            Bucket value (closest bucket >= target_tokens)
        """
        for bucket in self.buckets:
            if bucket >= target_tokens:
                return bucket
        return self.buckets[-1]
    
    def generate_bucket_samples(
        self,
        generator,
        bucket: int,
        num_samples: int,
        base_seed: int,
        **kwargs
    ) -> List[Tuple[int, dict]]:
        """Generate samples for a specific bucket
        
        Args:
            generator: Data generator instance
            bucket: Target token count for bucket
            num_samples: Number of samples to generate
            base_seed: Base random seed
            **kwargs: Additional generator parameters
            
        Returns:
            List of (sample_index, sample_dict) tuples
        """
        samples = []
        for i in range(num_samples):
            seed = base_seed + i * 1000 + bucket  # Ensure unique seeds
            sample = generator.generate_sample(bucket, seed, **kwargs)
            samples.append((i, sample))
        return samples
