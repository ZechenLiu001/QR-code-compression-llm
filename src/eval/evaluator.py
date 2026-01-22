"""Unified evaluator"""

from typing import Dict, Any
from .metrics import json_exact_match, json_structural_f1, needle_hit_at_1


class Evaluator:
    """Unified evaluator for different tasks"""
    
    def evaluate(self, task: str, prediction: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate prediction against ground truth
        
        Args:
            task: Task type ("json_restore" or "needle")
            prediction: Model prediction
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        if task == "json_restore":
            return self._evaluate_json(prediction, ground_truth)
        elif task == "needle":
            return self._evaluate_needle(prediction, ground_truth)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _evaluate_json(self, pred: str, gold: str) -> Dict[str, Any]:
        """Evaluate JSON restore task"""
        exact = json_exact_match(pred, gold)
        structural = json_structural_f1(pred, gold)
        
        return {
            "json_exact_match": exact,
            **structural,
        }
    
    def _evaluate_needle(self, pred: str, gold: str) -> Dict[str, Any]:
        """Evaluate needle-in-haystack task"""
        hit = needle_hit_at_1(pred, gold, fuzzy=True)
        
        return {
            "hit_at_1": hit,
        }
