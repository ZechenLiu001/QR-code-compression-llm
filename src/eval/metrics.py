"""Evaluation metrics"""

import json
from typing import Dict, Any


def json_exact_match(pred: str, gold: str) -> float:
    """JSON exact match (canonicalized comparison)
    
    - If both parse successfully: compare canonicalized JSON
    - If parse fails: fallback to strict string comparison
    
    Args:
        pred: Predicted JSON string
        gold: Ground truth JSON string
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    try:
        pred_obj = json.loads(pred)
        gold_obj = json.loads(gold)
        
        # Canonicalize: sort keys, compact format
        pred_canonical = json.dumps(pred_obj, sort_keys=True, separators=(',', ':'))
        gold_canonical = json.dumps(gold_obj, sort_keys=True, separators=(',', ':'))
        
        return 1.0 if pred_canonical == gold_canonical else 0.0
    except (json.JSONDecodeError, TypeError):
        # Fallback to string comparison
        return 1.0 if pred.strip() == gold.strip() else 0.0


def json_structural_f1(pred: str, gold: str) -> Dict[str, float]:
    """JSON structural F1 (flatten key-path)
    
    Flattens JSON to key-value pairs (including nested paths) and computes:
    - key_precision: fraction of predicted keys in gold
    - key_recall: fraction of gold keys in predictions
    - key_f1: harmonic mean of precision and recall
    - value_accuracy: fraction of matching keys with identical values
    - overall_f1: combined metric
    
    Args:
        pred: Predicted JSON string
        gold: Ground truth JSON string
        
    Returns:
        Dictionary with metrics
    """
    try:
        pred_obj = json.loads(pred)
        gold_obj = json.loads(gold)
    except (json.JSONDecodeError, TypeError):
        return {
            "key_precision": 0.0,
            "key_recall": 0.0,
            "key_f1": 0.0,
            "value_accuracy": 0.0,
            "overall_f1": 0.0,
        }
    
    # Flatten to key paths
    pred_keys = set(_flatten_keys(pred_obj))
    gold_keys = set(_flatten_keys(gold_obj))
    
    # Key metrics
    if len(pred_keys) == 0:
        key_precision = 0.0
    else:
        key_precision = len(pred_keys & gold_keys) / len(pred_keys)
    
    if len(gold_keys) == 0:
        key_recall = 0.0
    else:
        key_recall = len(pred_keys & gold_keys) / len(gold_keys)
    
    if key_precision + key_recall == 0:
        key_f1 = 0.0
    else:
        key_f1 = 2 * key_precision * key_recall / (key_precision + key_recall)
    
    # Value accuracy (for matching keys)
    matching_keys = pred_keys & gold_keys
    if len(matching_keys) == 0:
        value_accuracy = 0.0
    else:
        correct_values = 0
        for key in matching_keys:
            pred_val = _get_nested_value(pred_obj, key)
            gold_val = _get_nested_value(gold_obj, key)
            if pred_val == gold_val:
                correct_values += 1
        value_accuracy = correct_values / len(matching_keys)
    
    # Overall F1
    overall_f1 = (key_f1 + value_accuracy) / 2.0
    
    return {
        "key_precision": key_precision,
        "key_recall": key_recall,
        "key_f1": key_f1,
        "value_accuracy": value_accuracy,
        "overall_f1": overall_f1,
    }


def _flatten_keys(obj: Any, prefix: str = "") -> list:
    """Flatten JSON object to list of key paths"""
    keys = []
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{prefix}.{k}" if prefix else k
            keys.append(key_path)
            keys.extend(_flatten_keys(v, key_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            keys.append(key_path)
            keys.extend(_flatten_keys(v, key_path))
    
    return keys


def _get_nested_value(obj: Any, key_path: str) -> Any:
    """Get nested value from key path"""
    parts = key_path.split('.')
    current = obj
    
    for part in parts:
        if '[' in part:
            # Handle list indices
            base, indices = part.split('[', 1)
            if base:
                current = current[base]
            while ']' in indices:
                idx_str, rest = indices.split(']', 1)
                idx = int(idx_str)
                current = current[idx]
                if rest.startswith('['):
                    indices = rest[1:]
                else:
                    break
        else:
            current = current[part]
    
    return current


def needle_hit_at_1(pred: str, gold: str, fuzzy: bool = True) -> float:
    """Needle hit rate
    
    Args:
        pred: Predicted string
        gold: Ground truth needle string
        fuzzy: If True, allow gold to be substring of pred
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    pred_clean = pred.strip()
    gold_clean = gold.strip()
    
    if fuzzy:
        # Check if gold is substring of pred
        return 1.0 if gold_clean in pred_clean else 0.0
    else:
        # Exact match
        return 1.0 if pred_clean == gold_clean else 0.0
