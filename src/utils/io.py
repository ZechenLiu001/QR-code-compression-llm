"""IO utilities for config and data persistence"""

import json
import hashlib
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration for reproducibility
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SHA256 hash string (first 8 chars)
    """
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:8]


def save_prompt(
    sample_id: str,
    codec: str,
    augment_level: str,
    prompt_data: Dict[str, Any],
    output_dir: str = "outputs/prompts",
):
    """Save prompt data to JSON file
    
    Args:
        sample_id: Sample identifier
        codec: Codec type
        augment_level: Augmentation level
        prompt_data: Prompt data dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir) / f"{sample_id}_{codec}_{augment_level}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
