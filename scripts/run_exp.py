"""Run full experiment"""

import argparse
import json
import jsonlines
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path (must be before any src imports)
_this_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_this_file))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.model.loader import load_model
from src.data.json_task import generate_json_sample
from src.data.needle_task import generate_needle_sample
from src.data.length_bucket import LengthBucket
from src.codec.text_context import TextCodec
from src.codec.render_context import RenderCodec
from src.codec.codebook_context import CodebookCodec
from src.codec.codebook_external import CodebookExternalCodec
from src.augment.image_augment import apply_augment
from src.eval.evaluator import Evaluator
from src.utils.seed import set_seed
from src.utils.io import load_config, save_prompt, compute_config_hash
from src.utils.logger import setup_logger
# Import infer function
import importlib.util
spec = importlib.util.spec_from_file_location("infer", Path(__file__).parent / "infer.py")
infer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer_module)
infer = infer_module.infer

logger = setup_logger()


def create_codec(codec_config: dict):
    """Create codec from config"""
    codec_type = codec_config["type"]
    
    if codec_type == "text":
        return TextCodec()
    elif codec_type == "render":
        return RenderCodec(**{k: v for k, v in codec_config.items() if k != "type"})
    elif codec_type == "codebook":
        return CodebookCodec(**{k: v for k, v in codec_config.items() if k != "type"})
    elif codec_type == "codebook_external":
        return CodebookExternalCodec(**{k: v for k, v in codec_config.items() if k != "type"})
    else:
        raise ValueError(f"Unknown codec type: {codec_type}")


def run_experiment(config_path: str):
    """Run full experiment"""
    # Load config
    config = load_config(config_path)
    config_hash = compute_config_hash(config)
    logger.info(f"Config hash: {config_hash}")
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Load model
    logger.info("Loading model...")
    model, processor = load_model(**config["model"])
    logger.info("Model loaded")
    
    # Create codecs
    codecs = {}
    for codec_config in config["codecs"]:
        codec_type = codec_config["type"]
        codecs[codec_type] = create_codec(codec_config)
    
    # Create evaluator
    evaluator = Evaluator()
    
    # Setup output
    output_config = config.get("output", {})
    results_file = output_config.get("results_file", "outputs/results/results.jsonl")
    save_images = output_config.get("save_images", True)
    save_prompts = output_config.get("save_prompts", True)
    
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Get inference config
    inference_config = config.get("inference", {})
    max_new_tokens_map = inference_config.get("max_new_tokens", {})
    
    # Get data config
    data_config = config.get("data", {})
    tasks = data_config.get("tasks", ["json_restore"])
    length_buckets = data_config.get("length_buckets", [1024])
    samples_per_bucket = data_config.get("samples_per_bucket", 10)
    
    # Get augment config
    augment_config = config.get("augment", {})
    augment_enabled = augment_config.get("enabled", False)
    augment_levels = augment_config.get("levels", [{"name": "clean"}])
    
    # Run experiments
    results = []
    total_samples = len(tasks) * len(length_buckets) * samples_per_bucket * len(codecs) * len(augment_levels)
    
    with tqdm(total=total_samples, desc="Running experiments") as pbar:
        for task in tasks:
            for bucket in length_buckets:
                for sample_idx in range(samples_per_bucket):
                    # Generate sample
                    sample_seed = seed + bucket * 1000 + sample_idx * 100
                    
                    if task == "json_restore":
                        sample = generate_json_sample(bucket, field_count=10, seed=sample_seed)
                    elif task == "needle":
                        needle_pos = (sample_idx % 5) / 4.0  # Vary position
                        sample = generate_needle_sample(bucket, needle_pos, seed=sample_seed)
                    else:
                        continue
                    
                    context = sample["context"]
                    query = sample["query"]
                    answer = sample["answer"]
                    
                    # Count original text tokens
                    orig_text_tokens = len(context.split()) // 4  # Rough estimate
                    
                    for codec_name, codec in codecs.items():
                        # Encode context
                        encoded = codec.encode(context)
                        
                        # Determine context type
                        if isinstance(encoded, str):
                            context_type = "text"
                        else:
                            context_type = "image"
                        
                        # Get token cost estimate
                        token_cost = codec.get_token_cost(encoded, processor)
                        orig_text_tokens_actual = token_cost.get("text_tokens", orig_text_tokens)
                        
                        for aug_level_config in augment_levels:
                            aug_level_name = aug_level_config.get("name", "clean")
                            
                            # Apply augmentation if enabled
                            if augment_enabled and context_type == "image":
                                if isinstance(encoded, list):
                                    encoded_aug = [apply_augment(img, aug_level_config, seed=sample_seed) for img in encoded]
                                else:
                                    encoded_aug = apply_augment(encoded, aug_level_config, seed=sample_seed)
                            else:
                                encoded_aug = encoded
                            
                            # Save image if requested
                            if save_images and context_type == "image":
                                img_path = Path("outputs/images") / f"{task}_{bucket}_{sample_idx}_{codec_name}_{aug_level_name}.png"
                                img_path.parent.mkdir(parents=True, exist_ok=True)
                                if isinstance(encoded_aug, list):
                                    encoded_aug[0].save(str(img_path))
                                else:
                                    encoded_aug.save(str(img_path))
                            
                            # Run inference
                            max_new_tokens = max_new_tokens_map.get(task, 1024)
                            inference_cfg = inference_config.copy()
                            inference_cfg["max_new_tokens"] = max_new_tokens
                            
                            try:
                                # Use encoded_aug for image, encoded for text
                                context_for_infer = encoded_aug if context_type == "image" else encoded
                                
                                result = infer(
                                    model,
                                    processor,
                                    context_for_infer,
                                    query,
                                    context_type=context_type,
                                    config=inference_cfg,
                                    orig_text_tokens=orig_text_tokens_actual,
                                )
                                
                                prediction = result["answer"]
                                
                                # Evaluate
                                metrics = evaluator.evaluate(task, prediction, answer)
                                
                                # Save prompt if requested
                                if save_prompts:
                                    prompt_data = {
                                        "model_id": config["model"]["model_id"],
                                        "inference_config": inference_cfg,
                                        "context": context[:200] + "..." if len(context) > 200 else context,  # Truncate for storage
                                        "query": query,
                                        "token_cost": result["token_usage"],
                                    }
                                    save_prompt(
                                        f"{task}_{bucket}_{sample_idx}",
                                        codec_name,
                                        aug_level_name,
                                        prompt_data,
                                    )
                                
                                # Save result
                                result_entry = {
                                    "sample_id": f"{task}_{bucket}_{sample_idx}",
                                    "task": task,
                                    "codec": codec_name,
                                    "length_bucket": bucket,
                                    "augment_level": aug_level_name,
                                    "prediction": prediction,
                                    "ground_truth": answer,
                                    "metrics": metrics,
                                    "token_cost": result["token_usage"],
                                    "compression_ratio": result.get("compression_ratio"),
                                    "latency_ms": result.get("latency_ms", 0),
                                    "config_hash": config_hash,
                                }
                                
                                results.append(result_entry)
                                
                                # Write to file immediately
                                with open(results_file, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                                
                            except Exception as e:
                                logger.error(f"Error in inference: {e}", exc_info=True)
                            
                            pbar.update(1)
    
    logger.info(f"Experiment completed. Results saved to {results_file}")
    logger.info(f"Total samples: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Run image context compression experiment")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    
    args = parser.parse_args()
    
    run_experiment(args.config)


if __name__ == "__main__":
    main()
