"""Training data generator for LoRA fine-tuning

生成 (编码图像, 原始文本) 配对用于训练模型学习解码
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from src.data.json_task import generate_json_sample
from src.data.needle_task import generate_needle_sample
from src.codec.render_context import RenderCodec
from src.codec.codebook_context import CodebookCodec
from src.codec.codebook_external import CodebookExternalCodec
from src.augment.image_augment import apply_augment
from src.utils.seed import set_seed


class TrainDataGenerator:
    """训练数据生成器
    
    为每种 codec 生成训练样本，包括:
    - 编码后的图像
    - 原始文本 (作为 ground truth)
    - 问题和答案
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "outputs/train_data",
        seed: int = 42
    ):
        """
        Args:
            config: 训练配置
            output_dir: 输出目录
            seed: 随机种子
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # 创建输出目录
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 codecs
        self._init_codecs()
        
        # 数据生成配置
        data_config = config.get("data", {}).get("generation", {})
        self.num_samples_per_codec = data_config.get("num_samples_per_codec", 500)
        self.codecs_to_use = data_config.get("codecs", ["render", "codebook", "codebook_external"])
        self.tasks = data_config.get("tasks", ["json_restore", "needle"])
        self.task_weights = data_config.get("task_weights", {"json_restore": 0.7, "needle": 0.3})
        self.length_buckets = data_config.get("length_buckets", [256, 512, 1024])
        self.train_val_split = data_config.get("train_val_split", 0.9)
        
        # 增强配置
        self.augment_config = config.get("augment", {})
        self.use_augment = self.augment_config.get("enabled", False)
    
    def _init_codecs(self):
        """初始化编码器"""
        codec_config = self.config.get("codec", {})
        
        self.codecs = {}
        
        # Render codec
        render_cfg = codec_config.get("render", {})
        self.codecs["render"] = RenderCodec(
            font_size=render_cfg.get("font_size", 12),
            line_spacing=render_cfg.get("line_spacing", 4),
            margin=render_cfg.get("margin", 10),
            bg_color=tuple(render_cfg.get("bg_color", [255, 255, 255])),
            text_color=tuple(render_cfg.get("text_color", [0, 0, 0])),
        )
        
        # Codebook codec
        codebook_cfg = codec_config.get("codebook", {})
        self.codecs["codebook"] = CodebookCodec(
            mode=codebook_cfg.get("mode", "bw"),
            cell_size=codebook_cfg.get("cell_size", 4),
            use_compression=codebook_cfg.get("use_compression", True),
        )
        
        # Codebook external (QR code)
        external_cfg = codec_config.get("codebook_external", {})
        self.codecs["codebook_external"] = CodebookExternalCodec(
            code_type=external_cfg.get("type", "qrcode"),
            error_correction=external_cfg.get("error_correction", "M"),
            box_size=external_cfg.get("box_size", 10),
            border=external_cfg.get("border", 4),
        )
    
    def _select_task(self) -> str:
        """根据权重随机选择任务"""
        tasks = list(self.task_weights.keys())
        weights = [self.task_weights[t] for t in tasks]
        return random.choices(tasks, weights=weights, k=1)[0]
    
    def _generate_sample(
        self,
        sample_id: str,
        codec_name: str,
        task: str,
        target_tokens: int,
        seed: int
    ) -> Optional[Dict[str, Any]]:
        """生成单个训练样本
        
        Args:
            sample_id: 样本 ID
            codec_name: 编码器名称
            task: 任务类型
            target_tokens: 目标 token 数
            seed: 随机种子
            
        Returns:
            样本字典，包含图像路径、文本、问题、答案
        """
        set_seed(seed)
        
        # 生成原始数据
        if task == "json_restore":
            data = generate_json_sample(
                target_tokens=target_tokens,
                field_count=random.randint(5, 15),
                seed=seed,
                nesting_depth=random.randint(1, 3)
            )
            question = data["query"]
            answer = data["answer"]
            context = data["context"]
        elif task == "needle":
            needle_position = random.random()  # 随机位置
            data = generate_needle_sample(
                target_tokens=target_tokens,
                needle_position=needle_position,
                seed=seed
            )
            question = data["query"]
            answer = data["answer"]
            context = data["context"]
        else:
            return None
        
        # 编码为图像
        codec = self.codecs.get(codec_name)
        if codec is None:
            return None
        
        try:
            encoded = codec.encode(context)
            
            # 确保是图像
            if isinstance(encoded, list):
                # 多图情况，取第一张
                image = encoded[0] if encoded else None
            elif isinstance(encoded, Image.Image):
                image = encoded
            else:
                return None
            
            if image is None:
                return None
            
            # 应用数据增强 (训练时)
            if self.use_augment:
                augment_params = {
                    "scale": random.uniform(
                        self.augment_config.get("scale_range", [0.95, 1.05])[0],
                        self.augment_config.get("scale_range", [0.95, 1.05])[1]
                    ),
                    "rotation": random.uniform(
                        self.augment_config.get("rotation_range", [-3, 3])[0],
                        self.augment_config.get("rotation_range", [-3, 3])[1]
                    ),
                }
                # 随机应用模糊
                if random.random() < self.augment_config.get("blur_prob", 0.1):
                    augment_params["blur"] = random.uniform(
                        self.augment_config.get("blur_radius_range", [0.5, 1.0])[0],
                        self.augment_config.get("blur_radius_range", [0.5, 1.0])[1]
                    )
                # 随机应用噪声
                if random.random() < self.augment_config.get("noise_prob", 0.1):
                    augment_params["noise"] = random.uniform(
                        self.augment_config.get("noise_std_range", [1, 5])[0],
                        self.augment_config.get("noise_std_range", [1, 5])[1]
                    )
                
                image = apply_augment(image, augment_params, seed=seed)
            
            # 保存图像
            image_filename = f"{sample_id}.png"
            image_path = self.images_dir / image_filename
            image.save(image_path)
            
            return {
                "id": sample_id,
                "codec": codec_name,
                "task": task,
                "target_tokens": target_tokens,
                "image_path": str(image_path),
                "image_filename": image_filename,
                "context": context,
                "question": question,
                "answer": answer,
            }
            
        except Exception as e:
            print(f"Error generating sample {sample_id}: {e}")
            return None
    
    def generate(self) -> Dict[str, List[Dict[str, Any]]]:
        """生成所有训练数据
        
        Returns:
            {"train": [...], "val": [...]}
        """
        set_seed(self.seed)
        
        all_samples = []
        sample_idx = 0
        
        for codec_name in self.codecs_to_use:
            print(f"Generating samples for codec: {codec_name}")
            
            for i in range(self.num_samples_per_codec):
                # 选择任务和长度
                task = self._select_task()
                target_tokens = random.choice(self.length_buckets)
                
                # 生成样本
                sample_id = f"{codec_name}_{task}_{sample_idx:05d}"
                sample_seed = self.seed + sample_idx
                
                sample = self._generate_sample(
                    sample_id=sample_id,
                    codec_name=codec_name,
                    task=task,
                    target_tokens=target_tokens,
                    seed=sample_seed
                )
                
                if sample is not None:
                    all_samples.append(sample)
                
                sample_idx += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{self.num_samples_per_codec} samples")
        
        # 打乱并分割
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * self.train_val_split)
        
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        print(f"Total samples: {len(all_samples)}")
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        
        return {
            "train": train_samples,
            "val": val_samples
        }
    
    def save(self, data: Dict[str, List[Dict[str, Any]]]):
        """保存数据到 JSONL 文件
        
        Args:
            data: {"train": [...], "val": [...]}
        """
        for split, samples in data.items():
            output_path = self.output_dir / f"{split}.jsonl"
            
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            print(f"Saved {len(samples)} samples to {output_path}")
    
    def generate_and_save(self) -> Dict[str, str]:
        """生成并保存所有数据
        
        Returns:
            {"train": train_path, "val": val_path}
        """
        data = self.generate()
        self.save(data)
        
        return {
            "train": str(self.output_dir / "train.jsonl"),
            "val": str(self.output_dir / "val.jsonl"),
            "images_dir": str(self.images_dir)
        }
