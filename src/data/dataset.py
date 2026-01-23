"""PyTorch Dataset for LoRA training

加载训练数据，返回模型输入格式
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset


class CodebookDataset(Dataset):
    """Codebook 解码训练数据集
    
    加载 (图像, 问题, 答案) 三元组，用于训练模型学习从图像中还原文本
    """
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 2048,
        images_base_dir: Optional[str] = None,
    ):
        """
        Args:
            data_path: JSONL 数据文件路径
            processor: Qwen2-VL processor
            max_length: 最大序列长度
            images_base_dir: 图像基础目录 (如果 image_path 是相对路径)
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.max_length = max_length
        self.images_base_dir = Path(images_base_dir) if images_base_dir else None
        
        # 加载数据
        self.samples = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 JSONL 数据"""
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_image_path(self, sample: Dict[str, Any]) -> Path:
        """获取图像完整路径"""
        image_path = sample.get("image_path", "")
        
        if Path(image_path).is_absolute():
            return Path(image_path)
        
        # 如果是相对路径，尝试多种方式
        if self.images_base_dir:
            # 使用 images_base_dir
            full_path = self.images_base_dir / sample.get("image_filename", image_path)
            if full_path.exists():
                return full_path
        
        # 尝试相对于数据文件
        full_path = self.data_path.parent / "images" / sample.get("image_filename", "")
        if full_path.exists():
            return full_path
        
        # 最后尝试原始路径
        return Path(image_path)
    
    def _build_messages(self, sample: Dict[str, Any], image_path: Path) -> List[Dict]:
        """构建 chat 格式的消息
        
        格式:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "file://path/to/image.png"},
                    {"type": "text", "text": "问题"}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": "答案"}
                ]
            }
        ]
        """
        question = sample.get("question", "请还原图像中的内容。")
        answer = sample.get("answer", "")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        
        return messages
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本
        
        Returns:
            包含 input_ids, attention_mask, pixel_values, labels 的字典
        """
        sample = self.samples[idx]
        image_path = self._get_image_path(sample)
        
        # 构建消息
        messages = self._build_messages(sample, image_path)
        
        # 使用 processor 处理
        # 注意: Qwen2-VL 的处理方式
        try:
            # 应用 chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            
            # 处理输入
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移除 batch 维度
            result = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # 创建 labels (用于计算 loss)
            # 对于 causal LM，labels = input_ids，但需要 mask 掉 prompt 部分
            result["labels"] = result["input_ids"].clone()
            
            # 找到 assistant 回复的起始位置，mask 掉之前的部分
            # Qwen2-VL 使用 <|im_start|>assistant 作为标记
            input_ids = result["input_ids"]
            
            # 简化处理: 找到最后一个 assistant 标记后的位置
            # 实际实现可能需要更精确的处理
            assistant_token_id = self.processor.tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )
            
            # 找到 assistant 标记位置
            mask_end = 0
            for i in range(len(input_ids) - len(assistant_token_id)):
                if input_ids[i:i+len(assistant_token_id)].tolist() == assistant_token_id:
                    # 找到 assistant 标记，mask 到这个位置之后
                    mask_end = i + len(assistant_token_id)
                    # 继续找，取最后一个
            
            # 将 prompt 部分的 labels 设为 -100 (忽略)
            if mask_end > 0:
                result["labels"][:mask_end] = -100
            
            return result
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回一个空样本
            return {
                "input_ids": torch.zeros(1, dtype=torch.long),
                "attention_mask": torch.zeros(1, dtype=torch.long),
                "labels": torch.zeros(1, dtype=torch.long),
            }


class DataCollatorForCodebook:
    """Data collator for codebook training
    
    处理 batch padding 和 pixel_values 的特殊处理
    """
    
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = processor.tokenizer.pad_token_id or 0
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch
        
        Args:
            features: List of feature dicts from dataset
            
        Returns:
            Batched tensors
        """
        # 分离不同类型的数据
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        
        for f in features:
            input_ids_list.append(f["input_ids"])
            attention_mask_list.append(f["attention_mask"])
            labels_list.append(f["labels"])
            
            if "pixel_values" in f:
                pixel_values_list.append(f["pixel_values"])
            if "image_grid_thw" in f:
                image_grid_thw_list.append(f["image_grid_thw"])
        
        # Pad sequences
        max_len = max(ids.size(0) for ids in input_ids_list)
        max_len = min(max_len, self.max_length)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for input_ids, attention_mask, labels in zip(
            input_ids_list, attention_mask_list, labels_list
        ):
            # Truncate if needed
            if input_ids.size(0) > max_len:
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
            
            # Pad
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=attention_mask.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=labels.dtype)
                ])
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        result = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }
        
        # 处理 pixel_values (如果有)
        if pixel_values_list:
            # pixel_values 可能有不同的形状，需要特殊处理
            # 对于 Qwen2-VL，pixel_values 是 [num_patches, channels, height, width]
            # 简单起见，我们 concatenate 它们
            try:
                result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            except:
                # 如果形状不一致，保持 list
                result["pixel_values"] = pixel_values_list
        
        if image_grid_thw_list:
            try:
                result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
            except:
                result["image_grid_thw"] = image_grid_thw_list
        
        return result
