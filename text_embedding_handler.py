import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Optional, Union
import pickle
import os
from sam3.model.data_misc import FindStage


class TextEmbeddingHandler:
    """
    用于处理SAM3文本嵌入的工具类
    将文本提示转换为可重复使用的文本嵌入，并通过SAM3接口实现分割
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        bpe_path: str, 
        device: str = "cuda"
    ):
        """
        初始化TextEmbeddingHandler
        
        Args:
            checkpoint_path: SAM3模型检查点路径
            bpe_path: BPE词汇表路径
            device: 设备名称 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device)
        
        # 导入SAM3模型
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # 构建SAM3模型
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        
        # 创建处理器
        self.processor = Sam3Processor(self.model, device=self.device)
        
        # 文本嵌入缓存
        self.text_embeddings_cache = {}
        
    def encode_text_prompts(self, text_prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        将文本提示编码为嵌入
        
        Args:
            text_prompts: 文本提示列表
            
        Returns:
            包含语言特征和掩码的字典
        """
        # 创建缓存键
        cache_key = "_".join(sorted(text_prompts))
        
        if cache_key in self.text_embeddings_cache:
            print(f"Using cached text embeddings for {len(text_prompts)} prompts")
            return self.text_embeddings_cache[cache_key]
        
        print(f"Computing text embeddings for {len(text_prompts)} prompts...")
        
        # 使用SAM3的文本编码器
        with torch.no_grad():
            text_outputs = self.model.backbone.forward_text(text_prompts, device=self.device)
        
        # 缓存结果
        self.text_embeddings_cache[cache_key] = text_outputs
        print(f"Cached text embeddings for {len(text_prompts)} prompts")
        
        return text_outputs
    
    def segment_with_text_prompts(
        self, 
        image: Image.Image, 
        text_prompts: List[str],
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        使用文本提示对图像进行分割
        
        Args:
            image: 输入图像
            text_prompts: 文本提示列表
            confidence_threshold: 置信度阈值
            
        Returns:
            分割结果字典
        """
        # 设置图像
        inference_state = self.processor.set_image(image)
        
        # 编码文本提示
        text_outputs = self.encode_text_prompts(text_prompts)
        
        # 更新backbone_out中的语言特征
        inference_state["backbone_out"].update(text_outputs)
        
        # 创建FindStage输入
        num_prompts = len(text_prompts)
        find_stage = FindStage(
            img_ids=torch.zeros(1, device=self.device, dtype=torch.long),
            text_ids=torch.arange(num_prompts, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        
        # 更新处理器的find_stage
        original_find_stage = self.processor.find_stage
        self.processor.find_stage = find_stage
        
        # 初始化几何提示
        if "geometric_prompt" not in inference_state:
            inference_state["geometric_prompt"] = self.model._get_dummy_prompt()
        
        try:
            # 执行推理
            result = self.processor._forward_grounding(inference_state)
        finally:
            # 恢复原始find_stage
            self.processor.find_stage = original_find_stage
        
        return result
    
    def build_prob_map(
        self,
        outputs: Dict,
        height: int,
        width: int,
        confidence_threshold: float = 0.1,
        topk_inst: int = 100
    ) -> torch.Tensor:
        """
        构建概率图
        
        Args:
            outputs: 模型输出
            height: 图像高度
            width: 图像宽度
            confidence_threshold: 置信度阈值
            topk_inst: 保留的实例数量
            
        Returns:
            概率图张量
        """
        # 获取掩码logits
        masks_logits = outputs.get('masks_logits', None)
        semantic_logits = outputs.get('semantic_mask_logits', None)
        
        if masks_logits is not None and masks_logits.shape[0] > 0:
            # 处理实例分割结果
            num_classes = masks_logits.shape[0]
            prob_map = torch.zeros((num_classes, height, width), device=self.device)
            
            for i in range(min(masks_logits.shape[0], topk_inst)):
                mask = masks_logits[i]
                
                # 调整尺寸
                if mask.shape != (height, width):
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                # 应用置信度阈值
                obj_score = outputs.get('object_score', torch.ones(mask.shape[0], device=self.device))[i] if i < len(outputs.get('object_score', [])) else 1.0
                mask = mask * obj_score
                
                prob_map[i] = torch.maximum(prob_map[i], mask)
        
        elif semantic_logits is not None:
            # 处理语义分割结果
            if semantic_logits.shape[0] != height or semantic_logits.shape[1] != width:
                semantic_logits = F.interpolate(
                    semantic_logits.unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            prob_map = semantic_logits
        else:
            # 如果没有可用的输出，则返回零张量
            num_classes = len(self.last_text_prompts) if hasattr(self, 'last_text_prompts') else 1
            prob_map = torch.zeros((num_classes, height, width), device=self.device)
        
        return prob_map
    
    def save_embeddings_cache(self, cache_path: str):
        """
        保存嵌入缓存到文件
        
        Args:
            cache_path: 缓存文件路径
        """
        with open(cache_path, 'wb') as f:
            pickle.dump(self.text_embeddings_cache, f)
        print(f"Saved text embeddings cache to {cache_path}")
    
    def load_embeddings_cache(self, cache_path: str):
        """
        从文件加载嵌入缓存
        
        Args:
            cache_path: 缓存文件路径
        """
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.text_embeddings_cache = pickle.load(f)
            print(f"Loaded text embeddings cache from {cache_path}")
        else:
            print(f"Cache file {cache_path} does not exist")
    
    def clear_embeddings_cache(self):
        """清除嵌入缓存"""
        self.text_embeddings_cache.clear()
        print("Cleared text embeddings cache")


# 使用示例
if __name__ == "__main__":
    # 示例使用
    handler = TextEmbeddingHandler(
        checkpoint_path="/path/to/sam3.pt",  # 替换为实际路径
        bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz",  # 替换为实际路径
        device="cuda"
    )
    
    # 准备文本提示
    text_prompts = ["person", "car", "tree", "building"]
    
    # 加载图像
    image_path = "/path/to/your/image.jpg"  # 替换为实际图像路径
    image = Image.open(image_path).convert("RGB")
    
    # 执行分割
    outputs = handler.segment_with_text_prompts(image, text_prompts)
    
    # 构建概率图
    prob_map = handler.build_prob_map(
        outputs, 
        image.height, 
        image.width,
        confidence_threshold=0.1,
        topk_inst=100
    )
    
    print(f"Generated probability map with shape: {prob_map.shape}")