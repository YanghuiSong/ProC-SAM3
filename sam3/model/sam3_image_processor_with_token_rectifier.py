# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Image processor for SAM3 with token rectification support.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple
from PIL import Image

# 根据常见SAM模型结构，尝试多种可能的导入路径
def _try_import_resize_longest_side():
    import importlib
    possible_paths = [
        ('sam3.utils.transforms', 'ResizeLongestSide'),
        ('..utils.transforms', 'ResizeLongestSide'),
        ('sam3.model.utils.transforms', 'ResizeLongestSide'),
        ('..model.utils.transforms', 'ResizeLongestSide'),
        ('sam2.utils.transforms', 'ResizeLongestSide'),
    ]
    
    for module_path, class_name in possible_paths:
        try:
            module = importlib.import_module(module_path, package=__package__)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    
    # 如果所有导入都失败，定义一个简单的实现
    class ResizeLongestSide:
        def __init__(self, target_length: int):
            self.target_length = target_length

        def apply_image_torch(self, image_tensor: torch.Tensor) -> torch.Tensor:
            """
            Expects a torch tensor with shape BxCxHxW and applies the resize transformation.
            """
            _, _, h, w = image_tensor.shape
            
            # 确定最长边并缩放到目标长度
            if h >= w:
                new_h = self.target_length
                new_w = int(w * (self.target_length / h))
            else:
                new_w = self.target_length
                new_h = int(h * (self.target_length / w))
            
            # 使用双线性插值调整图像大小
            resized_image = F.interpolate(
                image_tensor,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            
            return resized_image
    
    return ResizeLongestSide

ResizeLongestSide = _try_import_resize_longest_side()


def get_sdpa_settings(prefer_flash_attention: bool):
    """
    Return SDPA settings based on the preferred attention mechanism.
    """
    try:
        # Try to use flash attention if available and preferred
        from torch.backends.cuda import sdp_kernel, SDPBackend
        flash_attn_available = True
    except ImportError:
        flash_attn_available = False
    
    if prefer_flash_attention and flash_attn_available:
        # Using flash attention
        return True, {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False}
    else:
        # Using standard attention
        return False, {"enable_math": True, "enable_flash": False, "enable_mem_efficient": True}


from ..model.vitdet import window_partition  # 替换 .image_encoder

def prepare_memory_attention_input(
    backbone_feats: List[torch.Tensor],
    queries: torch.Tensor,
    masks: torch.Tensor = None,
    multimask_output: bool = False,
    spatial_repeat_times: int = 1,
    downsample_ratio: float = None,
    shuffling_mode: str = "none",
    pos_enc_at_last: bool = True,
):
    """
    Prepares memory and attention input for the model based on backbone features,
    queries, and optional masks.
    """
    # 获取最后一层的特征作为内存特征
    high_res_feats = backbone_feats[-1]
    B, C, H, W = high_res_feats.shape
    
    # 处理掩码（如果提供）
    if masks is not None:
        # 应用mask解码器
        if multimask_output:
            # 如果启用多掩码输出，则处理多个掩码
            masks = masks[:, 0, ...]  # 取第一个掩码
        
        # 调整掩码尺寸以匹配特征图大小
        masks = F.interpolate(
            masks.unsqueeze(1).float(), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
    
    # 创建空间位置编码
    pos_enc = torch.zeros(1, C, H, W, device=queries.device, dtype=queries.dtype)
    
    # 如果启用了shuffle模式，则应用相应的处理
    if shuffling_mode == "window_partition":
        # 对特征进行窗口划分
        high_res_feats = window_partition(high_res_feats, window_size=7)
    
    # 如果需要重复空间特征
    if spatial_repeat_times > 1:
        # 重复特征映射以增加时间维度
        high_res_feats = high_res_feats.repeat(spatial_repeat_times, 1, 1, 1)
        pos_enc = pos_enc.repeat(spatial_repeat_times, 1, 1, 1)
    
    # 准备返回的内存相关参数
    memory_output = {
        "vision_features": high_res_feats,
        "vision_pos_enc": pos_enc,
        "backbone_out": {"feats": backbone_feats},
        "obj_ptrs": queries,
    }
    
    return memory_output


class Sam3ImageProcessorWithTokenRectifier:
    def __init__(
        self,
        model,
        confidence_threshold: float = 0.5,
        device="cuda",
        use_token_rectifier: bool = False,
        token_rectifier_alpha: float = 0.15,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_token_rectifier = use_token_rectifier
        self.token_rectifier_alpha = token_rectifier_alpha
        
        # Initialize token rectifier if enabled
        if use_token_rectifier:
            from .token_rectifier import DirectMaskRectifier
            self.mask_rectifier = DirectMaskRectifier(
                alpha=token_rectifier_alpha,
                enable=True
            )
        
        # Set up SDPA attention params
        # 检查模型是否具有_prefer_flash_attention属性，否则使用默认值
        prefer_flash_attention = getattr(model, '_prefer_flash_attention', False)
        flash_attn_enabled = get_sdpa_settings(prefer_flash_attention)[0]
        self.set_mask_downsample_ratio = (
            model.sparse_neck.set_mask_downsample_ratio
            if flash_attn_enabled and hasattr(model, "sparse_neck") and hasattr(model.sparse_neck, "set_mask_downsample_ratio")
            else None
        )

    def set_image(self, image: np.ndarray) -> Dict:
        """
        Preprocess an image for input to the model.
        """
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[-1] == 4:
            image = image[..., :3]

        # Get original image dimensions
        self._orig_hw = (image.shape[0], image.shape[1])
        
        # Convert to tensor and use the same preprocessing as the original processor
        import torchvision.transforms.v2 as v2
        transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),  # Resize to square as in original
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image = transform(image_tensor).unsqueeze(0).to(self.device)
        
        # Run image encoder - 使用backbone的forward_image方法
        backbone_out = self.model.backbone.forward_image(image)
        
        # 从backbone_out中提取所需的特征
        backbone_feats = backbone_out["backbone_fpn"]
        
        # 检查模型的num_feature_levels属性，并确保我们使用的特征数量与之匹配
        num_expected_levels = getattr(self.model, 'num_feature_levels', 1)  # 默认为1
        
        # 使用正确的特征级别数量，取backbone输出的最后几个层级
        if isinstance(backbone_feats, list):
            # 如果backbone_feats是一个列表，只取期望的数量
            if len(backbone_feats) >= num_expected_levels:
                img_feats = backbone_feats[-num_expected_levels:]  # 取最后num_expected_levels个
            else:
                # 如果backbone输出的特征层级不够，复制最后一个特征层
                img_feats = list(backbone_feats)
                while len(img_feats) < num_expected_levels:
                    img_feats.insert(0, img_feats[0])  # 在前面插入第一个特征层
        else:
            # 如果不是列表，转换为列表形式
            img_feats = [backbone_feats]
        
        # 确保位置编码也与特征数量匹配
        img_pos_embeds = backbone_out["vision_pos_enc"]
        if isinstance(img_pos_embeds, list):
            if len(img_pos_embeds) >= num_expected_levels:
                img_pos_embeds = img_pos_embeds[-num_expected_levels:]
            else:
                # 如果位置编码数量不够，复制第一个
                pos_embeds = list(img_pos_embeds)
                while len(pos_embeds) < num_expected_levels:
                    pos_embeds.insert(0, pos_embeds[0])
                img_pos_embeds = pos_embeds
        else:
            # 如果不是列表，复制成期望的数量
            img_pos_embeds = [img_pos_embeds] * num_expected_levels
        
        # 获取视觉特征尺寸
        vis_feat_sizes = [feat.shape[-2:] for feat in img_feats]
        
        # 正确处理多尺度特征，展平并准备transformer encoder的输入
        flattened_img_feats = []
        flattened_pos_embeds = []
        
        for src, pos in zip(img_feats, img_pos_embeds):
            # 将特征展平为 [H*W, B, C] 格式
            bs, c, h, w = src.shape
            src_flatten = src.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
            pos_flatten = pos.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
            
            flattened_img_feats.append(src_flatten)
            flattened_pos_embeds.append(pos_flatten)
        
        # 创建虚拟prompt用于运行encoder (使用零张量作为占位符)
        batch_size = image.shape[0]
        num_prompts = 1  # 假设只有一个查询
        d_model = self.model.transformer.d_model  # 获取模型的隐藏维度
        
        prompt = torch.zeros((num_prompts, batch_size, d_model), device=self.device)
        prompt_mask = torch.zeros((batch_size, num_prompts), device=self.device, dtype=torch.bool)
        
        # 直接调用transformer encoder，使用正确的参数格式
        memory = self.model.transformer.encoder(
            src=flattened_img_feats,  # 使用展平后的特征
            prompt=prompt,
            src_pos=flattened_pos_embeds,  # 展平后的位置编码
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes  # 特征尺寸
        )
        
        encoder_hidden_states = memory["memory"]
        
        # Prepare memory and attention input
        mem_out = prepare_memory_attention_input(
            backbone_feats=backbone_feats,
            queries=encoder_hidden_states,
            masks=None,
            multimask_output=False,
            spatial_repeat_times=1,  # 使用默认值1，而不是访问不存在的属性
            downsample_ratio=self.set_mask_downsample_ratio,
            shuffling_mode="none",  # 使用默认值"none"，而不是访问不存在的属性
            pos_enc_at_last=True,
        )
        
        # Create inference state
        inference_state = {
            "images": image,
            "original_size": self._orig_hw,
            "transformed_size": tuple(image.shape[-2:]),
            "is_hq": False,  # 默认设置为False，因为sam_prompt_encoder不存在
            "point_coords": [],
            "point_labels": [],
            "memories": [],
            "memory_temporal_stride": 1,  # 设置默认值
            "num_mask_tokens": self.model.sam_mask_decoder.iou_prediction_head.num_mask_tokens if hasattr(self.model, 'sam_mask_decoder') and hasattr(self.model.sam_mask_decoder, 'iou_prediction_head') else 1,
            "iou_predictions": [],
            "masks_logits": [],
            "sam_tokens": [],
            "object_score": [],
            "semantic_mask_logits": None,
            "presence_score": None,
            "pixel_embed": None,
            **mem_out,
        }
        
        # Process universal segmentation head
        # 注意：如果模型没有universal_seg_head，则跳过这部分
        if hasattr(self.model, 'universal_seg_head'):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                universal_seg_out = self.model.universal_seg_head(
                    backbone_feats=backbone_feats,
                    obj_queries=encoder_hidden_states,
                    image_ids=torch.tensor([0], device=self.device),
                    encoder_hidden_states=encoder_hidden_states,
                )
                
                inference_state["semantic_mask_logits"] = universal_seg_out["semantic_seg"].float()
                inference_state["presence_score"] = universal_seg_out["presence_logit"].float()  # 确保返回浮点数
        else:
            # 如果模型没有universal_seg_head，则设置默认值
            inference_state["semantic_mask_logits"] = None
            inference_state["presence_score"] = torch.tensor(0.0, device=self.device)
        
        return inference_state

    def set_text_prompt(self, state: Dict, prompt: str) -> Dict:
        """
        Set a text prompt for the model to segment.
        """
        # 检查模型是否有文本相关组件
        if not hasattr(self.model, 'tokenizer') or not hasattr(self.model, 'text_encoder'):
            # 如果模型不支持文本处理，返回原状态
            print(f"Warning: Model does not support text processing. Skipping text prompt: {prompt}")
            return state

        # Tokenize the prompt
        tokenized_output = self.model.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.model.max_tokens,
            return_tensors="pt",
        )
        tokenized_text = tokenized_output.input_ids.to(self.device)
        attention_mask = tokenized_output.attention_mask.to(self.device)

        # Get text embeddings
        text_token_mask = self.model.text_encoder(
            tokenized_text, attention_mask=attention_mask
        )
        text_embed = text_token_mask["text_embed"]
        
        # Normalize text embed
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        # Generate masks using the universal segmentation head
        backbone_feats = state["backbone_out"]["feats"]
        encoder_hidden_states = state["vision_pos_embed"]
        
        # 检查模型是否有universal_seg_head
        if hasattr(self.model, 'universal_seg_head'):
            # Process universal segmentation head with text guidance
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                universal_seg_out = self.model.universal_seg_head(
                    backbone_feats=backbone_feats,
                    obj_queries=text_embed.unsqueeze(0),
                    image_ids=torch.tensor([0], device=self.device),
                    encoder_hidden_states=encoder_hidden_states,
                    prompt=text_embed.unsqueeze(0),
                    prompt_mask=1 - attention_mask,
                )
                
                # Get original masks
                original_masks = universal_seg_out["pred_masks"].float()
                
                # Apply token rectification to the masks if enabled
                if self.use_token_rectifier:
                    from .token_rectifier import DirectMaskRectifier
                    
                    # Get the pixel embedding for guidance
                    if hasattr(self.model.universal_seg_head, '_embed_pixels'):
                        pixel_embed = self.model.universal_seg_head._embed_pixels(
                            backbone_feats=backbone_feats,
                            image_ids=torch.tensor([0], device=self.device),
                            encoder_hidden_states=encoder_hidden_states,
                        )
                        
                        # Apply direct mask rectification
                        mask_rectifier = DirectMaskRectifier(
                            alpha=self.token_rectifier_alpha,
                            enable=True
                        )
                        
                        rectified_masks = mask_rectifier(original_masks, pixel_embed)
                        state["masks_logits"] = rectified_masks
                    else:
                        state["masks_logits"] = original_masks
                else:
                    state["masks_logits"] = original_masks
                
                # Process masks to determine object scores
                masks = (state["masks_logits"][0] > 0).float()
                object_scores = torch.sum(masks, dim=[1, 2]) / (masks.shape[1] * masks.shape[2])
                state["object_score"] = object_scores
        else:
            # 如果模型没有universal_seg_head，跳过文本处理
            print("Warning: Model does not have universal_seg_head. Text prompt processing skipped.")
        
        return state

    def reset_all_prompts(self, state: Dict):
        """
        Reset all prompts for a new round of segmentation.
        """
        state["point_coords"] = []
        state["point_labels"] = []
        state["masks_logits"] = []
        state["sam_tokens"] = []
        state["object_score"] = []
        return state