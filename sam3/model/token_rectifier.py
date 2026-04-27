import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelGuidedTokenRectifier(nn.Module):
    """
    Training-free token rectification module.
    Now operates on instance_embeds for actual mask generation impact.
    """

    def __init__(self, alpha: float = 0.15, enable: bool = True):
        super().__init__()
        self.alpha = alpha
        self.enable = enable

    def forward(
        self,
        instance_embeds: torch.Tensor,             # [B, C, H, W] - pixel-level features used for mask generation
        pixel_guidance: torch.Tensor,              # [B, C, H, W] - guidance from pixel features
    ) -> torch.Tensor:
        if not self.enable:
            return instance_embeds

        B, C, H, W = instance_embeds.shape

        # Apply pixel-guided rectification to instance embeddings
        # This directly affects the mask generation since instance_embeds are used in mask prediction
        refined_instance_embeds = (
            (1.0 - self.alpha) * instance_embeds
            + self.alpha * pixel_guidance
        )

        return refined_instance_embeds


class PixelGuidedAttentionRectifier(nn.Module):
    """
    Alternative approach: Apply rectification in attention mechanism
    This modifies the attention weights based on pixel-level guidance
    """
    
    def __init__(self, alpha: float = 0.15, enable: bool = True):
        super().__init__()
        self.alpha = alpha
        self.enable = enable
        
    def forward(
        self,
        attention_weights: torch.Tensor,      # [B, num_heads, seq_len, seq_len]
        pixel_guidance: torch.Tensor,         # [B, C, H, W] - flattened to match attention dims
    ) -> torch.Tensor:
        if not self.enable:
            return attention_weights
            
        B, num_heads, seq_len, seq_len = attention_weights.shape
        
        # Get the shape parameters from pixel_guidance
        B_pg, C, H, W = pixel_guidance.shape  # Added definition of C from pixel_guidance shape
        
        # Reshape pixel guidance to match attention dimensions
        # This is a simplified approach - in practice would need to match specific attention dimensions
        px_flat = pixel_guidance.view(B_pg, C, -1)  # [B, C, H*W]
        
        # Create a spatial attention bias based on pixel features
        # Average across channels to get spatial importance
        spatial_bias = torch.mean(px_flat, dim=1, keepdim=True)  # [B, 1, H*W]
        
        # Normalize to prevent large value changes
        spatial_bias = F.normalize(spatial_bias, p=2, dim=-1)
        
        # Expand to match attention dimensions
        # Simplified implementation - assumes attention is between patches
        patch_size = int(seq_len ** 0.5)  # Assuming square patches
        if patch_size * patch_size == seq_len:
            spatial_bias = F.interpolate(
                spatial_bias.view(B_pg, 1, H, W),
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            ).view(B_pg, 1, -1)
            
            # Create symmetric bias matrix
            spatial_bias_matrix = torch.bmm(
                spatial_bias.transpose(-1, -2), 
                spatial_bias
            )  # [B, H*W, H*W] -> [B, seq_len, seq_len]
            
            # Apply the bias to attention weights
            rectified_attention = attention_weights + self.alpha * spatial_bias_matrix
            
            return rectified_attention
        else:
            # If not a perfect square, return original attention
            return attention_weights


class DirectMaskRectifier(nn.Module):
    """
    Direct approach: Apply rectification to generated masks using pixel guidance
    This operates after mask generation but before final application
    """
    
    def __init__(self, alpha: float = 0.15, enable: bool = True):
        super().__init__()
        self.alpha = alpha
        self.enable = enable
        
    def forward(
        self,
        mask_logits: torch.Tensor,           # [B, Q, H, W] - mask predictions
        pixel_guidance: torch.Tensor,        # [B, C, H, W] - pixel-level guidance
    ) -> torch.Tensor:
        if not self.enable:
            return mask_logits
            
        # Use pixel guidance to create a spatial gate for the masks
        # Average across channels to get spatial importance map
        spatial_gate = torch.mean(torch.abs(pixel_guidance), dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Normalize the gate to [0, 1] range
        spatial_gate = torch.sigmoid(spatial_gate)
        
        # Apply the gate to mask logits
        # This amplifies or suppresses mask values based on pixel-level features
        rectified_masks = (
            (1.0 - self.alpha) * mask_logits +
            self.alpha * mask_logits * spatial_gate
        )
        
        return rectified_masks