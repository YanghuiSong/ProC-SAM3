# feedback_module.py
"""
Feedback-Guided Query Calibration Framework (FGQC) for Remote Sensing Open-Vocabulary Semantic Segmentation.
This module implements a lightweight, reversible feedback mechanism that calibrates queries based on decoder outputs,
without modifying encoder features or requiring CPU post-processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes
import warnings
warnings.filterwarnings('ignore')


class FGQCCalibrationModule(nn.Module):
    """
    Feedback-Guided Query Calibration Framework (FGQC) for Remote Sensing OVSS.
    
    Implements a lightweight, reversible feedback mechanism that calibrates queries based on decoder outputs,
    without modifying encoder features or requiring CPU post-processing.
    
    The feedback is defined as:
    "Decoder results → Query/Prompt-level calibration"
    
    Key constraints:
    - Does not modify encoder/backbone features
    - Does not modify image tokens
    - Does not perform multi-round refinements
    - Does not use CPU post-processing
    - Operates entirely on GPU with tensor operations
    - Maintains zero-mean and reversible properties
    """
    def __init__(self, 
                 uncertainty_threshold=0.5,
                 consistency_threshold=0.3,
                 alpha_uncertainty=1.0,
                 beta_consistency=1.0,
                 lambda_cal=0.05,
                 min_area_ratio=0.001,
                 max_area_ratio=0.5,
                 max_perimeter_area_ratio=100.0):
        super(FGQCCalibrationModule, self).__init__()
        
        self.uncertainty_threshold = uncertainty_threshold
        self.consistency_threshold = consistency_threshold
        self.alpha_uncertainty = alpha_uncertainty
        self.beta_consistency = beta_consistency
        self.lambda_cal = lambda_cal
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_perimeter_area_ratio = max_perimeter_area_ratio
        
        print("FGQCCalibrationModule initialized:")
        print("- Calibrates queries based on decoder outputs only")
        print("- Does not modify encoder/decoder tokens or attention mechanisms")
        print("- Maintains zero-mean and reversible properties")
        print("- Operates entirely on GPU with tensor operations")
        print("- Applies conditional calibration based on uncertainty thresholds")

    def compute_uncertainty_feedback(self, mask_logits):
        """
        Compute uncertainty feedback from decoder outputs.
        Based on entropy, foreground ratio, and area consistency.
        """
        with torch.no_grad():
            # Sigmoid probabilities
            probs = torch.sigmoid(mask_logits)  # (N, H, W)
            N, H, W = probs.shape
            
            # 1. Entropy-based uncertainty
            eps = 1e-8
            p_log_p = probs * torch.log(probs + eps)
            n_p_log_1_p = (1 - probs) * torch.log(1 - probs + eps)
            entropy = -(p_log_p + n_p_log_1_p)  # (N, H, W)
            entropy_norm = entropy / torch.log(torch.tensor(2.0))  # Normalize to [0, 1]
            
            # 2. Foreground ratio uncertainty (deviation from expected)
            fg_ratios = probs.mean(dim=[1, 2])  # (N,) - mean over H, W
            expected_fg = 0.5  # Assumed baseline
            fg_deviation = torch.abs(fg_ratios - expected_fg) / expected_fg  # (N,)
            
            # Combine uncertainties per instance
            entropy_per_instance = entropy_norm.mean(dim=[1, 2])  # (N,)
            uncertainty = entropy_per_instance + fg_deviation  # (N,)
            
            # Normalize to [0, 1]
            uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
            
            return uncertainty  # (N,)

    def compute_consistency_feedback(self, mask_logits):
        """
        Compute consistency feedback from decoder outputs.
        Measures spatial consistency of mask predictions.
        """
        with torch.no_grad():
            probs = torch.sigmoid(mask_logits)  # (N, H, W)
            N, H, W = probs.shape
            
            # 1. Spatial variance as inconsistency measure
            spatial_vars = probs.view(N, -1).var(dim=1)  # (N,)
            
            # 2. Compare mask scores with actual area
            mask_scores = torch.sigmoid(mask_logits.mean(dim=[1, 2]))  # (N,)
            actual_areas = probs.mean(dim=[1, 2])  # (N,)
            
            # Difference between predicted score and actual area
            area_score_diff = torch.abs(mask_scores - actual_areas)  # (N,)
            
            # Combine inconsistencies
            consistency_measure = spatial_vars + area_score_diff  # (N,)
            
            # Convert to consistency score (higher = more consistent)
            consistency = torch.exp(-consistency_measure)  # (N,) in [0, 1]
            
            return consistency  # (N,)

    def compute_domain_feedback(self, mask_logits):
        """
        Compute domain-specific feedback based on remote sensing priors.
        Only applies suppression, never generates new structures.
        """
        with torch.no_grad():
            probs = torch.sigmoid(mask_logits)  # (N, H, W)
            binary_masks = (probs > 0.5).float()  # (N, H, W)
            
            N, H, W = binary_masks.shape
            total_pixels = H * W
            
            # Calculate area ratios
            areas = binary_masks.view(N, -1).sum(dim=1).float()  # (N,)
            area_ratios = areas / total_pixels  # (N,)
            
            # Area-based penalties
            area_penalty = torch.ones_like(area_ratios)
            below_min = area_ratios < self.min_area_ratio
            above_max = area_ratios > self.max_area_ratio
            area_penalty[below_min] = torch.clamp(
                area_ratios[below_min] / self.min_area_ratio, 0.1, 1.0
            )
            area_penalty[above_max] = torch.clamp(
                (self.max_area_ratio - (area_ratios[above_max] - self.max_area_ratio)) / self.max_area_ratio,
                0.1, 1.0
            )
            
            # Perimeter-area ratio calculation (approximate using convolutions)
            kernel = torch.tensor([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]], dtype=binary_masks.dtype, device=binary_masks.device)
            
            neighbor_sums = F.conv2d(binary_masks.unsqueeze(1), 
                                     kernel.unsqueeze(0).unsqueeze(0), 
                                     padding=1).squeeze(1)  # (N, H, W)
            
            perimeters = torch.sum(
                F.relu(neighbor_sums - 4 * binary_masks), dim=[1, 2]
            ).float()  # (N,)
            
            perimeter_area_ratios = torch.where(
                areas > 0,
                perimeters / areas,
                torch.zeros_like(areas)
            )
            
            # Shape-based penalty
            shape_penalty = torch.ones_like(perimeter_area_ratios)
            above_thresh = perimeter_area_ratios > self.max_perimeter_area_ratio
            shape_penalty[above_thresh] = torch.clamp(
                self.max_perimeter_area_ratio / (perimeter_area_ratios[above_thresh] + 1e-6),
                0.1, 1.0
            )
            
            # Combine domain penalties
            domain_confidence = 0.6 * area_penalty + 0.4 * shape_penalty
            domain_confidence = torch.clamp(domain_confidence, 0.1, 1.0)
            
            return domain_confidence  # (N,)

    def forward(self, query_tokens, mask_logits, image):
        """
        Apply feedback-guided query calibration.
        
        Args:
            query_tokens: Original query tokens from decoder (N, D)
            mask_logits: Decoder mask logits (N, H, W)
            image: Input image (C, H, W)
            
        Returns:
            calibrated_query_tokens: Adjusted query tokens (N, D)
            feedback_strength: Per-query feedback strength (N,)
        """
        # Compute feedback signals from decoder outputs
        uncertainty = self.compute_uncertainty_feedback(mask_logits)  # (N,)
        consistency = self.compute_consistency_feedback(mask_logits)  # (N,)
        domain_confidence = self.compute_domain_feedback(mask_logits)  # (N,)
        
        # Compute composite feedback signal
        # Only apply feedback if uncertainty is high AND consistency is low
        condition = (uncertainty > self.uncertainty_threshold) & (consistency < self.consistency_threshold)
        
        # Compute raw feedback response
        raw_response = torch.sigmoid(
            -self.alpha_uncertainty * uncertainty 
            + self.beta_consistency * consistency
        ) * domain_confidence  # (N,)
        
        # Apply conditional calibration
        feedback_response = torch.where(
            condition,
            raw_response,
            torch.ones_like(raw_response)  # No adjustment if condition not met
        )
        
        # Apply zero-mean calibration to queries
        feedback_adjusted = feedback_response - feedback_response.mean()  # Zero-mean
        
        # Apply calibration to query tokens
        calibrated_queries = query_tokens * (1 + self.lambda_cal * feedback_adjusted.unsqueeze(-1))
        
        return calibrated_queries, feedback_response


class EnhancedRSFeedbackModule(nn.Module):
    """
    增强遥感反馈模块，专注于提高分割结果的空间一致性和结构合理性，
    同时保持计算效率和内存友好性。
    """
    def __init__(self, 
                 consistency_threshold=0.5,
                 area_range=(0.001, 0.5),
                 max_connected_components=10,
                 max_perimeter_area_ratio=100.0,
                 max_holes=5,
                 multi_scale_threshold=0.6,
                 spatial_coherence_weight=0.1,
                 edge_alignment_weight=0.05):
        super(EnhancedRSFeedbackModule, self).__init__()
        
        self.consistency_threshold = consistency_threshold
        self.min_area, self.max_area = area_range
        self.max_connected_components = max_connected_components
        self.max_perimeter_area_ratio = max_perimeter_area_ratio
        self.max_holes = max_holes
        self.multi_scale_threshold = multi_scale_threshold
        self.spatial_coherence_weight = spatial_coherence_weight
        self.edge_alignment_weight = edge_alignment_weight
        
        print("EnhancedRSFeedbackModule initialized:")
        print("- Only modifies instance scores, not features or logits")
        print("- Does not alter encoder/decoder tokens or attention mechanisms")
        print("- Provides unidirectional feedback without refinement loops")
        print("- Maintains numerical stability with bounded gate values")

    def detect_anomalous_tokens(self, mask_logits, image_features=None):
        """
        检测结构异常的token，使用GPU友好的向量化方法
        """
        with torch.no_grad():
            # 使用sigmoid获得概率
            probs = torch.sigmoid(mask_logits)  # (N, H, W)
            num_instances = probs.shape[0]
            
            if num_instances <= 1:
                return torch.ones(num_instances, device=probs.device, dtype=probs.dtype)
            
            # 向量化计算低维嵌入表示（面积、形状粗糙度等）
            areas = probs.view(num_instances, -1).mean(dim=1)  # (N,)
            stds = probs.view(num_instances, -1).std(dim=1)    # (N,)
            sums = (probs > 0.5).view(num_instances, -1).sum(dim=1).float()  # (N,)
            
            # 组装嵌入向量
            embeddings = torch.stack([areas, stds, sums], dim=1)  # (N, 3)
            embeddings = F.normalize(embeddings, dim=1)
            
            # 计算余弦相似度矩阵
            similarity_matrix = torch.mm(embeddings, embeddings.t())  # (N, N)
            
            # 计算每个实例与其他实例的平均相似度
            avg_similarities = torch.mean(similarity_matrix, dim=1)  # (N,)
            
            # 异常检测：低相似度的实例被认为是异常的
            anomalous_scores = torch.sigmoid(avg_similarities - torch.mean(avg_similarities))  # (N,)
            
            return anomalous_scores

    def analyze_mask_structure(self, binary_mask):
        """
        分析单个掩码结构，使用GPU友好的近似方法
        """
        with torch.no_grad():
            h, w = binary_mask.shape
            
            # GPU友好的近似指标
            # 1. 连通组件数量的近似
            kernel = torch.tensor([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]], dtype=binary_mask.dtype, device=binary_mask.device)
            neighbor_sum = F.conv2d(binary_mask.unsqueeze(0).unsqueeze(0), 
                                   kernel.unsqueeze(0).unsqueeze(0), 
                                   padding=1).squeeze()
            
            # 粗略估计连通组件数量（基于邻域连接性）
            connectivity = (neighbor_sum * binary_mask).sum().item()
            
            # 2. 周长的近似
            perimeter_approx = torch.sum(
                F.relu(neighbor_sum - 4 * binary_mask)  # 减去内部点的连接数
            ).item()
            
            # 3. 面积
            area = binary_mask.sum().item()
            
            # 4. 周长面积比
            if area > 0:
                perimeter_area_ratio = perimeter_approx / area
            else:
                perimeter_area_ratio = 0
                
            return {
                'area': area,
                'perimeter_area_ratio': perimeter_area_ratio,
                'connectivity': connectivity
            }

    def apply_domain_knowledge(self, masks, height, width):
        """
        应用领域知识对掩码进行评估（向量化版本）
        """
        with torch.no_grad():
            # 向量化处理所有掩码
            probs = torch.sigmoid(masks)  # (N, H, W)
            binary_masks = (probs > 0.5).float()  # (N, H, W)
            
            # 计算所有掩码的面积
            areas = binary_masks.view(binary_masks.shape[0], -1).sum(dim=1).float()  # (N,)
            total_pixels = height * width
            area_ratios = areas / total_pixels  # (N,)
            
            # 面积评分
            area_scores = torch.ones_like(area_ratios)
            below_min = area_ratios < self.min_area
            above_max = area_ratios > self.max_area
            area_scores[below_min] = torch.clamp(
                1.0 - ((self.min_area - area_ratios[below_min]) / self.min_area), 0.1, 1.0
            )
            area_scores[above_max] = torch.clamp(
                1.0 - ((area_ratios[above_max] - self.max_area) / (0.5 - self.max_area + 1e-6)), 0.1, 1.0
            )
            
            # 周长近似计算（向量化）
            kernel = torch.tensor([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]], dtype=masks.dtype, device=masks.device)
            
            neighbor_sums = F.conv2d(binary_masks.unsqueeze(1), 
                                     kernel.unsqueeze(0).unsqueeze(0), 
                                     padding=1).squeeze(1)  # (N, H, W)
            
            perimeters = torch.sum(
                F.relu(neighbor_sums - 4 * binary_masks), dim=[1, 2]
            ).float()  # (N,)
            
            # 周长面积比
            perimeter_area_ratios = torch.where(
                areas > 0,
                perimeters / areas,
                torch.zeros_like(areas)
            )
            
            # 形状评分
            shape_scores = torch.ones_like(perimeter_area_ratios)
            above_thresh = perimeter_area_ratios > self.max_perimeter_area_ratio
            shape_scores[above_thresh] = torch.clamp(
                self.max_perimeter_area_ratio / (perimeter_area_ratios[above_thresh] + 1e-6),
                0.1, 1.0
            )
            
            # 综合评分
            overall_scores = 0.5 * area_scores + 0.5 * shape_scores
            overall_scores = torch.clamp(overall_scores, 0.1, 1.0)  # 确保在[0.1, 1.0]范围内
            
            return overall_scores

    def compute_edge_alignment(self, masks, image):
        """
        计算掩码与图像边缘的对齐程度（批量处理版本）
        """
        with torch.no_grad():
            # 使用Sobel算子计算图像梯度
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=image.dtype, device=image.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=image.dtype, device=image.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

            grad_x = F.conv2d(image.unsqueeze(0), sobel_x, padding=1, groups=3)
            grad_y = F.conv2d(image.unsqueeze(0), sobel_y, padding=1, groups=3)
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).mean(dim=1, keepdim=True)

            # 计算所有掩码的边界（向量化）
            masks_sigmoid = torch.sigmoid(masks)  # (N, H, W)
            mask_boundaries = self._compute_batch_boundaries(masks_sigmoid)  # (N, H, W)

            # 计算对齐度（向量化）
            aligned_edges = (mask_boundaries * gradient_magnitude.squeeze(0)).view(mask_boundaries.shape[0], -1).sum(dim=1)  # (N,)
            total_boundaries = mask_boundaries.view(mask_boundaries.shape[0], -1).sum(dim=1)  # (N,)
            
            # 避免除零
            alignment_scores = torch.where(
                total_boundaries > 0,
                aligned_edges / total_boundaries,
                torch.zeros_like(total_boundaries)
            )
            
            # 归一化到[0, 1]区间
            alignment_scores = torch.clamp(alignment_scores * 5, 0, 1)  # 缩放因子可调整
            
            # 确保最小值
            alignment_scores = torch.clamp(alignment_scores, 0.1, 1.0)
                
            return alignment_scores

    def _compute_batch_boundaries(self, masks):
        """
        计算一批掩码的边界（向量化）
        """
        kernel = torch.tensor([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]], dtype=masks.dtype, device=masks.device)
        
        # 批量膨胀
        dilated = F.conv2d(masks.unsqueeze(1), 
                           kernel.unsqueeze(0).unsqueeze(0), 
                           padding=1).squeeze(1)
        # 批量腐蚀
        eroded = 1 - F.conv2d((1 - masks).unsqueeze(1), 
                              kernel.unsqueeze(0).unsqueeze(0), 
                              padding=1).squeeze(1)
        
        boundaries = torch.clamp(dilated - eroded, min=0)
        return boundaries

    def apply_feedback(self, original_scores, mask_logits, image):
        """
        应用反馈机制调整实例得分（完全向量化版本）
        """
        with torch.no_grad():
            # 1. 检测异常token（向量化）
            anomaly_scores = self.detect_anomalous_tokens(mask_logits)
            
            # 2. 应用领域知识（向量化）
            domain_scores = self.apply_domain_knowledge(mask_logits, image.shape[1], image.shape[2])
            
            # 3. 计算边缘对齐度（批量）
            edge_alignment_scores = self.compute_edge_alignment(mask_logits, image)
            
            # 组合所有评分
            combined_scores = (
                original_scores * 0.4 +  # 保持原始置信度的基础
                anomaly_scores * 0.2 +   # 异常检测贡献
                domain_scores * 0.2 +    # 领域知识贡献
                edge_alignment_scores * 0.2  # 边缘对齐贡献
            )
            
            # 确保分数在合理范围内
            combined_scores = torch.clamp(combined_scores, min=0.05, max=1.0)
            
            # 返回相对于原始分数的调整比例
            adjustment_ratios = combined_scores / (original_scores + 1e-8)
            
            # 限制调整幅度以保持稳定性
            adjustment_ratios = torch.clamp(adjustment_ratios, min=0.5, max=1.5)
            
            final_scores = original_scores * adjustment_ratios
            
            return final_scores


class AnomalousTokenPerception(nn.Module):
    """
    异常Token感知模块，检测结构不一致的掩码
    """
    def __init__(self, threshold=0.3):
        super(AnomalousTokenPerception, self).__init__()
        self.threshold = threshold

    def forward(self, mask_logits):
        """
        输入: mask_logits (N, H, W) - N个实例的掩码logits
        输出: anomaly_scores (N,) - 每个实例的异常得分
        """
        return self.detect_anomalous_tokens(mask_logits)

    def detect_anomalous_tokens(self, mask_logits):
        """
        检测异常tokens，使用GPU友好的方法
        """
        with torch.no_grad():
            # 使用sigmoid获得概率
            probs = torch.sigmoid(mask_logits)
            num_instances = probs.shape[0]
            
            if num_instances <= 1:
                return torch.ones(num_instances, device=probs.device, dtype=probs.dtype)
            
            # 计算低维嵌入表示（面积、形状粗糙度等）
            embeddings = []
            for i in range(num_instances):
                m = probs[i]
                embeddings.append(torch.stack([
                    m.mean(),  # area proxy
                    m.std(),   # shape roughness
                    torch.sum(m > 0.5).float(),  # area
                ]))
            
            embeddings = torch.stack(embeddings)  # (N, 3)
            embeddings = F.normalize(embeddings, dim=1)
            
            # 计算余弦相似度矩阵
            similarity_matrix = torch.mm(embeddings, embeddings.t())
            
            # 计算每个实例与其他实例的平均相似度
            avg_similarities = torch.mean(similarity_matrix, dim=1)
            
            # 异常检测：低相似度的实例被认为是异常的
            anomalous_scores = torch.sigmoid(avg_similarities - torch.mean(avg_similarities))
            
            return anomalous_scores


class DomainAwareStructuralPrior(nn.Module):
    """
    领域感知结构先验模块，评估掩码的几何合理性
    """
    def __init__(self, area_range=(0.001, 0.5), max_components=10, max_perimeter_ratio=100.0):
        super(DomainAwareStructuralPrior, self).__init__()
        self.min_area, self.max_area = area_range
        self.max_components = max_components
        self.max_perimeter_ratio = max_perimeter_ratio

    def forward(self, masks):
        """
        输入: masks (N, H, W) - N个实例的掩码
        输出: validity_scores (N,) - 每个掩码的有效性得分
        """
        scores = []
        for mask in masks:
            score = self.evaluate_mask_validity(mask)
            scores.append(score)
        
        return torch.tensor(scores, dtype=masks.dtype, device=masks.device)

    def evaluate_mask_validity(self, mask):
        """
        评估单个掩码的有效性
        """
        with torch.no_grad():
            # 转换为二值掩码
            binary_mask = (torch.sigmoid(mask) > 0.5).float()
            h, w = binary_mask.shape
            
            # GPU友好的近似指标
            area = binary_mask.sum().item()
            total_pixels = h * w
            area_ratio = area / total_pixels
            
            # 面积合理性检查
            if self.min_area <= area_ratio <= self.max_area:
                area_score = 1.0
            else:
                area_score = 0.1  # 不符合面积要求的掩码得分很低
            
            # 连通组件数量的近似
            kernel = torch.tensor([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]], dtype=binary_mask.dtype, device=binary_mask.device)
            neighbor_sum = F.conv2d(binary_mask.unsqueeze(0).unsqueeze(0), 
                                   kernel.unsqueeze(0).unsqueeze(0), 
                                   padding=1).squeeze()
            
            # 粗略估计连通组件数量（基于邻域连接性）
            connectivity = (neighbor_sum * binary_mask).sum().item()
            
            # 连通组件数量合理性检查（近似）
            if connectivity > 0:
                # 假设连通组件数量与连接性成反比
                component_score = min(1.0, self.max_components / (connectivity / area + 1e-6))
            else:
                component_score = 0.1
            
            # 周长的近似
            perimeter_approx = torch.sum(
                F.relu(neighbor_sum - 4 * binary_mask)  # 减去内部点的连接数
            ).item()
            
            # 周长面积比
            if area > 0:
                perimeter_area_ratio = perimeter_approx / area
                if perimeter_area_ratio <= self.max_perimeter_ratio:
                    shape_score = 1.0
                else:
                    shape_score = max(0.1, self.max_perimeter_ratio / (perimeter_area_ratio + 1e-6))
            else:
                shape_score = 0.1
            
            # 综合评分
            validity_score = (area_score * 0.4 + component_score * 0.3 + shape_score * 0.3)
            return max(0.05, validity_score)  # 确保最小分数


class RemoteSensingGenericKnowledge(nn.Module):
    """
    遥感通用知识模块，基于多视角一致性的约束
    """
    def __init__(self, multi_scale_threshold=0.6):
        super(RemoteSensingGenericKnowledge, self).__init__()
        self.multi_scale_threshold = multi_scale_threshold

    def forward(self, masks, image_features=None):
        """
        输入: masks (N, H, W) - N个实例的掩码
        输出: knowledge_scores (N,) - 每个掩码的知识一致性得分
        """
        scores = []
        for mask in masks:
            score = self.assess_consistency(mask)
            scores.append(score)
        
        return torch.tensor(scores, dtype=masks.dtype, device=masks.device)

    def assess_consistency(self, mask):
        """
        评估掩码的一致性
        """
        with torch.no_grad():
            # 使用不同尺度的掩码来评估一致性
            mask_sigmoid = torch.sigmoid(mask)
            
            # 获取原始尺寸
            h, w = mask_sigmoid.shape
            
            # 尺度一致性评估：在不同尺度下掩码的稳定性
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            consistency_measures = []
            
            for scale in scales:
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 0 and new_w > 0:
                    # 缩放掩码
                    scaled_mask = F.interpolate(
                        mask_sigmoid.unsqueeze(0).unsqueeze(0),
                        size=(new_h, new_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    # 再缩放回原始尺寸
                    rescaled_mask = F.interpolate(
                        scaled_mask.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    # 计算一致性（使用IoU作为度量）
                    intersection = (mask_sigmoid * rescaled_mask).sum()
                    union = (mask_sigmoid + rescaled_mask - mask_sigmoid * rescaled_mask).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        consistency_measures.append(iou.item())
            
            if consistency_measures:
                avg_consistency = sum(consistency_measures) / len(consistency_measures)
                consistency_score = max(0.1, min(1.0, avg_consistency / self.multi_scale_threshold))
            else:
                consistency_score = 0.1
                
            return consistency_score


class EfficientDecoderGuidedStructuralCalibration(nn.Module):
    """
    高效的Decoder-Guided Structural Calibration (DGSC) 模块
    利用Decoder的结构不稳定性，生成patch-level反馈场对backbone进行校准
    优化版本，减少了不必要的计算步骤
    """
    def __init__(self, lambda_factor=0.05, downsample_ratio=4):
        super(EfficientDecoderGuidedStructuralCalibration, self).__init__()
        self.lambda_factor = lambda_factor  # 控制校准强度
        self.downsample_ratio = downsample_ratio  # 降低分辨率以加速计算

    def forward(self, backbone_feat, decoder_feat, mask_logits_list):
        """
        对backbone特征进行结构校准
        :param backbone_feat: backbone特征 (B, C, H, W)
        :param decoder_feat: decoder特征 (B, C, H, W)
        :param mask_logits_list: 多个mask logits列表
        :return: 校准后的backbone特征 (B, C, H, W)
        """
        return self.efficient_decoder_guided_backbone_correction(
            backbone_feat, decoder_feat, mask_logits_list
        )

    def efficient_decoder_guided_backbone_correction(self, backbone_feat, decoder_feat, mask_logits_list):
        """
        实现高效的decoder引导的backbone校正算法
        注意：此版本仅用于离线分析，不应在推理阶段使用
        """
        B, C, H, W = backbone_feat.shape

        # 快速降采样以加速计算
        ds_ratio = self.downsample_ratio
        if ds_ratio > 1:
            backbone_feat_ds = F.interpolate(
                backbone_feat, 
                size=(H//ds_ratio, W//ds_ratio), 
                mode="bilinear", 
                align_corners=False
            )
            decoder_feat_ds = F.interpolate(
                decoder_feat, 
                size=(H//ds_ratio, W//ds_ratio), 
                mode="bilinear", 
                align_corners=False
            )
        else:
            backbone_feat_ds = backbone_feat
            decoder_feat_ds = decoder_feat

        # -------- 快速Decoder Reliability Analysis --------
        # 特征方差不确定性 (快速计算)
        feat_std = torch.std(decoder_feat_ds, dim=1, keepdim=True)

        # 如果有mask logits，进行快速不确定性分析
        if len(mask_logits_list) > 0:
            # 快速降采样mask logits以加速处理
            mask_logits_tensor = torch.stack(mask_logits_list)
            if ds_ratio > 1:
                mask_logits_tensor = F.interpolate(
                    mask_logits_tensor.unsqueeze(0), 
                    size=(H//ds_ratio, W//ds_ratio), 
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0)
            
            # 快速计算概率和熵
            probs = torch.sigmoid(mask_logits_tensor)
            entropy = (-probs * torch.log(probs + 1e-6) - (1 - probs) * torch.log(1 - probs + 1e-6)).mean(0, keepdim=True)
            
            # 快速计算prompt变异性
            prompt_var = torch.var(probs, dim=0, keepdim=True)
            
            # 组合不确定性
            uncertainty = 0.4 * feat_std + 0.3 * entropy + 0.3 * prompt_var
        else:
            uncertainty = feat_std

        # 上采样回原始尺寸
        if ds_ratio > 1:
            uncertainty = F.interpolate(
                uncertainty, 
                size=(H, W), 
                mode="bilinear", 
                align_corners=False
            )

        # 使用sigmoid归一化到[0,1]区间
        uncertainty = torch.sigmoid(uncertainty)

        # -------- Structural Feedback Field --------
        # 生成稳定性门控
        gate = torch.exp(-uncertainty)  # (B, 1, H, W)，值在(0, 1]之间

        # -------- 快速Backbone Token Stabilization --------
        # 快速计算局部均值作为结构参考
        local_mean = F.avg_pool2d(backbone_feat, kernel_size=3, stride=1, padding=1)
        
        # 快速计算结构偏移（零均值）
        delta = local_mean - backbone_feat
        # 确保零均值
        delta = delta - delta.mean(dim=(-2, -1), keepdim=True)

        # 应用decoder引导的修正
        # 只有在decoder认为不稳定的区域才进行修正
        corrected_feat = backbone_feat + self.lambda_factor * (1 - gate) * delta
        
        return corrected_feat


class DecoderGuidedStructuralCalibration(nn.Module):
    """
    Decoder-Guided Structural Calibration (DGSC) 模块
    利用Decoder的结构不稳定性，生成patch-level反馈场对backbone进行校准
    为了兼容性保留原始版本，但推荐使用Efficient版本
    """
    def __init__(self, lambda_factor=0.05):
        super(DecoderGuidedStructuralCalibration, self).__init__()
        self.lambda_factor = lambda_factor  # 控制校准强度

    def forward(self, backbone_feat, decoder_feat, mask_logits_list):
        """
        对backbone特征进行结构校准
        :param backbone_feat: backbone特征 (B, C, H, W)
        :param decoder_feat: decoder特征 (B, C, H, W)
        :param mask_logits_list: 多个mask logits列表
        :return: 校准后的backbone特征 (B, C, H, W)
        """
        return self.decoder_guided_backbone_correction(
            backbone_feat, decoder_feat, mask_logits_list
        )

    def decoder_guided_backbone_correction(self, backbone_feat, decoder_feat, mask_logits_list):
        """
        实现核心的decoder引导的backbone校正算法
        """
        B, C, H, W = backbone_feat.shape

        # -------- Decoder Reliability Analysis --------
        # 特征方差不确定性
        feat_std = torch.std(decoder_feat, dim=1, keepdim=True)

        # 如果有mask logits，计算熵和prompt变异性
        if len(mask_logits_list) > 0:
            # 计算概率
            probs = torch.stack([torch.sigmoid(m) for m in mask_logits_list])
            
            # 计算熵
            entropy = (-probs * torch.log(probs + 1e-6) - (1 - probs) * torch.log(1 - probs + 1e-6)).mean(0, keepdim=True)
            
            # 计算prompt变异性
            prompt_var = torch.var(probs, dim=0, keepdim=True)
            
            # 组合不确定性
            uncertainty = 0.4 * feat_std + 0.3 * entropy + 0.3 * prompt_var
        else:
            uncertainty = feat_std

        # 插值到backbone特征尺寸
        uncertainty = F.interpolate(
            uncertainty, size=backbone_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        
        # 使用sigmoid归一化到[0,1]区间
        uncertainty = torch.sigmoid(uncertainty)

        # -------- Structural Feedback Field --------
        # 生成稳定性门控
        gate = torch.exp(-uncertainty)  # (B, 1, H, W)，值在(0, 1]之间

        # -------- Backbone Token Stabilization --------
        # 计算局部均值作为结构参考
        local_mean = F.avg_pool2d(backbone_feat, kernel_size=3, stride=1, padding=1)
        
        # 计算结构偏移（零均值）
        delta = local_mean - backbone_feat
        # 确保零均值
        delta = delta - delta.mean(dim=(-2, -1), keepdim=True)

        # 应用decoder引导的修正
        # 只有在decoder认为不稳定的区域才进行修正
        corrected_feat = backbone_feat + self.lambda_factor * (1 - gate) * delta
        
        return corrected_feat


def apply_dgsc_calibration(backbone_feat, decoder_feat, mask_logits_list, lambda_factor=0.05):
    """
    应用DGSC校准的便捷函数
    """
    dgsc = EfficientDecoderGuidedStructuralCalibration(lambda_factor=lambda_factor)
    return dgsc(backbone_feat, decoder_feat, mask_logits_list)