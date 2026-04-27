#!/usr/bin/env python3
"""
Intelligent Segmentation Coordinator for SAM3 with Qwen3 Agent
This module coordinates between the Qwen3 agent and SAM3 segmenter
to optimize segmentation performance through iterative refinement
and adaptive prompting strategies.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2
from scipy.ndimage import label
from sklearn.cluster import KMeans

from .qwen3_agent import Qwen3Agent
from ..model.sam1_task_predictor import SAM3InteractiveImagePredictor


class IntelligentSegmentationCoordinator:
    """
    Coordinates between Qwen3 agent and SAM3 segmenter for optimal segmentation.
    Implements iterative refinement and adaptive prompting strategies.
    """
    
    def __init__(self, 
                 qwen3_agent: Qwen3Agent, 
                 sam_predictor: SAM3InteractiveImagePredictor,
                 confidence_threshold: float = 0.5,
                 max_iterations: int = 3):
        """
        Initialize the coordinator
        
        Args:
            qwen3_agent: Qwen3 agent for generating prompts
            sam_predictor: SAM predictor for segmentation
            confidence_threshold: Minimum confidence for accepting predictions
            max_iterations: Maximum iterations for refinement
        """
        self.qwen3_agent = qwen3_agent
        self.sam_predictor = sam_predictor
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.device = sam_predictor.device
        
    def coordinate_segmentation(self, 
                               image: np.ndarray, 
                               class_names: List[str],
                               image_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Coordinate segmentation for all classes
        
        Args:
            image: Input image array (H, W, C)
            class_names: List of class names to segment
            image_path: Path to image file (if available for Qwen3 agent)
            
        Returns:
            Dictionary mapping class names to segmentation masks
        """
        # Prepare image for SAM
        self.sam_predictor.set_image(image)
        
        # Get descriptions from Qwen3 agent
        if image_path:
            descriptions = self.qwen3_agent.generate_count_and_descriptions(
                image_path=image_path, 
                class_names=class_names
            )
        else:
            # If no image path, generate basic descriptions
            descriptions = {cn: {'descriptions': [cn]} for cn in class_names}
        
        results = {}
        
        for class_name in class_names:
            # Get the best prompts for this class
            class_descriptions = descriptions[class_name]['descriptions']
            
            # Perform segmentation with multiple prompts
            mask = self._refine_segmentation(
                image=image,
                class_name=class_name,
                prompts=class_descriptions,
                class_idx=class_names.index(class_name)
            )
            
            results[class_name] = mask
        
        return results
    
    def _refine_segmentation(self, 
                             image: np.ndarray,
                             class_name: str,
                             prompts: List[str],
                             class_idx: int) -> np.ndarray:
        """
        Refine segmentation for a specific class using multiple prompts and iterations
        """
        best_mask = None
        best_score = -1
        
        # Try different prompts for the class
        for prompt in prompts:
            # Generate mask using SAM with the prompt
            mask = self._segment_with_prompt(image, prompt)
            
            # Evaluate the quality of the mask
            score = self._evaluate_mask_quality(mask, image, prompt, class_idx)
            
            if score > best_score:
                best_score = score
                best_mask = mask.copy()
        
        # If we have a low-scoring mask, attempt iterative refinement
        if best_score < self.confidence_threshold and best_mask is not None:
            best_mask = self._iterative_refinement(
                image=image,
                initial_mask=best_mask,
                class_name=class_name,
                prompts=prompts
            )
        
        return best_mask if best_mask is not None else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def _segment_with_prompt(self, image: np.ndarray, prompt: str) -> np.ndarray:
        """
        Segment the image using SAM with a text prompt
        """
        # In a real implementation, this would convert the text prompt to
        # visual features that SAM can work with, possibly using CLIP
        # For now, we'll simulate this with a simple approach
        
        # Get image embedding
        image_embedding = self.sam_predictor.get_image_embedding()
        
        # This is a simplified approach - in reality, we'd use RemoteCLIP or similar
        # to convert text prompt to visual features
        h, w = image.shape[:2]
        
        # Create a default point for segmentation (center of image)
        input_points = np.array([[w // 2, h // 2]])
        input_labels = np.array([1])
        
        # Predict mask
        masks, _, _ = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        
        # Return the largest mask if multiple are returned
        if len(masks) > 0:
            # Find the largest connected component
            largest_mask = self._get_largest_connected_component(masks[0])
            return largest_mask.astype(np.uint8)
        else:
            return np.zeros((h, w), dtype=np.uint8)
    
    def _evaluate_mask_quality(self, 
                              mask: np.ndarray, 
                              image: np.ndarray, 
                              prompt: str, 
                              class_idx: int) -> float:
        """
        Evaluate the quality of a segmentation mask
        """
        if mask.sum() == 0:  # Empty mask
            return 0.0
        
        # Calculate basic statistics
        mask_area = mask.sum() / mask.size
        if mask_area > 0.9:  # Too large - likely incorrect
            return 0.1
        
        # Check color consistency within the mask
        masked_image = image[mask.astype(bool)]
        if len(masked_image) == 0:
            return 0.0
        
        # Compute color variance in the masked region
        color_variance = np.var(masked_image, axis=0).mean()
        
        # Lower color variance indicates more consistent region
        # Normalize to 0-1 range (assuming RGB in 0-255 range)
        normalized_variance = min(color_variance / (255.0 ** 2), 1.0)
        
        # Invert so higher values mean better consistency
        color_consistency_score = 1.0 - normalized_variance
        
        # Consider mask size (prefer medium-sized regions)
        size_score = 1.0 - abs(0.3 - mask_area)  # Optimal around 30% of image
        
        # Combine scores
        quality_score = 0.5 * color_consistency_score + 0.5 * size_score
        
        return max(quality_score, 0.0)  # Ensure non-negative
    
    def _get_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Get the largest connected component from a binary mask
        """
        labeled_mask, num_labels = label(mask)
        
        if num_labels <= 1:
            return mask  # Either all zeros or all ones
        
        # Find the largest component (excluding background label 0)
        largest_label = 1
        largest_size = 0
        
        for label_num in range(1, num_labels + 1):
            component_size = np.sum(labeled_mask == label_num)
            if component_size > largest_size:
                largest_size = component_size
                largest_label = label_num
        
        # Create mask with only the largest component
        largest_component = (labeled_mask == largest_label).astype(np.uint8)
        
        return largest_component
    
    def _iterative_refinement(self, 
                             image: np.ndarray,
                             initial_mask: np.ndarray,
                             class_name: str,
                             prompts: List[str]) -> np.ndarray:
        """
        Attempt to refine a low-quality mask through iterative processing
        """
        current_mask = initial_mask.copy()
        refined_mask = None
        
        for iteration in range(self.max_iterations):
            # Find border regions of the current mask
            border_mask = self._find_mask_border(current_mask, image.shape[:2])
            
            # If the current mask is too small, try expanding
            if current_mask.sum() < 100:  # Very small mask
                expanded_mask = self._expand_mask_region(image, current_mask, class_name, prompts)
                if expanded_mask.sum() > current_mask.sum():
                    current_mask = expanded_mask
                    refined_mask = current_mask
                    continue
            
            # Try to improve the mask using border information
            improved_mask = self._improve_mask_using_border(image, current_mask, border_mask, class_name, prompts)
            
            if improved_mask.sum() > 0:
                # Evaluate improvement
                old_score = self._evaluate_mask_quality(current_mask, image, prompts[0], 0)
                new_score = self._evaluate_mask_quality(improved_mask, image, prompts[0], 0)
                
                if new_score > old_score:
                    current_mask = improved_mask
                    refined_mask = current_mask
                else:
                    break  # No improvement, stop refining
            else:
                break  # No improvement possible
        
        return refined_mask if refined_mask is not None else initial_mask
    
    def _find_mask_border(self, mask: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Find the border pixels of a binary mask
        """
        # Expand mask to ensure proper convolution
        padded_mask = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        
        # Define kernel for border detection
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
        
        # Convolve to find borders
        border = cv2.filter2D(padded_mask.astype(np.float32), -1, kernel)
        border = border[1:-1, 1:-1]  # Remove padding
        
        # Threshold to create binary border mask
        border_mask = (border > 0).astype(np.uint8)
        
        return border_mask
    
    def _expand_mask_region(self, 
                           image: np.ndarray, 
                           mask: np.ndarray, 
                           class_name: str, 
                           prompts: List[str]) -> np.ndarray:
        """
        Expand a small mask region based on visual similarity
        """
        if mask.sum() == 0:
            return mask
        
        # Get representative colors from the mask region
        masked_pixels = image[mask.astype(bool)]
        if len(masked_pixels) == 0:
            return mask
        
        # Get average color in the mask region
        avg_color = np.mean(masked_pixels, axis=0)
        
        # Create expanded mask based on color similarity
        color_diff = np.linalg.norm(image.astype(np.float32) - avg_color.reshape(1, 1, 3), axis=2)
        
        # Adaptive threshold based on class
        if class_name == 'clutter':
            # For clutter, use a higher threshold to capture diverse content
            threshold = 80.0
        else:
            # For other classes, use a lower threshold for precision
            threshold = 50.0
        
        expanded_mask = (color_diff < threshold).astype(np.uint8)
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, kernel)
        expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, kernel)
        
        return expanded_mask
    
    def _improve_mask_using_border(self, 
                                  image: np.ndarray, 
                                  mask: np.ndarray, 
                                  border_mask: np.ndarray, 
                                  class_name: str, 
                                  prompts: List[str]) -> np.ndarray:
        """
        Improve a mask using information from its border
        """
        if border_mask.sum() == 0:
            return mask
        
        # Get border pixels
        border_pixels = image[border_mask.astype(bool)]
        if len(border_pixels) == 0:
            return mask
        
        # Perform clustering on border pixels to identify different regions
        n_clusters = min(3, len(border_pixels))
        if n_clusters < 2:
            return mask
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(border_pixels)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Determine which cluster center is most similar to the main mask
        main_mask_pixels = image[mask.astype(bool)]
        if len(main_mask_pixels) == 0:
            return mask
        
        main_avg_color = np.mean(main_mask_pixels, axis=0)
        
        # Find the closest cluster to the main mask color
        distances_to_main = np.linalg.norm(cluster_centers - main_avg_color, axis=1)
        main_cluster_idx = np.argmin(distances_to_main)
        
        # Expand mask to include pixels similar to the main cluster
        main_cluster_center = cluster_centers[main_cluster_idx]
        
        # Calculate distances from each pixel to the main cluster center
        color_diff = np.linalg.norm(image.astype(np.float32) - main_cluster_center.reshape(1, 1, 3), axis=2)
        
        # Create a new mask by including pixels close to the main cluster
        if class_name == 'clutter':
            # For clutter, be more permissive since it's a diverse class
            threshold = 60.0
        else:
            # For other classes, be more restrictive
            threshold = 40.0
        
        new_mask = (color_diff < threshold).astype(np.uint8)
        
        # Combine with original mask
        combined_mask = np.logical_or(mask, new_mask).astype(np.uint8)
        
        # Apply morphological operations to smooth
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask