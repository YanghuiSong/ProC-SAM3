import os
import torch
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
import torchvision.transforms as T
from ..model_builder import build_sam3_image_model
from ..model.sam3_image_processor import Sam3Processor


def move_to_device(obj, device):
    """Move tensor or nested structure to specified device"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj


class SAM3Adapter:
    """
    Adapter for SAM3 model to work with our system, supporting multi-GPU deployment
    """
    
    def __init__(self, 
                 bpe_path: str = None,
                 checkpoint_path: str = None,
                 device: str = "cuda:0",
                 confidence_threshold: float = 0.5,
                 max_memory_allocated: int = None,
                 use_fp16: bool = False):  # 默认改为False，使用FP32
        """
        Initialize SAM3 adapter with GPU allocation support and type consistency
        
        Args:
            bpe_path: Path to BPE vocabulary (currently unused)
            checkpoint_path: Path to SAM3 checkpoint
            device: Device to run the model on, e.g., "cuda:0", "cuda:1"
            confidence_threshold: Threshold for mask confidence
            max_memory_allocated: Maximum GPU memory to allocate in MB (optional)
            use_fp16: Whether to use fp16 to reduce memory usage
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16
        
        # Extract GPU index from device string
        if ':' in device:
            gpu_idx = int(device.split(':')[1])
        else:
            gpu_idx = 0
            
        print(f"Loading SAM3 model onto device: {device}")
        print(f"Using FP16: {use_fp16}")
        
        # Set memory limit if specified
        if max_memory_allocated is not None and device.startswith('cuda'):
            total_memory = torch.cuda.get_device_properties(gpu_idx).total_memory
            fraction = (max_memory_allocated * 1024**2) / total_memory
            if fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(fraction, gpu_idx)
                print(f"GPU memory limited to {max_memory_allocated} MB ({fraction:.1%} of total)")
            else:
                print(f"Requested memory exceeds GPU capacity, using full memory")
        
        # Load SAM3 model directly on target device to avoid GPU memory issues
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            device=self.device,  # Changed from "cpu" to self.device
            eval_mode=True
        )
        
        # Apply precision setting (FP16 or FP32)
        if self.use_fp16 and "cuda" in self.device:
            self.model = self.model.half()
            self.dtype = torch.float16
        else:
            self.model = self.model.float()
            self.dtype = torch.float32
            
        # Model is already on target device, no need to move again
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded with dtype: {self.dtype} on device: {self.device}")
        
        # Create processor instance - ensure it's also on the correct device
        self.processor = Sam3Processor(
            model=self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold
        )
        
        print(f"SAM3 model loaded successfully on {device}")

    def set_image(self, image) -> Dict:
        """
        Set image for segmentation
        
        Args:
            image: Input can be image path (str), PIL Image, or numpy array
            
        Returns:
            State dictionary containing image information
        """
        if isinstance(image, str):
            # Image path
            pil_image = Image.open(image).convert("RGB")
            image_path = image
        elif isinstance(image, Image.Image):
            # PIL Image
            pil_image = image.convert("RGB")
            image_path = "pil_image_input"
        elif isinstance(image, np.ndarray):
            # NumPy array
            if image.ndim == 3:
                pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
            else:
                raise ValueError(f"Unsupported numpy array shape: {image.shape}")
            image_path = "numpy_array_input"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        # Convert to numpy array for processor
        np_image = np.array(pil_image)
        original_size = pil_image.size  # (width, height)
        
        # Create state dictionary using the processor
        state = self.processor.set_image(np_image)
        
        # Add original size information to state
        state["original_size"] = original_size
        state["original_width"] = original_size[0]
        state["original_height"] = original_size[1]
        state["image_path"] = image_path
        
        return state

    def semantic_filter(self, prompts: List[str], class_names: List[str], remoteclip_model=None) -> List[str]:
        """
        Filter prompts using RemoteCLIP similarity to class names
        
        Args:
            prompts: List of prompts to filter
            class_names: Target class names for alignment
            remoteclip_model: RemoteCLIP model for similarity computation
            
        Returns:
            Filtered list of prompts aligned to class names
        """
        if remoteclip_model is None:
            # If no model provided, return original prompts
            return prompts
        
        try:
            # Encode class names
            class_embeddings = remoteclip_model.encode_dataset_categories(class_names)
            
            # Encode prompts
            prompt_embeddings = remoteclip_model.encode_qwen_prompts(prompts)
            
            # Compute similarity matrix
            similarity_matrix = prompt_embeddings @ class_embeddings.T
            
            # Get top-k similar prompts per class (keeping only high-confidence alignments)
            top_k = min(3, len(prompts))
            top_similarities, top_indices = torch.topk(similarity_matrix, k=top_k, dim=0)
            
            # Flatten indices and get unique prompts
            flat_indices = torch.unique(top_indices.flatten())
            aligned_prompts = [prompts[i] for i in flat_indices.tolist() if i < len(prompts)]
            
            # Only return prompts with high similarity (> 0.3 threshold)
            filtered_prompts = []
            for i, prompt in enumerate(aligned_prompts):
                max_sim = torch.max(similarity_matrix[i])
                if max_sim > 0.3:  # Similarity threshold
                    filtered_prompts.append(prompt)
            
            return filtered_prompts if filtered_prompts else prompts[:1]  # At least return one prompt
            
        except Exception as e:
            print(f"Error in semantic filtering: {e}")
            # On error, return original prompts
            return prompts

    def segment_with_prompts_and_alignment(self, 
                                          image: Image.Image, 
                                          class_names: List[str],
                                          qwen_prompts: Dict[str, List[str]],
                                          remoteclip_model=None) -> Dict:
        """
        Segment image using RemoteCLIP-aligned prompts
        
        Args:
            image: Input PIL Image
            class_names: List of target class names
            qwen_prompts: Dictionary mapping class names to their prompts {class_name: [prompts]}
            remoteclip_model: RemoteCLIP model for alignment
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                np_image = np.array(image)
            else:
                np_image = image
                
            # Initialize state with image
            state = self.processor.set_image(np_image)
            
            # Store results for each class
            class_results = {}
            
            for class_name in class_names:
                if class_name not in qwen_prompts:
                    continue
                    
                # Get prompts for this class
                class_prompts = qwen_prompts[class_name]
                
                # Apply semantic filtering using RemoteCLIP
                if remoteclip_model:
                    aligned_prompts = self.semantic_filter(class_prompts, [class_name], remoteclip_model)
                else:
                    # Just use the original prompts if no model provided
                    aligned_prompts = class_prompts[:5]  # Limit to 5 prompts
                
                # Process each aligned prompt
                all_masks = []
                all_scores = []
                all_logits = []
                all_presence_scores = []
                
                # Clean prompts to remove special tokens
                import re
                cleaned_prompts = []
                for prompt in aligned_prompts:
                    # Remove common special tokens that might cause issues
                    cleaned_prompt = re.sub(r'</s>|<s>|<extra_id_\d+>|<pad>|<bos>|<eos>', '', prompt)
                    # Remove other potential special tokens
                    cleaned_prompt = re.sub(r'<\|.*?\|>', '', cleaned_prompt)
                    # Remove newlines and other control characters
                    cleaned_prompt = re.sub(r'[\r\n\t]', ' ', cleaned_prompt)
                    # Remove multiple consecutive spaces
                    cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt)
                    # Strip leading/trailing whitespace
                    cleaned_prompt = cleaned_prompt.strip()
                    
                    if cleaned_prompt:  # Only add if not empty after cleaning
                        cleaned_prompts.append(cleaned_prompt)
                
                # Process each cleaned prompt
                for prompt in cleaned_prompts:
                    try:
                        # Reset previous prompts to ensure clean state
                        self.processor.reset_all_prompts(state)
                        
                        # Process the prompt
                        state = self.processor.set_text_prompt(prompt=prompt, state=state)
                        
                        # Extract results and ensure they match model's dtype and device
                        masks_logits = state.get("masks_logits", torch.empty(0, 0, 0, device=self.device, dtype=self.dtype))
                        scores = state.get("scores", torch.empty(0, device=self.device, dtype=self.dtype))
                        logits = state.get("masks_logits", torch.empty(0, 0, 0, device=self.device, dtype=self.dtype))
                        presence_score = state.get("presence_score", torch.tensor(0.0, device=self.device, dtype=self.dtype))
                        
                        # Ensure masks have correct dtype and device
                        if masks_logits.numel() > 0:
                            # Ensure masks_logits is on the correct device before processing
                            masks_logits = masks_logits.to(device=self.device, dtype=self.dtype)
                            masks = (masks_logits.sigmoid() > 0.5).to(dtype=self.dtype)
                        else:
                            masks = torch.empty(0, 0, 0, device=self.device, dtype=self.dtype)
                        
                        # Only keep results with reasonable presence score
                        if presence_score >= 0.1 and masks.numel() > 0:
                            # Squeeze dimension if needed
                            if masks.dim() == 4 and masks.size(1) == 1:
                                masks = masks.squeeze(1)
                                logits = logits.squeeze(1)
                                
                            all_masks.append(masks)
                            all_scores.append(scores)
                            all_logits.append(logits)
                            all_presence_scores.append(presence_score)
                            
                    except Exception as e:
                        print(f"Error processing prompt '{prompt}': {e}")
                        continue
                
                # Combine results for this class
                if len(all_masks) > 0:
                    # Concatenate all results
                    if len(all_masks) > 1:
                        # Ensure all tensors are on the correct device before concatenating
                        all_masks = [mask.to(self.device) for mask in all_masks]
                        all_scores = [score.to(self.device) for score in all_scores]
                        all_logits = [logit.to(self.device) for logit in all_logits]
                        
                        combined_masks = torch.cat(all_masks, dim=0)
                        combined_scores = torch.cat(all_scores, dim=0)
                        combined_logits = torch.cat(all_logits, dim=0)
                        avg_presence_score = torch.mean(torch.stack(all_presence_scores))
                    else:
                        combined_masks = all_masks[0]
                        combined_scores = all_scores[0]
                        combined_logits = all_logits[0]
                        avg_presence_score = all_presence_scores[0]
                        
                    # Apply confidence threshold
                    if combined_scores.numel() > 0:
                        valid_mask = combined_scores >= self.confidence_threshold
                        if valid_mask.any():
                            combined_masks = combined_masks[valid_mask]
                            combined_scores = combined_scores[valid_mask]
                            combined_logits = combined_logits[valid_mask]
                    
                    class_results[class_name] = {
                        "masks": combined_masks.cpu(),
                        "scores": combined_scores.cpu(),
                        "logits": combined_logits.cpu(),
                        "presence_score": avg_presence_score.cpu().item() if 'avg_presence_score' in locals() else None,
                        "prompts_used": cleaned_prompts[:len(all_masks)]
                    }
                else:
                    # No valid results for this class
                    class_results[class_name] = {
                        "masks": torch.zeros((0, np_image.shape[0], np_image.shape[1]), dtype=self.dtype),
                        "scores": torch.zeros(0, dtype=self.dtype),
                        "logits": torch.zeros((0, np_image.shape[0], np_image.shape[1]), dtype=self.dtype),
                        "presence_score": None,
                        "prompts_used": []
                    }
            
            # Return results for all classes
            return {
                "class_results": class_results,
                "image_shape": np_image.shape
            }
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
            
            # Clear GPU cache on error
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
            return {
                "class_results": {},
                "image_shape": (0, 0, 0)
            }

    def _move_to_device(self, obj):
        """Move tensor or nested structure to this adapter's device"""
        if torch.is_tensor(obj):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        else:
            return obj

    def segment_with_prompts(self, 
                           image: Image.Image, 
                           prompts: List[str],
                           use_multi_prompt_fusion: bool = True) -> Dict:
        """
        Segment image using multiple text prompts with optional fusion
        
        Args:
            image: Input PIL Image
            prompts: List of text prompts for segmentation
            use_multi_prompt_fusion: Whether to fuse results from multiple prompts
                
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                np_image = np.array(image)
            else:
                np_image = image
                
            # Initialize state with image
            state = self.processor.set_image(np_image)
            
            # Check if state is valid
            if state is None:
                print("Error: Failed to initialize state with image")
                return {
                    "masks": torch.empty(0, 0, 0),
                    "scores": torch.empty(0),
                    "logits": torch.empty(0, 0, 0),
                    "presence_score": None,
                    "prompts_used": []
                }
            
            # Store results from all prompts
            all_masks = []
            all_scores = []
            all_logits = []
            all_presence_scores = []
            
            # Thoroughly clean prompts to remove special tokens that might cause device issues
            import re
            cleaned_prompts = []
            for prompt in prompts:
                # Remove common special tokens that might cause issues
                cleaned_prompt = re.sub(r'</s>|<s>|<extra_id_\d+>|<pad>|<bos>|<eos>', '', prompt)
                # Remove other potential special tokens
                cleaned_prompt = re.sub(r'<\|.*?\|>', '', cleaned_prompt)
                # Remove newlines and other control characters
                cleaned_prompt = re.sub(r'[\r\n\t]', ' ', cleaned_prompt)
                # Remove multiple consecutive spaces
                cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt)
                # Strip leading/trailing whitespace
                cleaned_prompt = cleaned_prompt.strip()
                
                if cleaned_prompt:  # Only add if not empty after cleaning
                    cleaned_prompts.append(cleaned_prompt)
        
            # Process each cleaned prompt
            for prompt in cleaned_prompts:
                try:
                    # Instead of resetting all prompts (which may cause state to be None),
                    # we'll work with a fresh state for each prompt
                    fresh_state = self.processor.set_image(np_image)
                    
                    # Check if fresh_state is valid
                    if fresh_state is None:
                        print(f"Error: Could not create fresh state for prompt '{prompt}'")
                        continue
                    
                    # Ensure all components in the fresh state are on the correct device
                    for key, value in fresh_state.items():
                        if torch.is_tensor(value):
                            fresh_state[key] = value.to(self.device)
                    
                    # Ensure the model itself is on the correct device (double-check)
                    self.model = self.model.to(self.device)
                    
                    # Ensure the processor model is also on the correct device
                    self.processor.model = self.processor.model.to(self.device)
                    
                    # Ensure ALL model parameters and buffers are on the correct device
                    for name, param in self.model.named_parameters():
                        if param.device != self.device:
                            param.data = param.data.to(self.device)
                    for name, buffer in self.model.named_buffers():
                        if buffer.device != self.device:
                            buffer.data = buffer.data.to(self.device)
                    
                    # Additionally, ensure the processor's internal states are on the correct device
                    if hasattr(self.processor, 'model') and self.processor.model is not None:
                        for name, param in self.processor.model.named_parameters():
                            if param.device != self.device:
                                param.data = param.data.to(self.device)
                        for name, buffer in self.processor.model.named_buffers():
                            if buffer.device != self.device:
                                buffer.data = buffer.data.to(self.device)
                    
                    # Force synchronize CUDA devices
                    if self.device.startswith('cuda'):
                        torch.cuda.synchronize(self.device)
                    
                    # Temporarily set the default tensor creation device to our target device
                    prev_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else torch.device('cpu')
                    torch.cuda.set_device(self.device)
                    
                    try:
                        # Use torch.cuda.device context manager to ensure operations happen on the correct device
                        with torch.cuda.device(self.device):
                            # Call set_text_prompt and check for None return
                            result_state = self.processor.set_text_prompt(prompt=prompt, state=fresh_state)
                    finally:
                        # Restore the previous default device
                        if hasattr(torch, 'get_default_device'):
                            torch.set_default_device(prev_device)
                    
                    if result_state is None:
                        print(f"Error: State became None after processing prompt '{prompt}'")
                        continue
                    
                    # Ensure result state tensors are on correct device
                    result_state = move_to_device(result_state, self.device)
                    
                    # Double-check that critical tensors are on the right device and dtype
                    if "masks_logits" in result_state:
                        result_state["masks_logits"] = result_state["masks_logits"].to(device=self.device, dtype=self.dtype)
                    if "scores" in result_state:
                        result_state["scores"] = result_state["scores"].to(device=self.device, dtype=self.dtype)
                    if "presence_score" in result_state:
                        result_state["presence_score"] = result_state["presence_score"].to(device=self.device, dtype=self.dtype)
                    
                    # Extract results and ensure they match model's dtype and device
                    # Ensure all extracted values are on the correct device
                    masks_logits = result_state.get("masks_logits", torch.empty(0, 0, 0, device=self.device, dtype=self.dtype))
                    scores = result_state.get("scores", torch.empty(0, device=self.device, dtype=self.dtype))
                    logits = result_state.get("masks_logits", torch.empty(0, 0, 0, device=self.device, dtype=self.dtype))
                    presence_score = result_state.get("presence_score", torch.tensor(0.0, device=self.device, dtype=self.dtype))
                    
                    # Ensure masks have correct dtype and device
                    if masks_logits.numel() > 0:
                        # Ensure masks_logits is on the correct device before processing
                        masks_logits = masks_logits.to(device=self.device, dtype=self.dtype)
                        masks = (masks_logits.sigmoid() > 0.5).to(dtype=self.dtype)
                    else:
                        masks = torch.empty(0, 0, 0, device=self.device, dtype=self.dtype)
                    
                    # Only keep results with reasonable presence score
                    if presence_score >= 0.1 and masks.numel() > 0:
                        # Squeeze dimension if needed
                        if masks.dim() == 4 and masks.size(1) == 1:
                            masks = masks.squeeze(1)
                            logits = logits.squeeze(1)
                            
                        all_masks.append(masks)
                        all_scores.append(scores)
                        all_logits.append(logits)
                        all_presence_scores.append(presence_score)
                        
                except Exception as e:
                    print(f"Error processing prompt '{prompt}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
            # Ensure all collected results are on the correct device before fusion
            all_masks = [mask.to(self.device) for mask in all_masks]
            all_scores = [score.to(self.device) for score in all_scores]
            all_logits = [logit.to(self.device) for logit in all_logits]
            
            # Fuse results from multiple prompts if requested and available
            if use_multi_prompt_fusion and len(all_masks) > 0:
                # Concatenate all results
                if len(all_masks) > 1:
                    # Ensure all tensors are on the correct device before concatenating
                    combined_masks = torch.cat(all_masks, dim=0)
                    combined_scores = torch.cat(all_scores, dim=0)
                    combined_logits = torch.cat(all_logits, dim=0)
                    avg_presence_score = torch.mean(torch.stack(all_presence_scores))
                else:
                    combined_masks = all_masks[0]
                    combined_scores = all_scores[0]
                    combined_logits = all_logits[0]
                    avg_presence_score = all_presence_scores[0]
                    
                # Apply confidence threshold
                if combined_scores.numel() > 0:
                    valid_mask = combined_scores >= self.confidence_threshold
                    if valid_mask.any():
                        combined_masks = combined_masks[valid_mask]
                        combined_scores = combined_scores[valid_mask]
                        combined_logits = combined_logits[valid_mask]
            
                result = {
                    "masks": combined_masks.cpu(),
                    "scores": combined_scores.cpu(),
                    "logits": combined_logits.cpu(),
                    "presence_score": avg_presence_score.cpu().item() if 'avg_presence_score' in locals() else None,
                    "prompts_used": cleaned_prompts[:len(all_masks)]
                }
            elif len(all_masks) > 0:
                # Ensure tensors are on correct device before returning
                masks = all_masks[0].to(self.device)
                scores = all_scores[0].to(self.device)
                logits = all_logits[0].to(self.device)
                
                # Return first valid result without fusion
                result = {
                    "masks": masks.cpu(),
                    "scores": scores.cpu(),
                    "logits": logits.cpu(),
                    "presence_score": all_presence_scores[0].cpu().item(),
                    "prompts_used": [cleaned_prompts[0]]
                }
            else:
                # No valid results
                result = {
                    "masks": torch.zeros((0, np_image.shape[0], np_image.shape[1]), dtype=self.dtype),
                    "scores": torch.zeros(0, dtype=self.dtype),
                    "logits": torch.zeros((0, np_image.shape[0], np_image.shape[1]), dtype=self.dtype),
                    "presence_score": None,
                    "prompts_used": []
                }
                
            # Clear GPU cache to free memory
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
            return result
            
        except Exception as e:
            print(f"Error during segment_with_prompts: {e}")
            import traceback
            traceback.print_exc()
            
            # Clear GPU cache on error
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
            return {
                "masks": torch.empty(0, 0, 0),
                "scores": torch.empty(0),
                "logits": torch.empty(0, 0, 0),
                "presence_score": None,
                "prompts_used": []
            }

    def segment_with_box(self, image: Image.Image, box: List[float]) -> Dict:
        """
        Segment image using bounding box prompt
        
        Args:
            image: Input PIL Image
            box: Bounding box coordinates [x1, y1, x2, y2]
                
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                np_image = np.array(image)
            else:
                np_image = image
                
            height, width = np_image.shape[:2]
            
            # Normalize box coordinates to [0,1] range
            normalized_box = [
                box[0] / width,  # x1
                box[1] / height, # y1
                box[2] / width,  # x2
                box[3] / height  # y2
            ]
            
            # Initialize state with image
            state = self.processor.set_image(np_image)
            
            # Add geometric prompt (bounding box)
            state = self.processor.add_geometric_prompt(normalized_box, True, state)
            
            # Extract results
            masks = state.get("masks_logits", torch.empty(0, 0, 0, device=self.device, dtype=self.dtype))
            scores = state.get("scores", torch.empty(0, device=self.device, dtype=self.dtype))
            logits = state.get("masks_logits", torch.empty(0, 0, 0, device=self.device, dtype=self.dtype))
            presence_score = state.get("presence_score", torch.tensor(0.0, device=self.device, dtype=self.dtype))
            
            # Apply presence score filter
            if presence_score < 0.1 or masks.numel() == 0:
                result = {
                    "masks": torch.zeros((0, height, width), dtype=self.dtype, device=self.device),
                    "scores": torch.zeros(0, dtype=self.dtype, device=self.device),
                    "logits": torch.zeros((0, height, width), dtype=self.dtype, device=self.device),
                    "presence_score": None,
                    "box_used": box
                }
            else:
                # Squeeze dimension if needed
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)
                    logits = logits.squeeze(1)
                    
                # Apply confidence threshold
                if scores.numel() > 0:
                    valid_mask = scores >= self.confidence_threshold
                    if valid_mask.any():
                        masks = masks[valid_mask]
                        scores = scores[valid_mask]
                        logits = logits[valid_mask]
                
                result = {
                    "masks": masks.cpu(),
                    "scores": scores.cpu(),
                    "logits": logits.cpu(),
                    "presence_score": presence_score.cpu(),
                    "box_used": box
                }
            
            # Clear GPU cache to free memory
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
            return result
            
        except Exception as e:
            print(f"Error during box-based segmentation: {e}")
            import traceback
            traceback.print_exc()
            
            # Clear GPU cache on error
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
            return {
                "masks": torch.zeros((0, 0, 0), dtype=self.dtype),
                "scores": torch.zeros(0),
                "logits": torch.zeros((0, 0, 0)),
                "presence_score": None,
                "box_used": box
            }

    def ensemble_masks(self, masks_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Ensemble multiple masks for the same class using max fusion
        
        Args:
            masks_list: List of mask tensors for the same class
            
        Returns:
            Fused mask tensor
        """
        if not masks_list:
            return torch.empty(0, dtype=self.dtype)
        
        # Stack all masks and take the maximum along the first dimension
        stacked_masks = torch.stack(masks_list)
        ensembled_mask = stacked_masks.max(0)[0]
        return ensembled_mask

    def create_prediction_map(self, class_results: Dict, image_shape: tuple) -> np.ndarray:
        """
        Create a prediction map assigning each pixel to a class based on mask confidence
        
        Args:
            class_results: Results from segmentation grouped by class
            image_shape: Shape of the original image (H, W, C)
            
        Returns:
            prediction_map: Array of shape (H, W) with class IDs for each pixel
        """
        height, width = image_shape[0], image_shape[1]
        prediction_map = np.zeros((height, width), dtype=np.int32)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        
        # Create a mapping from class names to IDs
        class_names = list(class_results.keys())
        class_to_id = {name: idx+1 for idx, name in enumerate(class_names)}  # Start from 1 to avoid background (0)
        
        # Process each class result
        for class_name, result in class_results.items():
            masks = result["masks"]
            scores = result["scores"]
            
            if masks.numel() == 0:
                continue
                
            # Ensure masks are on CPU and convert to numpy
            masks_np = masks.cpu().numpy()
            scores_np = scores.cpu().numpy() if scores.numel() > 0 else np.array([])
            
            # Process each mask for this class
            for i, mask in enumerate(masks_np):
                # Get the score for this mask (default to 1.0 if no scores available)
                score = scores_np[i] if i < len(scores_np) else 1.0
                
                # Create binary mask
                binary_mask = (mask > 0.5).astype(bool)
                
                # Update prediction map where this mask has higher confidence
                update_mask = binary_mask & (score > confidence_map)
                prediction_map[update_mask] = class_to_id[class_name]
                confidence_map[update_mask] = score
        
        return prediction_map