import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData, InstanceData
from mmseg.structures import SegDataSample
from mmseg.registry import MODELS
from PIL import Image


@MODELS.register_module()
class BasicSAM3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device='cuda:0',  # 默认使用cuda:0
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=False,  # 改为False，只使用语义输出
                 temperature=1.0,  # 温度参数，用于LogSumExp
                 **kwargs):
        super().__init__()
        
        # Convert device to torch device if it's a string
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Initialize SAM3 model
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # 获取checkpoint路径，如果不存在则使用默认值
        checkpoint_path = kwargs.get('checkpoint_path', '/data/public/sam3/sam3.pt')
        bpe_path = kwargs.get('bpe_path', '/data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
        
        model = build_sam3_image_model(
            bpe_path=bpe_path, 
            checkpoint_path=checkpoint_path, 
            device=self.device  # 使用指定的设备
        )
        # 确保模型参数也在正确的设备上
        model = model.to(self.device)
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=self.device)
        
        # Load class names and indices
        self.query_words, self.query_idx = self.get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(self.device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder  # 现在只使用语义输出
        self.temperature = temperature  # 用于LogSumExp的温度参数

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        # 为每个查询存储分割logits，初始化为负无穷以支持LogSumExp
        seg_logits = torch.full((self.num_queries, h, w), float('-inf'), device=self.device, dtype=torch.float32)

        with torch.no_grad():
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                # 只使用语义分割头
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if len(semantic_logits.shape) == 4:  # 处理不同形状的输出
                        semantic_logits = semantic_logits.squeeze(0).squeeze(0)
                    elif len(semantic_logits.shape) == 3:
                        semantic_logits = semantic_logits.squeeze(0)
                    
                    if semantic_logits.shape != (h, w):
                        semantic_logits = F.interpolate(
                            semantic_logits.unsqueeze(0).unsqueeze(0) if len(semantic_logits.shape) == 2 else semantic_logits.unsqueeze(0), 
                            size=(h, w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze()
                
                    # 更新seg_logits中的对应位置
                    seg_logits[query_idx] = semantic_logits
                
                # Apply presence score if enabled
                if self.use_presence_score:
                    presence_score = inference_state["presence_score"]
                    seg_logits[query_idx] = seg_logits[query_idx] * presence_score
                
        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        
        # Initialize accumulators - 使用logits累积和计数矩阵
        # Initialize preds with -inf for LogSumExp identity? 
        # But we are adding to it. Let's stick to the reference logic which initializes with 0 
        # but uses a specific LogSumExp update rule that accounts for averaging.
        # Reference:
        # preds = torch.zeros(...)
        # update: preds = logsumexp([preds, crop + log(count)]) - log(count + 1)
        # This formula effectively maintains the running LogSumExp average.
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device, dtype=torch.float32)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                
                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))
                
                # Inference on crop
                crop_seg_logit = self._inference_single_view(crop_img)
                
                # Accumulate results using LogSumExp for soft aggregation
                # Formula: LSE(a, b) = log(exp(a) + exp(b))
                # To average N items using LSE incrementally:
                # NewAvg = LSE(OldAvg * (N-1), NewItem) - log(N)  <-- This is complex because OldAvg is already divided.
                # The reference uses:
                # preds_new = LSE(preds_old, crop_logit + log(count_old)) - log(count_old + 1)
                # Let's verify: 
                # exp(preds_new) = (exp(preds_old) + exp(crop_logit + log(count_old))) / (count_old + 1)
                # exp(preds_new) = (exp(preds_old) + exp(crop_logit)*count_old) / (count_old + 1)
                # This doesn't look like a standard average accumulation if preds_old is just sum.
                # However, if preds stores the LSE-sum (log of sum of exponentials), then:
                # SumExp_new = SumExp_old + Exp(crop)
                # LSE_sum_new = log(SumExp_new) = log(exp(LSE_sum_old) + exp(crop))
                # Average_logit = LSE_sum_new - log(count)
                
                # The reference code snippet provided:
                # preds[:, y1:y2, x1:x2] = torch.logsumexp(
                #     torch.stack([
                #         preds[:, y1:y2, x1:x2], 
                #         crop_seg_logit + torch.log(count_mat[:, y1:y2, x1:x2] + 1e-8)
                #     ]), 
                #     dim=0
                # ) - torch.log(count_mat[:, y1:y2, x1:x2] + 1.0)
                
                # This implies `preds` holds the `LSE_sum` (log of sum of exponentials) of previous crops?
                # No, if preds was LSE_sum, we would just do LSE(preds, crop).
                # The term `crop + log(count)` suggests it's trying to weight the new crop by the current count?
                # Actually, let's just follow the reference implementation exactly as requested.
                
                preds[:, y1:y2, x1:x2] = torch.logsumexp(
                    torch.stack([
                        preds[:, y1:y2, x1:x2], 
                        crop_seg_logit + torch.log(count_mat[:, y1:y2, x1:x2] + 1e-8)
                    ]), 
                    dim=0
                ) - torch.log(count_mat[:, y1:y2, x1:x2] + 1.0)
                
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing: Map queries to classes using soft aggregation
            if self.num_cls != self.num_queries:
                # Create class-wise logits by aggregating query logits belonging to the same class
                class_logits = torch.full((self.num_cls, *seg_logits.shape[1:]), float('-inf'), device=self.device, dtype=torch.float32)
                
                for cls_idx in range(self.num_cls):
                    # Find all queries that belong to this class
                    query_mask = (self.query_idx == cls_idx)
                    if query_mask.any():
                        # Get logits for queries of this class
                        cls_query_logits = seg_logits[query_mask]
                        # Use LogSumExp to aggregate logits of the same class
                        if cls_query_logits.numel() > 0:
                            # Reshape to (num_queries_for_class, height, width) and apply LogSumExp
                            class_logits[cls_idx] = torch.logsumexp(cls_query_logits / self.temperature, dim=0) * self.temperature

                seg_logits = class_logits

            # Apply probability threshold after aggregation
            seg_pred = torch.argmax(seg_logits, dim=0)
            max_vals = torch.max(seg_logits, dim=0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            # Prepare data for AP/IoU evaluation
            # 1. Semantic Segmentation Output (Standard)
            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })

            # 2. Instance/Panoptic-like Output for AP Evaluation
            # AP evaluators often expect 'pred_instances' with masks and scores
            masks_list = []
            scores_list = []
            labels_list = []

            # Iterate over each class to extract binary masks and scores
            for cls_idx in range(self.num_cls):
                if cls_idx == self.bg_idx:
                    continue
                
                # Get logits for this specific class
                cls_logits = seg_logits[cls_idx]
                
                # Create binary mask: 
                # A pixel belongs to this class if it's the argmax AND the logit is above threshold
                # Or simply if the logit is high enough? 
                # Standard practice for "AP" in semantic context often uses the confident regions of each class.
                # Here we use: Mask is where this class is the winner AND confidence > thresh
                is_max = (seg_pred == cls_idx)
                is_confident = (cls_logits >= self.prob_thd)
                mask = is_max & is_confident
                
                if mask.sum() > 0:
                    masks_list.append(mask.float())
                    # Score: Mean of logits in the masked region, or max logit?
                    # Using max logit of the map as presence score is common for image-level or weakly supervised
                    # For pixel-level AP, usually the mask quality matters. 
                    # Let's use the max value of the logits for this class as the detection score.
                    score = torch.max(cls_logits[mask])
                    scores_list.append(score)
                    labels_list.append(torch.tensor(cls_idx, device=self.device))

            if masks_list:
                masks = torch.stack(masks_list, dim=0)
                scores = torch.stack(scores_list, dim=0)
                labels = torch.stack(labels_list, dim=0)
                
                instances = InstanceData()
                instances.masks = masks.cpu().numpy() # AP eval often expects numpy masks or tensors
                instances.scores = scores.cpu()
                instances.labels = labels.cpu()
                
                data_samples[i].pred_instances = instances
            else:
                # Empty instances
                instances = InstanceData()
                instances.masks = torch.zeros((0, *seg_pred.shape), dtype=torch.float32).cpu().numpy()
                instances.scores = torch.zeros((0,), dtype=torch.float32).cpu()
                instances.labels = torch.zeros((0,), dtype=torch.int64).cpu()
                data_samples[i].pred_instances = instances
            
        return data_samples
    
    def _forward(self, data_samples):
        """Forward function for training mode.
        
        Since SAM3 is a prompt-based model and typically used for zero-shot inference
        or fine-tuning with specific adapters, this method currently acts as a placeholder
        or delegates to predict logic if needed during a custom training loop.
        """
        return self.predict(None, data_samples)

    def inference(self, img, batch_img_metas):
        """Inference function.
        
        Args:
            img (Tensor): Input images.
            batch_img_metas (list[dict]): Meta information of each image.
            
        Returns:
            Tensor: Segmentation logits.
        """
        # In our implementation, we rely on loading images from paths in data_samples
        # because SAM3 processor expects PIL images. 
        # This method is kept for compatibility but might not be directly used 
        # if predict() overrides the flow.
        return self.encode_decode(img, batch_img_metas)

    def encode_decode(self, inputs, batch_img_metas):
        """Encode images with backbone and decode into a semantic segmentation map of the same size as input."""
        # Note: 'inputs' here are usually tensors from dataloader, but our logic 
        # in predict/slide_inference expects PIL images loaded from path.
        # We iterate through batch_img_metas to get paths, similar to predict.
        
        seg_logits_list = []
        for i, meta in enumerate(batch_img_metas):
            image_path = meta.get('img_path')
            if image_path:
                image = Image.open(image_path).convert('RGB')
            else:
                # Fallback if path is missing, though unlikely in standard pipeline
                continue
                
            ori_shape = meta['ori_shape']
            
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)
                
            # Resize to original shape if necessary
            if seg_logits.shape[-2:] != ori_shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing: Map queries to classes using soft aggregation
            if self.num_cls != self.num_queries:
                # Create class-wise logits by aggregating query logits belonging to the same class
                class_logits = torch.full((self.num_cls, *seg_logits.shape[1:]), float('-inf'), device=self.device, dtype=torch.float32)
                
                for cls_idx in range(self.num_cls):
                    # Find all queries that belong to this class
                    query_mask = (self.query_idx == cls_idx)
                    if query_mask.any():
                        # Get logits for queries of this class
                        cls_query_logits = seg_logits[query_mask]
                        # Use LogSumExp to aggregate logits of the same class
                        if cls_query_logits.numel() > 0:
                            # Reshape to (num_queries_for_class, height, width) and apply LogSumExp
                            class_logits[cls_idx] = torch.logsumexp(cls_query_logits / self.temperature, dim=0) * self.temperature

                seg_logits = class_logits
                
            seg_logits_list.append(seg_logits)
            
        if seg_logits_list:
            return torch.stack(seg_logits_list, dim=0)
        return None

    def extract_feat(self, inputs):
        """Extract features from images.
        
        For SAM3, this effectively runs the inference to get logits/masks.
        """
        # This method is often called by base class during inference/testing.
        # We can return None or dummy features if the base class requires it,
        # but logically our "features" are the segmentation results.
        pass

    def loss(self, inputs, data_samples):
        """Calculate losses.
        
        Since this is primarily an inference wrapper for a pre-trained SAM3,
        we return empty losses unless specific fine-tuning losses are implemented.
        """
        return dict()

    def get_cls_idx(self, path):
        with open(path, 'r') as f:
            name_sets = f.readlines()
        num_cls = len(name_sets)

        class_names, class_indices = [], []
        for idx in range(num_cls):
            names_i = name_sets[idx].split(',')
            names_i = [i.strip() for i in names_i]
            class_names += names_i
            class_indices += [idx for _ in range(len(names_i))]
        class_names = [item.replace('\n', '') for item in class_names]
        return class_names, class_indices


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices