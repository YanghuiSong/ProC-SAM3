import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image
import pickle
import os

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device='cuda',  # 修改为字符串参数，便于配置
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 expanded_prompt_pool_path=None,  # 新增参数：扩展提示池路径
                 optimize_method='guided',  # 新增参数：优化方法
                 **kwargs):
        super().__init__()
        
        # 处理设备参数
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path=f"/data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='/data/public/sam3/sam3.pt', 
            device=self.device  # 使用传递的设备参数
        )
        # 确保模型参数也在正确的设备上
        model = model.to(self.device)
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=self.device)
        
        # Load class names and indices
        self.query_words, self.query_idx = self.get_cls_idx(classname_path)
        
        print(f"Initial query words from {classname_path}: {len(self.query_words)}")
        for i, (word, idx) in enumerate(zip(self.query_words, self.query_idx[:len(self.query_words)])):
            print(f"  {i+1}. {word} (index {idx})")
        
        # 加载扩展提示池（如果提供路径）
        self.expanded_prompt_pool = None
        print(f"Looking for expanded prompt pool at: {expanded_prompt_pool_path}")
        if expanded_prompt_pool_path and os.path.exists(expanded_prompt_pool_path):
            print(f"Loading expanded prompt pool from {expanded_prompt_pool_path}")
            with open(expanded_prompt_pool_path, 'rb') as f:
                self.expanded_prompt_pool = pickle.load(f)
            
            # 输出加载的扩展提示池内容
            print(f"Loaded expanded prompt pool with {len(self.expanded_prompt_pool)} classes:")
            for class_name, variants in self.expanded_prompt_pool.items():
                print(f"  {class_name}: {variants}")
            
            # 使用扩展提示池替换或扩充原始查询词
            expanded_query_words = []
            expanded_query_idx = []
            
            for original_idx, class_name in enumerate(self.query_words):
                # 检查是否是基本类名（如"building"），然后查找包含它的复合键（如"building,house"）
                matched = False
                for pool_key in self.expanded_prompt_pool.keys():
                    # 检查当前类名是否是池中某个键的一部分
                    if class_name in pool_key.split(','):
                        # 使用找到的键对应的扩展提示词
                        for prompt_variant in self.expanded_prompt_pool[pool_key]:
                            # 处理逗号分隔的提示词变体
                            if ',' in prompt_variant:
                                # 如果提示词本身包含逗号，分割并添加每一个
                                sub_variants = [pv.strip() for pv in prompt_variant.split(',')]
                                for sub_variant in sub_variants:
                                    if sub_variant not in expanded_query_words:  # 避免重复
                                        expanded_query_words.append(sub_variant)
                                        expanded_query_idx.append(self.query_idx[original_idx])
                            else:
                                if prompt_variant not in expanded_query_words:  # 避免重复
                                    expanded_query_words.append(prompt_variant)
                                    expanded_query_idx.append(self.query_idx[original_idx])
                        matched = True
                        break
                
                if not matched:
                    # 如果没有找到匹配的扩展提示词，则使用原始提示词
                    expanded_query_words.append(class_name)
                    original_class_idx = self.query_idx[original_idx]
                    expanded_query_idx.append(original_class_idx)
            
            self.query_words = expanded_query_words
            self.query_idx = torch.tensor(expanded_query_idx, dtype=torch.int64, device=self.device)
            
            # 输出实际使用的查询词以验证
            print(f"Using expanded prompt pool with {len(self.query_words)} query words:")
            for i, (word, idx) in enumerate(zip(self.query_words, self.query_idx)):
                print(f"  {i+1:2d}. {word} (class {int(idx)})")
        else:
            if expanded_prompt_pool_path:
                print(f"Expanded prompt pool file not found: {expanded_prompt_pool_path}")
            else:
                print("No expanded prompt pool path provided")
            
            self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(self.device)
            print(f"Using original {len(self.query_words)} query words without expansion:")
            for i, (word, idx) in enumerate(zip(self.query_words, self.query_idx)):
                print(f"  {i+1:2d}. {word} (class {int(idx)})")
        
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_words)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        # 移除autocast以避免精度类型不匹配问题
        with torch.no_grad():
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            # instance_mask = inference_state['masks'][inst_id].squeeze()
                            
                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                    
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                
                if self.use_presence_score:
                    seg_logits[query_idx] = seg_logits[query_idx] * inference_state["presence_score"]
                
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
        
        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
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
                
                # Inference on crop - removing autocast to avoid precision issues
                crop_seg_logit = self._inference_single_view(crop_img)
                
                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
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
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            seg_pred = torch.argmax(seg_logits, dim=0)
            
            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })
            
        return data_samples
    
    def _forward(self, data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """

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