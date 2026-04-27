"""
自定义计数分析模块
用于对图像中的对象进行计数分析，并与SAM3的分割结果进行对比
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
from core.qwen_agent import QwenAgent
from sam3_segmentor import SegEarthOV3Segmentation
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class CustomCountingAnalyzer:
    """
    自定义计数分析器
    实现Qwen3按扩展提示池进行常态化类别计数分析
    """
    
    def __init__(self, 
                 qwen_model_path=None, 
                 sam3_model_path="/data/public/sam3/sam3.pt",
                 device="cuda"):
        """
        初始化分析器
        
        Args:
            qwen_model_path: Qwen3模型路径
            sam3_model_path: SAM3模型路径
            device: 运行设备
        """
        self.device = torch.device(device)
        self.qwen_agent = QwenAgent(model_path=qwen_model_path, device=device)
        
        # 初始化SAM3模型用于对比分析
        self.sam3_model = build_sam3_image_model(
            bpe_path="/data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=sam3_model_path,
            device=self.device
        )
        self.sam3_processor = Sam3Processor(
            self.sam3_model, 
            confidence_threshold=0.5, 
            device=self.device
        )
    
    def analyze_image_counts(self, 
                           image_path, 
                           prompt_pool_path=None, 
                           class_names=None,
                           use_predefined_prompts=True):
        """
        对单个图像进行计数分析
        
        Args:
            image_path: 图像路径
            prompt_pool_path: 提示池路径
            class_names: 类别名称列表
            use_predefined_prompts: 是否使用预定义提示
        
        Returns:
            dict: 包含Qwen3计数和SAM3分割结果的字典
        """
        # 加载提示池
        prompt_pool = None
        if prompt_pool_path and os.path.exists(prompt_pool_path):
            with open(prompt_pool_path, 'rb') as f:
                prompt_pool = pickle.load(f)
        
        # Qwen3计数分析
        if use_predefined_prompts and prompt_pool:
            qwen_counts = self.qwen_agent.count_objects_by_prompt_pool(image_path, prompt_pool)
        else:
            # 如果没有预定义提示池，使用基本类别名称
            if class_names:
                basic_prompt_pool = {name: [name] for name in class_names}
                qwen_counts = self.qwen_agent.count_objects_by_prompt_pool(image_path, basic_prompt_pool)
            else:
                qwen_counts = {}
        
        # SAM3分割计数分析
        sam3_counts = self._sam3_count_objects(image_path, class_names)
        
        return {
            'qwen_counts': qwen_counts,
            'sam3_counts': sam3_counts,
            'image_path': image_path
        }
    
    def _sam3_count_objects(self, image_path, class_names):
        """
        使用SAM3模型对图像中的对象进行计数
        
        Args:
            image_path: 图像路径
            class_names: 类别名称列表
        
        Returns:
            dict: SAM3计数结果
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            
            # 设置图像
            inference_state = self.sam3_processor.set_image(image)
            
            # 对每个类别进行计数
            counts = {}
            for class_name in class_names or []:
                # 重置提示
                self.sam3_processor.reset_all_prompts(inference_state)
                
                # 设置文本提示
                inference_state = self.sam3_processor.set_text_prompt(
                    state=inference_state, 
                    prompt=class_name
                )
                
                # 获取掩码数量（即对象数量）
                if hasattr(inference_state, 'masks_logits'):
                    mask_count = inference_state['masks_logits'].shape[0] if inference_state['masks_logits'].numel() > 0 else 0
                else:
                    mask_count = 0
                
                counts[class_name] = mask_count
            
            return counts
        except Exception as e:
            print(f"SAM3 counting error for {image_path}: {e}")
            return {}
    
    def batch_analyze(self, 
                      image_paths, 
                      prompt_pool_path=None, 
                      class_names=None,
                      output_dir="./counting_analysis_results/"):
        """
        批量分析图像
        
        Args:
            image_paths: 图像路径列表
            prompt_pool_path: 提示池路径
            class_names: 类别名称列表
            output_dir: 输出目录
        
        Returns:
            list: 分析结果列表
        """
        results = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            print(f"Analyzing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            result = self.analyze_image_counts(
                img_path, 
                prompt_pool_path, 
                class_names
            )
            results.append(result)
        
        # 保存结果
        output_path = os.path.join(output_dir, "counting_analysis_results.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Batch analysis completed. Results saved to {output_path}")
        
        # 生成对比报告
        self._generate_comparison_report(results, output_dir)
        
        return results
    
    def _generate_comparison_report(self, results, output_dir):
        """
        生成Qwen3与SAM3计数结果的对比报告
        """
        report_path = os.path.join(output_dir, "comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Qwen3 vs SAM3 Counting Analysis Comparison Report\n")
            f.write("=" * 60 + "\n\n")
            
            # 统计信息
            all_classes = set()
            for result in results:
                all_classes.update(result['qwen_counts'].keys())
                all_classes.update(result['sam3_counts'].keys())
            
            for class_name in sorted(all_classes):
                f.write(f"\nClass: {class_name}\n")
                f.write("-" * 30 + "\n")
                
                total_qwen = 0
                total_sam3 = 0
                
                for result in results:
                    qwen_count = result['qwen_counts'].get(class_name, 0)
                    sam3_count = result['sam3_counts'].get(class_name, 0)
                    
                    f.write(f"  {os.path.basename(result['image_path'])}: Qwen3={qwen_count}, SAM3={sam3_count}\n")
                    
                    total_qwen += qwen_count
                    total_sam3 += sam3_count
                
                f.write(f"  Totals: Qwen3={total_qwen}, SAM3={total_sam3}\n")
        
        print(f"Comparison report saved to {report_path}")