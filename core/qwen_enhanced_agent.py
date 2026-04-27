# core/qwen_enhanced_agent.py
import torch
import json
import re
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

class EnhancedQwenAgent:
    """增强版Qwen3-VL智能代理，为SAM3提供智能分割策略"""
    
    def __init__(self, model_path, device=None):
        print("加载增强版Qwen3-VL模型...")
        
        self.device = device or "cuda:0"
        self.model_path = model_path
        
        try:
            # 尝试导入新版本的transformers
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                print("⚠ AutoModelForImageTextToText不可用，尝试使用旧版本...")
                from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
            
            # 加载模型
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # 加载处理器
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            print("✓ 增强版Qwen3-VL模型加载成功")
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("尝试备用加载方法...")
            self._load_with_fallback(model_path)
    
    def _load_with_fallback(self, model_path):
        """备用加载方法"""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            print("✓ 使用备用方法加载成功")
            
        except Exception as e:
            print(f"✗ 备用加载方法也失败: {e}")
            raise RuntimeError("无法加载Qwen3-VL模型")
    
    # ... 其他方法保持不变 ...
    
    def intelligent_scene_analysis(self, image_path) -> Dict:
        """
        智能场景分析，返回结构化分析结果
        为SAM3分割提供决策支持
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            
            # 系统提示词 - 设计用于生成SAM3友好的分割策略
            system_prompt = """You are an expert image segmentation strategist. Analyze this image and generate an optimal segmentation strategy for SAM3 model.

SAM3 CAPABILITIES & LIMITATIONS:
- Understands simple nouns and short phrases (1-3 words)
- Can segment multiple instances of same object
- Works best with clear, concrete object categories
- Struggles with abstract concepts and complex phrases

YOUR TASK:
1. Identify ALL distinct objects/regions in the image
2. For each object, provide SAM3-friendly prompts
3. Group similar objects intelligently
4. Suggest segmentation order and strategy

OUTPUT FORMAT (JSON):
{
    "primary_objects": [
        {"name": "car", "prompts": ["car", "vehicle", "automobile"], "priority": 1, "expected_count": 2},
        {"name": "building", "prompts": ["building", "house", "structure"], "priority": 1, "expected_count": 3}
    ],
    "secondary_objects": [
        {"name": "tree", "prompts": ["tree"], "priority": 2, "expected_count": 5}
    ],
    "background_regions": [
        {"name": "sky", "prompts": ["sky", "blue sky"]},
        {"name": "road", "prompts": ["road", "street", "pavement"]}
    ],
    "segmentation_strategy": "Start with large distinct objects, then details",
    "total_expected_regions": 15
}"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 准备输入
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成分析结果
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=0.1,
                    do_sample=False,
                    num_beams=3
                )
            
            # 解码输出
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # 提取JSON部分
            analysis_result = self._extract_json_response(generated_text)
            
            if not analysis_result:
                print("⚠ 无法解析JSON，使用备用分析")
                analysis_result = self._fallback_analysis(image)
            
            print(f"✓ 场景分析完成，预期区域数: {analysis_result.get('total_expected_regions', '未知')}")
            return analysis_result
            
        except Exception as e:
            print(f"场景分析失败: {e}")
            return self._fallback_analysis(image_path)
    
    def generate_sam3_prompts(self, analysis_result: Dict) -> List[Dict]:
        """
        为SAM3生成优化的提示词序列
        考虑SAM3的语义理解限制
        """
        prompts = []
        
        # 1. 主要物体（高优先级）
        for obj in analysis_result.get("primary_objects", []):
            for prompt in obj.get("prompts", [obj["name"]]):
                if len(prompt.split()) <= 3:  # SAM3能理解的短短语
                    prompts.append({
                        "text": prompt,
                        "priority": obj.get("priority", 1),
                        "category": obj["name"],
                        "type": "primary"
                    })
        
        # 2. 次要物体
        for obj in analysis_result.get("secondary_objects", []):
            for prompt in obj.get("prompts", [obj["name"]]):
                if len(prompt.split()) <= 3:
                    prompts.append({
                        "text": prompt,
                        "priority": obj.get("priority", 2),
                        "category": obj["name"],
                        "type": "secondary"
                    })
        
        # 3. 背景区域
        for region in analysis_result.get("background_regions", []):
            for prompt in region.get("prompts", [region["name"]]):
                if len(prompt.split()) <= 2:  # 背景用更简单的提示
                    prompts.append({
                        "text": prompt,
                        "priority": 3,
                        "category": region["name"],
                        "type": "background"
                    })
        
        # 按优先级排序
        prompts.sort(key=lambda x: x["priority"])
        
        # 去重
        unique_prompts = []
        seen = set()
        for p in prompts:
            if p["text"] not in seen:
                seen.add(p["text"])
                unique_prompts.append(p)
        
        print(f"✓ 生成 {len(unique_prompts)} 个SAM3优化提示词")
        return unique_prompts
    
    def analyze_segmentation_results(self, image_path, segmentation_results: Dict) -> Dict:
        """
        分析分割结果，识别遗漏区域并生成补充提示
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # 创建结果摘要
            result_summary = f"Current segmentation has found {len(segmentation_results.get('masks', []))} regions."
            
            if segmentation_results.get('categories'):
                category_counts = {}
                for cat in segmentation_results['categories']:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                result_summary += " Categories found: " + ", ".join([f"{k}({v})" for k, v in category_counts.items()])
            
            # 请求Qwen3分析缺失部分
            analysis_prompt = f"""Analyze this image and suggest what objects/regions might be MISSING from current segmentation.

Current segmentation results: {result_summary}

What important objects or regions do you still see in the image that haven't been segmented yet?
Provide simple SAM3-friendly prompts (1-3 words each) for missing regions.

Focus on:
1. Small objects that might have been missed
2. Parts of large objects not fully segmented
3. Background elements that should be separated

Output as JSON list of prompts: ["missing_object1", "missing_object2", ...]"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": analysis_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=False
                )
            
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # 提取缺失的提示词
            missing_prompts = self._extract_missing_prompts(generated_text)
            
            return {
                "missing_prompts": missing_prompts,
                "current_coverage": len(segmentation_results.get('masks', [])),
                "needs_refinement": len(missing_prompts) > 0
            }
            
        except Exception as e:
            print(f"分割结果分析失败: {e}")
            return {"missing_prompts": [], "current_coverage": 0, "needs_refinement": False}
    
    def _extract_json_response(self, text: str) -> Dict:
        """从模型响应中提取JSON"""
        import json
        
        # 尝试找到JSON部分
        try:
            # 查找{...}模式
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # 如果找不到JSON，尝试解析文本
        return self._parse_text_analysis(text)
    
    def _parse_text_analysis(self, text: str) -> Dict:
        """解析文本分析结果"""
        # 提取对象名称
        objects = re.findall(r'["\']([a-zA-Z\s]{2,20})["\']', text)
        
        if not objects:
            # 尝试其他模式
            objects = re.findall(r'(\b[a-zA-Z]{3,15}\b)', text)[:20]
        
        # 构建结构化结果
        result = {
            "primary_objects": [],
            "secondary_objects": [],
            "background_regions": [],
            "segmentation_strategy": "Proceed with identified objects",
            "total_expected_regions": len(objects) * 2  # 估计值
        }
        
        # 常见物体分类
        common_objects = {
            "person": "primary", "car": "primary", "building": "primary", 
            "tree": "secondary", "road": "background", "sky": "background",
            "water": "background", "grass": "background", "house": "primary"
        }
        
        for obj in objects:
            obj_lower = obj.lower()
            obj_type = common_objects.get(obj_lower, "secondary")
            
            if obj_type == "primary":
                result["primary_objects"].append({
                    "name": obj_lower,
                    "prompts": [obj_lower],
                    "priority": 1,
                    "expected_count": 1
                })
            elif obj_type == "background":
                result["background_regions"].append({
                    "name": obj_lower,
                    "prompts": [obj_lower]
                })
            else:
                result["secondary_objects"].append({
                    "name": obj_lower,
                    "prompts": [obj_lower],
                    "priority": 2,
                    "expected_count": 1
                })
        
        return result
    
    def _fallback_analysis(self, image_path) -> Dict:
        """备用分析方法"""
        image = Image.open(image_path).convert("RGB")
        
        # 使用标准物体列表
        return {
            "primary_objects": [
                {"name": "building", "prompts": ["building", "house"], "priority": 1, "expected_count": 3},
                {"name": "vehicle", "prompts": ["car", "vehicle", "truck"], "priority": 1, "expected_count": 2},
                {"name": "person", "prompts": ["person", "human", "people"], "priority": 1, "expected_count": 2}
            ],
            "secondary_objects": [
                {"name": "tree", "prompts": ["tree", "plant"], "priority": 2, "expected_count": 5},
                {"name": "window", "prompts": ["window"], "priority": 2, "expected_count": 10}
            ],
            "background_regions": [
                {"name": "sky", "prompts": ["sky"]},
                {"name": "road", "prompts": ["road", "street"]},
                {"name": "grass", "prompts": ["grass", "field"]}
            ],
            "segmentation_strategy": "Standard fallback strategy",
            "total_expected_regions": 25
        }
    
    def _extract_missing_prompts(self, text: str) -> List[str]:
        """提取缺失的提示词"""
        prompts = []
        
        # 查找列表格式
        list_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
        if list_match:
            items = re.findall(r'["\']([^"\']+)["\']', list_match.group(1))
            prompts.extend(items)
        
        # 查找以逗号分隔的项目
        if not prompts:
            items = re.findall(r'\b([a-z]{3,15})\b', text.lower())
            prompts.extend(items[:10])  # 限制数量
        
        # 去重并过滤
        unique_prompts = []
        seen = set()
        for p in prompts:
            if p not in seen and len(p) >= 3 and len(p.split()) <= 3:
                seen.add(p)
                unique_prompts.append(p)
        
        return unique_prompts