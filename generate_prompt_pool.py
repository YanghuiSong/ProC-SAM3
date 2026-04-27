"""
生成预处理prompt池，用于加速SAM3分割流程
"""
import json
import os
import pickle
from collections import defaultdict
from PIL import Image
import torch
from sam3.agent.qwen3_agent import Qwen3Agent
from config_qwen3_sam3 import QWEN_MODEL_PATH


def generate_prompt_pool(image_paths, class_names, qwen_model_path=None, output_path="./prompt_pool.pkl"):
    """
    使用Qwen3处理少量图像，生成prompt池
    
    Args:
        image_paths: 图像路径列表
        class_names: 类别名称列表
        qwen_model_path: Qwen3模型路径
        output_path: 输出文件路径
    """
    print(f"Processing {len(image_paths)} images to generate prompt pool...")
    
    # 初始化Qwen3代理
    qwen_agent = Qwen3Agent(
        model_path=qwen_model_path or QWEN_MODEL_PATH,
        device="cuda:0"
    )
    
    # 存储所有类别的prompts
    all_prompts = defaultdict(set)  # 使用set去重
    
    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
        
        try:
            # 生成增强描述
            hierarchical_data = qwen_agent.generate_count_and_descriptions(image_path, class_names)
            
            # 收集每个类别的prompts
            for class_name, class_data in hierarchical_data.items():
                # 包括基础类名
                all_prompts[class_name].add(class_name)
                
                # 添加实例特征
                for feature in class_data.get('instance_features', []):
                    if feature and len(feature.split()) <= 3:  # 确保简洁
                        all_prompts[class_name].add(feature)
                
                # 添加语义特征
                for feature in class_data.get('semantic_features', []):
                    if feature and len(feature.split()) <= 3:  # 确保简洁
                        all_prompts[class_name].add(feature)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # 将set转换为list
    prompt_pool = {}
    for class_name, prompts in all_prompts.items():
        prompt_pool[class_name] = list(prompts)
        print(f"{class_name}: {len(prompts)} unique prompts generated")
    
    # 保存prompt池
    with open(output_path, 'wb') as f:
        pickle.dump(prompt_pool, f)
    
    print(f"Prompt pool saved to {output_path}")
    return prompt_pool


def compress_prompt(text):
    """
    将生成的prompt压缩为简洁形式，适合SAM3使用
    """
    # 只保留核心名词短语
    parts = text.split()
    if len(parts) <= 3:
        return text
    
    # 保留最后的名词或名词短语（通常是最重要的部分）
    return ' '.join(parts[-2:])  # 取最后两个词作为关键词


def compress_prompt_pool(prompt_pool):
    """
    压缩prompt池中的所有prompts
    """
    compressed_pool = {}
    
    for class_name, prompts in prompt_pool.items():
        compressed_prompts = set()
        for prompt in prompts:
            compressed = compress_prompt(prompt)
            if compressed and len(compressed.split()) <= 3:  # 确保不超过3个词
                compressed_prompts.add(compressed)
        
        compressed_pool[class_name] = list(compressed_prompts)
        print(f"Compressed {class_name}: {len(prompts)} -> {len(compressed_prompts)} prompts")
    
    return compressed_pool


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate prompt pool using Qwen3')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--class_file', type=str, required=True, help='File containing class names')
    parser.add_argument('--output_path', type=str, default='./prompt_pool.pkl', help='Output file path')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to process (use 10% of dataset)')
    parser.add_argument('--qwen_model_path', type=str, default=None, help='Qwen3 model path')
    
    args = parser.parse_args()
    
    # 读取类别名称
    with open(args.class_file, 'r') as f:
        class_names = [line.strip().split(',')[0] for line in f.readlines()]
    
    # 获取图像列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    
    for root, dirs, files in os.walk(args.image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    # 只使用指定数量的图像
    image_paths = image_paths[:args.num_images]
    
    print(f"Found {len(image_paths)} images to process")
    print(f"Classes: {class_names}")
    
    # 生成prompt池
    prompt_pool = generate_prompt_pool(
        image_paths=image_paths,
        class_names=class_names,
        qwen_model_path=args.qwen_model_path,
        output_path=args.output_path
    )
    
    # 压缩prompt池
    compressed_pool = compress_prompt_pool(prompt_pool)
    
    # 保存压缩后的prompt池
    compressed_output_path = args.output_path.replace('.pkl', '_compressed.pkl')
    with open(compressed_output_path, 'wb') as f:
        pickle.dump(compressed_pool, f)
    
    print(f"Compressed prompt pool saved to {compressed_output_path}")