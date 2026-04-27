import json
import numpy as np
from collections import defaultdict

def analyze_results(json_file_path):
    """
    分析评估结果，比较原始和扩展提示词的效果
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 按数据集和提示词类型组织结果
    dataset_results = {}
    for result in results:
        dataset = result['dataset']
        prompt_type = result['prompt_type']
        
        if dataset not in dataset_results:
            dataset_results[dataset] = {}
        
        dataset_results[dataset][prompt_type] = result
    
    print("="*120)
    print("DETAILED COMPARISON BETWEEN ORIGINAL AND EXPANDED PROMPTS")
    print("="*120)
    
    # 为每个数据集比较原始和扩展提示词
    for dataset, data in dataset_results.items():
        print(f"\n{dataset.upper()} DATASET:")
        print("-" * 60)
        
        original_data = data.get('original')
        expanded_data = data.get('expanded')
        
        if original_data and expanded_data:
            print(f"Original Mean IoU: {original_data['overall_mean_iou']:.3f}")
            print(f"Expanded Mean IoU: {expanded_data['overall_mean_iou']:.3f}")
            
            if expanded_data['overall_mean_iou'] > original_data['overall_mean_iou']:
                improvement = expanded_data['overall_mean_iou'] - original_data['overall_mean_iou']
                print(f"Improvement with expanded prompts: +{improvement:.3f}")
            else:
                improvement = original_data['overall_mean_iou'] - expanded_data['overall_mean_iou']
                print(f"Original prompts performed better by: +{improvement:.3f}")
            
            print("\nPer-class comparison:")
            # 获取所有类别名称（来自原始或扩展，取并集）
            all_classes = set(original_data['per_class_avg_iou'].keys()).union(
                set(expanded_data['per_class_avg_iou'].keys())
            )
            
            print(f"{'Class':<50} {'Original':<10} {'Expanded':<10} {'Diff':<10}")
            print("-" * 80)
            
            for class_name in sorted(all_classes):
                orig_iou = original_data['per_class_avg_iou'].get(class_name)
                exp_iou = expanded_data['per_class_avg_iou'].get(class_name)
                
                if orig_iou is not None and exp_iou is not None:
                    diff = exp_iou - orig_iou
                    print(f"{class_name:<50} {orig_iou:<10.3f} {exp_iou:<10.3f} {diff:<10.3f}")
                elif orig_iou is not None:
                    print(f"{class_name:<50} {orig_iou:<10.3f} {'N/A':<10} {'N/A':<10}")
                elif exp_iou is not None:
                    print(f"{class_name:<50} {'N/A':<10} {exp_iou:<10.3f} {'N/A':<10}")
    
    print("\n" + "="*120)
    print("BEST IMAGES FOR EACH DATASET (BY MEAN IoU WITH EXPANDED PROMPTS)")
    print("="*120)
    
    # 找出每个数据集中扩展提示词效果最好的图像
    for dataset, data in dataset_results.items():
        expanded_data = data.get('expanded')
        if expanded_data:
            # 按mean_iou排序
            sorted_images = sorted(
                expanded_data['per_image_results'], 
                key=lambda x: x['mean_iou'], 
                reverse=True
            )
            
            print(f"\n{dataset.upper()} - Top 5 Images with Expanded Prompts:")
            print("-" * 80)
            print(f"{'Rank':<4} {'Mean IoU':<10} {'Image Path':<50}")
            print("-" * 80)
            
            for i, img_result in enumerate(sorted_images[:5]):
                mean_iou = img_result['mean_iou']
                img_path = img_result['image_path'].split('/')[-1]  # 只显示文件名
                print(f"{i+1:<4} {mean_iou:<10.3f} {img_path:<50}")
    
    print("\n" + "="*120)
    print("IMAGES WITH LARGEST IMPROVEMENT FROM EXPANDED PROMPTS")
    print("="*120)
    
    # 找出每张图像使用扩展提示词的最大改进
    for dataset, data in dataset_results.items():
        original_data = data.get('original')
        expanded_data = data.get('expanded')
        
        if original_data and expanded_data:
            # 建立图像路径到结果的映射
            orig_img_map = {r['image_path']: r for r in original_data['per_image_results']}
            exp_img_map = {r['image_path']: r for r in expanded_data['per_image_results']}
            
            improvements = []
            for img_path, exp_result in exp_img_map.items():
                if img_path in orig_img_map:
                    orig_result = orig_img_map[img_path]
                    improvement = exp_result['mean_iou'] - orig_result['mean_iou']
                    improvements.append({
                        'image_path': img_path,
                        'original_mean_iou': orig_result['mean_iou'],
                        'expanded_mean_iou': exp_result['mean_iou'],
                        'improvement': improvement
                    })
            
            # 按改进程度排序
            improvements.sort(key=lambda x: x['improvement'], reverse=True)
            
            print(f"\n{dataset.upper()} - Top 5 Images with Largest Improvement:")
            print("-" * 100)
            print(f"{'Image Path':<50} {'Orig IoU':<10} {'Exp IoU':<10} {'Improvement':<12}")
            print("-" * 100)
            
            for imp in improvements[:5]:
                img_path = imp['image_path'].split('/')[-1]
                print(f"{img_path:<50} {imp['original_mean_iou']:<10.3f} {imp['expanded_mean_iou']:<10.3f} {imp['improvement']:<12.3f}")

if __name__ == "__main__":
    analyze_results("evaluation_results.json")