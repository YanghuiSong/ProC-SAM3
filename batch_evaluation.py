import os
import torch
import numpy as np
from PIL import Image
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import json
import pickle
import tempfile

from sam3_segmentor import SegEarthOV3Segmentation


def compute_iou(pred_mask, gt_mask, num_classes):
    """
    计算IoU指标
    """
    ious = []
    for i in range(num_classes):
        pred_i = (pred_mask == i)
        gt_i = (gt_mask == i)
        
        intersection = np.logical_and(pred_i, gt_i).sum()
        union = np.logical_or(pred_i, gt_i).sum()
        
        if union == 0:
            iou = float('nan')  # 忽略没有对应类别的像素
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.array(ious)


def load_class_names(class_file_path):
    """加载类别名称"""
    with open(class_file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def aggregate_predictions_by_class(seg_logits, query_idx, num_cls):
    """
    根据类别索引聚合预测结果，与sam3_segmentor.py中的逻辑保持一致
    """
    # 如果类别数与查询数不一致，需要进行聚合
    if num_cls != seg_logits.shape[0]:
        seg_logits = seg_logits.unsqueeze(0)
        cls_index = torch.nn.functional.one_hot(query_idx)
        cls_index = cls_index.T.view(num_cls, len(query_idx), 1, 1).to(seg_logits.device)
        aggregated_logits = (seg_logits * cls_index).max(1)[0]
        return torch.argmax(aggregated_logits, dim=0).cpu().numpy()
    else:
        return torch.argmax(seg_logits, dim=0).cpu().numpy()


def create_expanded_prompt_pickle_from_txt(txt_path, temp_pkl_path):
    """
    从文本文件创建扩展提示词的pickle文件，格式与SegEarthOV3Segmentation期望的一致
    """
    if not os.path.exists(txt_path):
        return False
    
    expanded_prompt_pool = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # 解析形如 "building,house,structure" 的行
            parts = [part.strip() for part in line.split(',') if part.strip()]
            if len(parts) > 0:
                # 使用整行内容作为键，各个部分作为该类的扩展提示词
                key = ','.join(parts)  # 使用整行作为键
                expanded_prompt_pool[key] = parts  # 所有部分都是该类的提示词变体
    
    # 保存为pickle文件
    with open(temp_pkl_path, 'wb') as f:
        pickle.dump(expanded_prompt_pool, f)
    
    return True


def evaluate_dataset(img_dir, gt_dir, dataset_name, device, use_expanded_prompts=False, ratio=1.0):
    """
    评估整个数据集
    """
    print(f"Processing {dataset_name} with {'expanded' if use_expanded_prompts else 'original'} prompts...")
    
    # 设置类别文件路径，遵循项目规范使用实际存在的配置文件
    if dataset_name == "Vaihingen":
        class_file = "./configs/cls_vaihingen.txt"
        exp_class_file = "./configs/cls_vaihingen_exp.txt"
    elif dataset_name == "Potsdam":
        class_file = "./configs/cls_potsdam.txt"
        exp_class_file = "./configs/cls_potsdam_exp.txt"
    elif dataset_name == "LoveDA":
        class_file = "./configs/cls_loveda.txt"
        exp_class_file = "./configs/cls_loveda_exp.txt"
    elif dataset_name == "iSAID":
        class_file = "./configs/cls_iSAID.txt"
        exp_class_file = "./configs/cls_iSAID_exp.txt"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 选择使用哪个类别文件
    class_file_to_use = exp_class_file if use_expanded_prompts else class_file
    
    # 获取类别数
    class_names = load_class_names(class_file_to_use)
    num_classes = len(class_names)
    
    # 获取图像文件列表
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(img_dir).glob(f"*{ext}")))
    
    # 根据比例限制处理数量
    num_to_process = int(len(image_files) * ratio)
    image_files = image_files[:num_to_process]
    
    print(f"Found {len(image_files)} images to process (ratio: {ratio})")
    
    total_ious = defaultdict(list)
    per_image_ious = []
    
    # 为扩展提示词创建临时pickle文件
    temp_pkl_path = None
    if use_expanded_prompts:
        # 使用系统的临时目录，确保在Linux服务器上也可以访问
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_pkl_path = tmp_file.name
            if not create_expanded_prompt_pickle_from_txt(exp_class_file, temp_pkl_path):
                print(f"Warning: Could not create expanded prompts for {dataset_name}, falling back to original prompts")
                temp_pkl_path = None
    
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_path.name}")
        
        try:
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            
            # 确定扩展提示池路径
            expanded_prompt_pool_path = temp_pkl_path if temp_pkl_path else None
            
            # 创建模型实例
            model = SegEarthOV3Segmentation(
                classname_path=class_file_to_use,
                device=device,
                prob_thd=0.1,
                confidence_threshold=0.4,
                use_sem_seg=True,
                use_presence_score=True,
                use_transformer_decoder=True,
                expanded_prompt_pool_path=expanded_prompt_pool_path  # 传递扩展提示池路径
            )
            
            # 进行推理
            seg_logits = model._inference_single_view(img)
            
            # 根据模型的类别索引聚合预测结果
            seg_pred = aggregate_predictions_by_class(seg_logits, model.query_idx, model.num_cls)
            
            # 尝试加载真实的ground truth标签
            # 根据数据集名称确定ground truth路径
            gt_path = None
            if dataset_name == "Vaihingen":
                gt_base_path = "/data/public/Vaihingen/ann_dir/val"  # 服务器上的Vaihingen标注路径
            elif dataset_name == "Potsdam":
                gt_base_path = "/data/public/Potsdam/ann_dir/val"  # 服务器上的Potsdam标注路径
            elif dataset_name == "LoveDA":
                gt_base_path = "/data/public/LoveDA/ann_dir/val"  # 服务器上的LoveDA标注路径
            elif dataset_name == "iSAID":
                gt_base_path = "/data/public/iSAID/ann_dir/val"  # 服务器上的iSAID标注路径

            if gt_base_path and os.path.exists(gt_base_path):
                # 构建对应的ground truth文件路径
                img_stem = img_path.stem
                
                # 根据数据集特点使用不同的匹配策略
                if dataset_name == "iSAID":
                    # iSAID: 图像文件名可能包含坐标信息，需要灵活匹配
                    # 例如图像可能是 P1242_512_1408_2048_2944.png
                    # 标签可能是 P1242_512_1408_2048_2944_instance_color_RGB.png 或类似格式
                    for gt_candidate in Path(gt_base_path).glob("*.png"):
                        # 如果标签文件名包含图像文件名的关键部分
                        if img_stem.split('_')[0] in gt_candidate.name and \
                           img_stem.split('_')[1] in gt_candidate.name:
                            gt_path = gt_candidate
                            break
                        # 或者尝试精确匹配
                        gt_stem = gt_candidate.stem
                        if img_stem.replace('_instance_color_RGB', '') == gt_stem or \
                           img_stem == gt_stem.replace('_instance_color_RGB', ''):
                            gt_path = gt_candidate
                            break
                elif dataset_name == "Vaihingen":
                    # Vaihingen: 格式如 3_13_2048_512_2560_1024.png
                    for gt_candidate in Path(gt_base_path).glob("*.png"):
                        if img_stem == gt_candidate.stem:
                            gt_path = gt_candidate
                            break
                        # 尝试匹配部分名称
                        if img_stem.split('_')[0] in gt_candidate.name and \
                           img_stem.split('_')[1] in gt_candidate.name:
                            gt_path = gt_candidate
                            break
                elif dataset_name == "Potsdam":
                    # Potsdam: 格式如 2605.png
                    for gt_candidate in Path(gt_base_path).glob("*.png"):
                        if img_stem == gt_candidate.stem:
                            gt_path = gt_candidate
                            break
                elif dataset_name == "LoveDA":
                    # LoveDA: 尝试直接匹配
                    for gt_candidate in Path(gt_base_path).glob("*.png"):
                        if img_stem == gt_candidate.stem:
                            gt_path = gt_candidate
                            break
            
            if gt_path and os.path.exists(gt_path):
                # 加载真实的ground truth
                gt_img = Image.open(gt_path)
                # 将ground truth转换为numpy数组
                gt_array = np.array(gt_img)
                
                # 如果ground truth是RGB图像，则需要将其转换为类别索引图
                if len(gt_array.shape) == 3:  # RGB图像
                    # 这里需要根据数据集的配色方案将RGB值转换为类别索引
                    # 对于这个示例，我们假设输入是类别索引图
                    gt_mask = gt_array.astype(np.int32)
                else:
                    gt_mask = gt_array
            else:
                # 如果找不到ground truth，使用全零矩阵（表示无分割）
                print(f"Warning: Ground truth not found for {img_path}, using zeros matrix.")
                print(f"Looked in: {gt_base_path}")
                print(f"Expected name similar to: {img_path.stem}")
                gt_mask = np.zeros_like(seg_pred)  # 替换为真实的GT
            
            # 确保ground truth的尺寸与预测结果一致
            if gt_mask.shape != seg_pred.shape:
                print(f"Warning: Ground truth shape {gt_mask.shape} doesn't match prediction shape {seg_pred.shape}")
                # 调整ground truth尺寸以匹配预测结果
                from PIL import Image as PILImage
                gt_resized = np.array(PILImage.fromarray(gt_mask.astype(np.uint8)).resize(
                    (seg_pred.shape[1], seg_pred.shape[0]), resample=PILImage.NEAREST))
                gt_mask = gt_resized
            
            # 计算IoU
            ious = compute_iou(seg_pred, gt_mask, model.num_cls)  # 使用模型的实际类别数
            
            # 保存每张图片的IoU
            image_result = {
                'image_path': str(img_path),
                'ious': ious.tolist(),  # 转换为列表以便JSON序列化
                'mean_iou': float(np.nanmean(ious)),
                'per_class_iou': {class_names[j % len(class_names)]: float(ious[j]) if not np.isnan(ious[j]) else None 
                                 for j in range(len(ious))}
            }
            per_image_ious.append(image_result)
            
            # 打印每张图片的详细IoU结果
            print(f"  Detailed IoU for {img_path.name}:")
            for idx, class_name in enumerate(class_names):
                if idx < len(ious):  # 防止索引超出范围
                    iou_val = ious[idx]
                    if not np.isnan(iou_val):
                        print(f"    {class_name}: {iou_val:.3f}")
                    else:
                        print(f"    {class_name}: NaN")
            print(f"  Mean IoU: {float(np.nanmean(ious)):.3f}")
            
            # 累积每个类别的IoU
            for cls_idx, iou in enumerate(ious):
                if not np.isnan(iou) and cls_idx < model.num_cls:
                    total_ious[cls_idx].append(iou)
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 删除临时pickle文件
    if temp_pkl_path and os.path.exists(temp_pkl_path):
        os.remove(temp_pkl_path)
    
    # 计算各类别平均IoU
    class_avg_ious = {}
    for cls_idx, iou_list in total_ious.items():
        if iou_list:
            class_avg_ious[f"class_{cls_idx}"] = np.mean(iou_list)
        else:
            class_avg_ious[f"class_{cls_idx}"] = float('nan')
    
    overall_mean_iou = np.nanmean([result['mean_iou'] for result in per_image_ious])
    
    evaluation_result = {
        'dataset': dataset_name,
        'prompt_type': 'expanded' if use_expanded_prompts else 'original',
        'overall_mean_iou': float(overall_mean_iou),
        'per_class_avg_iou': class_avg_ious,
        'per_image_results': per_image_ious
    }
    
    return evaluation_result


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation of segmentation models')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for computation')
    parser.add_argument('--dataset', type=str, choices=['Vaihingen', 'Potsdam', 'LoveDA', 'iSAID', 'all'], 
                        default='all', help='Dataset to evaluate (default: all)')
    parser.add_argument('--ratio', type=float, default=1.0, help='Ratio of dataset to process (default: 1.0)')
    args = parser.parse_args()
    
    # 检查CUDA设备
    if not torch.cuda.is_available():
        print("CUDA is not available, exiting...")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据集配置
    datasets_config = [
        {
            'img_dir': '/data/public/Vaihingen/img_dir/val',
            'gt_dir': '/data/public/Vaihingen/ann_dir/val',
            'name': 'Vaihingen'
        },
        {
            'img_dir': '/data/public/Potsdam/img_dir/val',
            'gt_dir': '/data/public/Potsdam/ann_dir/val',
            'name': 'Potsdam'
        },
        {
            'img_dir': '/data/public/LoveDA/img_dir/val',
            'gt_dir': '/data/public/LoveDA/ann_dir/val',
            'name': 'LoveDA'
        },
        {
            'img_dir': '/data/public/iSAID/img_dir/val',
            'gt_dir': '/data/public/iSAID/ann_dir/val',
            'name': 'iSAID'
        }
    ]
    
    # 如果指定了特定数据集，只处理那个数据集
    if args.dataset != 'all':
        datasets_config = [ds for ds in datasets_config if ds['name'] == args.dataset]
    
    all_results = []
    
    # 对每个数据集运行原始和扩展提示词评估
    for dataset_config in datasets_config:
        for use_expanded in [False, True]:
            result = evaluate_dataset(
                img_dir=dataset_config['img_dir'],
                gt_dir=dataset_config['gt_dir'],
                dataset_name=dataset_config['name'],
                device=args.device,
                use_expanded_prompts=use_expanded,
                ratio=args.ratio
            )
            all_results.append(result)
    
    # 保存结果到JSON文件
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Evaluation completed. Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()