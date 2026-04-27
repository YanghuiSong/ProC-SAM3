import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tempfile
from pathlib import Path
import pickle
import argparse

from sam3_segmentor import SegEarthOV3Segmentation
from palettes import _DATASET_METAINFO
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample


def visualize_segmentation_result(image_path, model, dataset_type="Vaihingen"):
    """
    使用模型进行分割推理并可视化结果
    """
    # 读取图像
    pil_img = Image.open(image_path).convert('RGB')
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取原始尺寸
    H, W = img.shape[:2]

    # 准备数据样本
    data_sample = SegDataSample()
    img_meta = {
        'img_path': image_path,
        'ori_shape': (H, W)  # (height, width)
    }
    data_sample.set_metainfo(img_meta)

    # 进行分割推理
    seg_result = model.predict([pil_img], [data_sample])
    seg_pred = seg_result[0].pred_sem_seg.data.cpu().numpy().squeeze()
    
    # 调整分割结果尺寸到原始图像尺寸
    if seg_pred.shape[0] != H or seg_pred.shape[1] != W:
        seg_pred_resized = cv2.resize(seg_pred.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        seg_pred_resized = seg_pred

    # 获取数据集的官方配色方案 (RGB格式)
    if dataset_type == "iSAID":
        palette_rgb = _DATASET_METAINFO['iSAIDDataset']['palette']
    elif dataset_type == "LoveDA":
        palette_rgb = _DATASET_METAINFO['LoveDADataset']['palette']
    elif dataset_type == "Potsdam":
        palette_rgb = _DATASET_METAINFO['PotsdamDataset']['palette']
    elif dataset_type == "Vaihingen":
        palette_rgb = _DATASET_METAINFO['ISPRSDataset']['palette']  # Vaihingen dataset
    else:
        # 默认使用iSAID的配色
        palette_rgb = _DATASET_METAINFO['iSAIDDataset']['palette']

    # 确保分割类别不超过调色板大小
    max_class_idx = min(seg_pred_resized.max(), len(palette_rgb)-1)
    palette_rgb = np.array(palette_rgb[:max_class_idx+1], dtype=np.uint8)

    # 创建彩色分割图
    h, w = seg_pred_resized.shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 为每个类别分配颜色
    unique_labels = np.unique(seg_pred_resized)
    for label in unique_labels:
        # 检查标签值是否在有效范围内
        if 0 <= label < len(palette_rgb):
            vis_img[seg_pred_resized == label] = palette_rgb[int(label)]
        else:
            # 将超出范围的标签映射为黑色
            black_color = np.array([0, 0, 0], dtype=np.uint8)  # 黑色
            vis_img[seg_pred_resized == label] = black_color

    return vis_img


def create_model_with_class_file(class_file, dataset_name):
    """创建使用指定类别文件的模型实例"""
    # 为当前数据集确定扩展提示池路径
    expanded_prompt_pool_path = None
    if "_exp.txt" in class_file:  # 如果是扩展提示词文件
        # 根据数据集类型确定扩展提示词文件
        if dataset_name == "iSAID":
            exp_class_file = "./configs/cls_iSAID_exp.txt"
        elif dataset_name == "LoveDA":
            exp_class_file = "./configs/cls_loveda_exp.txt"
        elif dataset_name == "Potsdam":
            exp_class_file = "./configs/cls_potsdam_exp.txt"
        elif dataset_name == "Vaihingen":
            exp_class_file = "./configs/cls_vaihingen_exp.txt"
        else:
            exp_class_file = "./configs/cls_iSAID_exp.txt"  # 默认
        
        # 创建临时的扩展提示池pickle文件
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_pkl_path = tmp_file.name
            
            # 从文本文件创建扩展提示池
            expanded_prompt_pool = {}
            with open(exp_class_file, 'r', encoding='utf-8') as f:
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
            
            expanded_prompt_pool_path = temp_pkl_path
    
    model = SegEarthOV3Segmentation(
        classname_path=class_file,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        prob_thd=0.1,
        confidence_threshold=0.4,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        expanded_prompt_pool_path=expanded_prompt_pool_path
    )
    
    return model, expanded_prompt_pool_path


def process_improved_images(target_dataset=None):
    """
    处理所有改进最显著的图像，包括原始词汇和扩展词汇的对比
    """
    # 定义数据集和对应的图像路径
    improved_images = {
        "Vaihingen": [
            "area38_1536_2038_2048_2550.png",
            "area24_512_2034_1024_2546.png",
            "area38_1024_512_1536_1024.png",
            "area16_512_1536_1024_2048.png",
            "area4_512_1536_1024_2048.png"
        ],
        "Potsdam": [
            "2_13_0_2560_512_3072.png",
            "2_13_2048_0_2560_512.png",
            "3_14_5120_4608_5632_5120.png",
            "2_14_0_2048_512_2560.png",
            "2_13_1024_4608_1536_5120.png"
        ],
        "LoveDA": [
            "2996.png",
            "2581.png",
            "2583.png",
            "2846.png",
            "2633.png"
        ],
        "iSAID": [
            "P1604_0_896_1024_1920.png",
            "P1242_1536_2432_1536_2432.png",
            "P0837_1824_2720_0_896.png",
            "P1179_1536_2432_3584_4480.png",
            "P2645_2048_2944_3584_4480.png"
        ]
    }

    # 如果指定了目标数据集，则只处理该数据集
    if target_dataset:
        if target_dataset not in improved_images:
            print(f"Error: Dataset '{target_dataset}' not recognized. Available datasets: {list(improved_images.keys())}")
            return
        improved_images = {target_dataset: improved_images[target_dataset]}
    
    # 测试数据路径 - 使用Linux服务器路径
    base_path = "/data/users/syh/QwSAM3/QwSAM3TestData"
    
    # 创建结果保存目录
    output_dir = "./segmentation_visualizations_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # 为每个数据集初始化模型
    for dataset_name, image_list in improved_images.items():
        print(f"Processing {dataset_name} dataset...")
        
        # 为当前数据集选择原始和扩展提示词文件
        if dataset_name == "iSAID":
            orig_class_file = "./configs/cls_iSAID.txt"
            exp_class_file = "./configs/cls_iSAID_exp.txt"
        elif dataset_name == "LoveDA":
            orig_class_file = "./configs/cls_loveda.txt"
            exp_class_file = "./configs/cls_loveda_exp.txt"
        elif dataset_name == "Potsdam":
            orig_class_file = "./configs/cls_potsdam.txt"
            exp_class_file = "./configs/cls_potsdam_exp.txt"
        elif dataset_name == "Vaihingen":
            orig_class_file = "./configs/cls_vaihingen.txt"
            exp_class_file = "./configs/cls_vaihingen_exp.txt"
        else:
            orig_class_file = "./configs/cls_iSAID.txt"  # 默认使用iSAID原始提示词
            exp_class_file = "./configs/cls_iSAID_exp.txt"  # 默认使用iSAID扩展提示词
        
        # 处理当前数据集的所有图像
        for img_name in image_list:
            try:
                img_path = os.path.join(base_path, dataset_name, img_name)
                
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}, skipping...")
                    continue
                
                print(f"Processing {img_name}...")
                
                # 创建使用原始提示词的模型
                orig_model, _ = create_model_with_class_file(orig_class_file, dataset_name)
                
                # 创建使用扩展提示词的模型
                exp_model, exp_pool_path = create_model_with_class_file(exp_class_file, dataset_name)
                
                # 执行原始提示词的分割可视化
                orig_result = visualize_segmentation_result(img_path, orig_model, dataset_type=dataset_name)
                
                # 执行扩展提示词的分割可视化
                exp_result = visualize_segmentation_result(img_path, exp_model, dataset_type=dataset_name)
                
                # 读取原始图像用于对比
                orig_img = cv2.imread(img_path)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                # 创建对比图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(orig_img)
                axes[0].set_title(f'Original Image - {dataset_name}')
                axes[0].axis('off')
                
                axes[1].imshow(orig_result)
                axes[1].set_title('Original Prompts Segmentation')
                axes[1].axis('off')
                
                axes[2].imshow(exp_result)
                axes[2].set_title('Extended Prompts Segmentation')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # 保存对比图
                output_path = os.path.join(output_dir, f"{dataset_name}_{img_name.replace('.png', '_comparison.png')}")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved comparison visualization to {output_path}")
                
                # 清理扩展提示池文件（如果有的话）
                if exp_pool_path and os.path.exists(exp_pool_path):
                    os.remove(exp_pool_path)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈跟踪
                continue

        print(f"Completed processing {dataset_name} dataset\n")


def main():
    parser = argparse.ArgumentParser(description='可视化数据集的分割结果')
    parser.add_argument('--dataset', type=str, 
                        choices=['iSAID', 'LoveDA', 'Potsdam', 'Vaihingen'],
                        help='要可视化的数据集名称，如果不指定则处理所有数据集')
    
    args = parser.parse_args()
    
    process_improved_images(target_dataset=args.dataset)


if __name__ == "__main__":
    main()