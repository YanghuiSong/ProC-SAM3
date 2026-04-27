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

from sam3_segmentor import SegEarthOV3Segmentation
from palettes import _DATASET_METAINFO


def visualize_patch_clustering(image_path, model, num_clusters=16, dataset_type="Vaihingen"):
    """
    使用 SAM3 的 image_encoder 提取 patch tokens，并进行聚类可视化
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取原始尺寸
    H, W = img.shape[:2]

    # 预处理：缩放到模型输入大小（如 1008x1008）
    input_size = 1008
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(model.device)  # BxCxHxW

    # 提取 patch tokens（通过 SAM3 模型的 backbone）
    with torch.no_grad():
        # 通过处理器访问底层模型的 backbone
        backbone_out = model.processor.model.backbone.forward_image(img_tensor)  # 获取 backbone 特征
        
        # 获取backbone特征 - 检查特征数量并获取特征
        backbone_features = backbone_out["backbone_fpn"]
        
        # 检查backbone_features是否为列表或元组
        if isinstance(backbone_features, (list, tuple)):
            # 如果是列表或元组，取最后一层特征
            features = backbone_features[-1]  # 获取最高层特征 [B, C, H, W]
        else:
            # 如果是字典或其他类型，尝试访问
            if hasattr(backbone_features, '__getitem__'):
                # 如果支持索引，尝试获取最后一个
                if len(backbone_features) > 0:
                    # 尝试获取最后一个元素
                    if isinstance(backbone_features, dict):
                        # 如果是字典，取最后一个值
                        features = list(backbone_features.values())[-1]
                    else:
                        # 如果是其他可索引类型，取最后一个
                        features = backbone_features[-1] if len(backbone_features) > 1 else backbone_features[0]
                else:
                    # 如果长度为0，直接使用
                    features = backbone_features
            else:
                # 如果不支持索引，直接使用
                features = backbone_features

    # 确保features是tensor且形状正确
    if not isinstance(features, torch.Tensor):
        # 如果仍然无法正确访问backbone特征，抛出错误以便调试
        print(f"Error: Expected tensor for features, got {type(features)}")
        print(f"Available keys in backbone_out: {list(backbone_out.keys())}")
        print(f"Type of backbone_fpn: {type(backbone_out['backbone_fpn'])}")
        if isinstance(backbone_out['backbone_fpn'], dict):
            print(f"Keys in backbone_fpn: {list(backbone_out['backbone_fpn'].keys())}")
        raise TypeError(f"Features is not a tensor: {type(features)}")

    # 提取 patch tokens（展平）
    B, C, H_feat, W_feat = features.shape
    patch_tokens = features.permute(0, 2, 3, 1).reshape(B * H_feat * W_feat, C)  # (N, C)

    # 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=3)
    cluster_labels = kmeans.fit_predict(patch_tokens.cpu().numpy())

    # 映射回图像空间
    cluster_map = cluster_labels.reshape(H_feat, W_feat)

    # 插值回原始图像尺寸
    cluster_map_resized = cv2.resize(cluster_map.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

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

    # 确保聚类数量不超过调色板大小
    effective_clusters = min(num_clusters, len(palette_rgb))
    palette_rgb = np.array(palette_rgb[:effective_clusters], dtype=np.uint8)

    # 创建彩色分割图
    h, w = cluster_map_resized.shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 为每个聚类分配颜色
    unique_labels = np.unique(cluster_map_resized)
    for label in unique_labels:
        # 检查color_idx是否在有效范围内
        color_idx = int(label % len(palette_rgb)) if len(palette_rgb) > 0 else 0
        if len(palette_rgb) > 0:
            vis_img[cluster_map_resized == label] = palette_rgb[color_idx]
        else:
            # 如果没有有效的调色板，使用随机颜色
            random_color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            vis_img[cluster_map_resized == label] = random_color

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


def process_improved_images():
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

    # 测试数据路径 - 使用Linux服务器路径
    base_path = "/data/users/syh/QwSAM3/QwSAM3TestData"
    
    # 创建结果保存目录
    output_dir = "./patch_cluster_visualizations_comparison"
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
                
                # 执行原始提示词的聚类可视化
                orig_result = visualize_patch_clustering(img_path, orig_model, num_clusters=16, dataset_type=dataset_name)
                
                # 执行扩展提示词的聚类可视化
                exp_result = visualize_patch_clustering(img_path, exp_model, num_clusters=16, dataset_type=dataset_name)
                
                # 读取原始图像用于对比
                orig_img = cv2.imread(img_path)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                # 创建对比图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(orig_img)
                axes[0].set_title(f'Original Image - {dataset_name}')
                axes[0].axis('off')
                
                axes[1].imshow(orig_result)
                axes[1].set_title('Original Prompts Clustering')
                axes[1].axis('off')
                
                axes[2].imshow(exp_result)
                axes[2].set_title('Extended Prompts Clustering')
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


if __name__ == "__main__":
    process_improved_images()