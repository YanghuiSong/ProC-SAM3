import numpy as np
import cv2
import os
import torch

# 1. 集中存储所有数据集的调色板信息 (RGB格式)
_DATASET_METAINFO = {
    'OpenEarthMapDataset': {
        'classes': ('background', 'bareland', 'grass', 'pavement', 'road', 'tree', 'water', 'cropland', 'building'),
        'palette': [[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7]]
    },
    'UAVidDataset': {
        'classes': ('background', 'building', 'road', 'car', 'tree', 'vegetation', 'human'),
        'palette': [[0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192], [0, 128, 0], [128, 128, 0], [64, 64, 0]]
    },
    'UDD5Dataset': {
        'classes': ('vegetation', 'building', 'road', 'vehicle', 'other'),
        'palette': [[107, 142, 35], [102, 102, 156], [128, 64, 128], [0, 0, 142], [0, 0, 0]]
    },
    'VDDDataset': {
        'classes': ('other', 'wall', 'road', 'vegetation', 'vehicle', 'roof', 'water'),
        'palette': [[0, 0, 0], [102, 102, 156], [128, 64, 128], [107, 142, 35], [0, 0, 142], [70, 70, 70], [0, 69, 255]]
    },
    'iSAIDDataset': {
        'classes': ('background', 'ship', 'store tank', 'baseball diamond', 'tennis court', 'basketball court', 'Ground Track Field', 'Bridge', 
                    'Large_Vehicle', 'Small Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout', 'Soccer ball field', 'plane', 'Harbor'),
        'palette': [[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7], 
                    [128, 64, 128], [192, 0, 192], [0, 128, 0], [128, 128, 0], [107, 142, 35], [0, 0, 142], [70, 70, 70]]
    },
    'LoveDADataset': {
        'classes': ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'),
        'palette': [[0, 0, 0], [222, 31, 7], [255, 255, 255], [0, 69, 255], [128, 0, 0], [34, 97, 38],[75, 181, 73]]
    },
    'PotsdamDataset': {
        'classes': ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'),
        'palette': [[255, 255, 255], [222, 31, 7], [0, 255, 36], [34, 97, 38], [192, 0, 192],[0, 0, 0]]
    },
    'ISPRSDataset': { # Vaihingen dataset
        'classes': ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'),
        'palette': [[255, 255, 255], [222, 31, 7], [0, 255, 36], [34, 97, 38], [192, 0, 192], [0, 0, 0]]
    },
    # Single class
    'WHUDataset': {
        'classes': ('background', 'building'),
        'palette': [[146, 208, 234], [222, 31, 7]]
    },
    'InriaDataset': {
        'classes': ('background', 'building'),
        'palette': [[146, 208, 234], [222, 31, 7]]
    },
    'xBDDataset': {
        'classes': ('background', 'building'),
        'palette': [[146, 208, 234], [222, 31, 7]]
    },
    'CHN6_CUGDataset': {
        'classes': ('background', 'road'),
        'palette': [[248, 222, 215], [89, 233, 133]]
    },
    'RoadValDataset': {
        'classes': ('background', 'road'),
        'palette': [[248, 222, 215], [89, 233, 133]]
    },
    'WaterDataset': {
        'classes': ('background', 'water'),
        'palette': [[0, 0, 0], [0, 69, 255]]
    }
}

def get_palette(dataset_type: str):
    """
    根据数据集类型获取固定的调色板 (BGR格式)。

    Args:
        dataset_type (str): 数据集的名称。

    Returns:
        np.ndarray: OpenCV友好的BGR格式调色板, 形状为 (num_classes, 3)。
                    如果找不到数据集，则返回一个默认的随机调色板。
    """
    if dataset_type in _DATASET_METAINFO:
        # 从元信息字典中提取 'palette'
        palette_rgb = _DATASET_METAINFO[dataset_type]['palette']
        # 转换为Numpy数组并从RGB转为BGR
        palette_bgr = np.array(palette_rgb, dtype=np.uint8)[:, ::-1]
        return palette_bgr
    else:
        print(f"警告: 未找到数据集 '{dataset_type}' 的预定义调色板，将使用默认调色板。")
        np.random.seed(42)
        return np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

def get_classes(dataset_type: str):
    """
    根据数据集类型获取类别名称元组。

    Args:
        dataset_type (str): 数据集的名称。

    Returns:
        tuple[str] or None: 类别名称的元组。如果找不到数据集，则返回 None。
    """
    if dataset_type in _DATASET_METAINFO:
        return _DATASET_METAINFO[dataset_type]['classes']
    else:
        print(f"警告: 未找到数据集 '{dataset_type}' 的类别名称。")
        return None
    
def _create_overlay_image(mask_np, original_img, palette_bgr, overlay_alpha, contour_thickness):
    """辅助函数：根据输入的掩码生成叠加图像。"""
    h, w, _ = original_img.shape
    
    # 1. 创建彩色掩码
    color_mask = np.zeros_like(original_img, dtype=np.uint8)
    unique_classes = np.unique(mask_np)
    for class_id in unique_classes:
        if class_id >= len(palette_bgr):
            # 警告信息在主函数中打印，这里直接跳过
            continue
        color_mask[mask_np == class_id] = palette_bgr[class_id]

    # 2. 将彩色掩码与原始图像混合
    overlay_img = cv2.addWeighted(original_img, 1 - overlay_alpha, color_mask, overlay_alpha, 0)

    # 3. 绘制轮廓
    if contour_thickness > 0:
        for class_id in unique_classes:
            if class_id >= len(palette_bgr):
                continue
            
            class_mask = (mask_np == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour_color = tuple(palette_bgr[class_id].tolist())
            cv2.drawContours(overlay_img, contours, -1, contour_color, contour_thickness)
            
    return overlay_img

def visualize_segmentation(dataset_type, data_sample, seg_pred, visualize_gt=False):
    """
    可视化分割结果，并可选择性地可视化真实标签(GT)。

    Args:
        dataset_type (str): 数据集类型，用于获取调色板。
        data_sample: 包含元数据(img_path)和真实标签(gt_sem_seg)的对象。
        seg_pred (torch.Tensor): 预测的分割图。
        visualize_gt (bool): 是否也对Ground Truth进行可视化。默认为 True。
    """
    # --- 保存路径 ---
    save_dir = f'visual_{dataset_type}'

    # --- 可视化参数 ---
    overlay_alpha = 0.5
    contour_thickness = 0  # 一定要是整数，取值范围，-1表示不绘制轮廓线并且填充内部，0,1,2,3分别表示绘制由细到粗的轮廓线
    # --------------------

    # 1. 公共数据准备
    original_img = cv2.imread(data_sample.img_path)
    if original_img is None:
        print(f"警告: 无法读取图像 {data_sample.img_path}，跳过可视化。")
        return

    # 以预测结果的尺寸为基准进行缩放
    h, w = seg_pred.squeeze().shape
    original_img_resized = cv2.resize(original_img, (w, h))

    # 获取调色板
    palette_bgr = get_palette(dataset_type)
    
    # 文件名准备
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(data_sample.img_path)
    file_name, _ = os.path.splitext(base_name)

    # 2. 可视化预测结果 (Prediction)
    pred_mask_np = seg_pred.squeeze().cpu().numpy().astype(np.uint8)
    
    # 调用辅助函数生成叠加图
    pred_overlay_img = _create_overlay_image(pred_mask_np, original_img_resized, palette_bgr, overlay_alpha, contour_thickness)
    
    # 保存预测结果图
    pred_save_path = os.path.join(save_dir, f'{file_name}_pred.png')
    cv2.imwrite(pred_save_path, pred_overlay_img)
    # print(f"预测可视化结果已保存至: {pred_save_path}")

    # 3. (可选) 可视化真实标签 (Ground Truth)
    if visualize_gt:
        # 检查 data_sample 中是否存在 gt_sem_seg
        if not hasattr(data_sample, 'gt_sem_seg') or data_sample.gt_sem_seg is None:
            print(f"警告: 'data_sample' 中缺少 'gt_sem_seg'，无法可视化GT。")
            return
            
        gt_mask_tensor = data_sample.gt_sem_seg.data
        gt_mask_np = gt_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        
        # 确保GT和原图尺寸一致 (通常是一致的，但以防万一)
        if gt_mask_np.shape != (h, w):
            gt_mask_np = cv2.resize(gt_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # 调用同一个辅助函数生成GT的叠加图
        gt_overlay_img = _create_overlay_image(gt_mask_np, original_img_resized, palette_bgr, overlay_alpha, contour_thickness)

        # 保存GT结果图
        gt_save_path = os.path.join(save_dir, f'{file_name}_GT.png')
        cv2.imwrite(gt_save_path, gt_overlay_img)
        # print(f"GT 可视化结果已保存至: {gt_save_path}")
