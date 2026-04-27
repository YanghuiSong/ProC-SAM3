import os
import cv2
import numpy as np
from PIL import Image
import argparse
from palettes import get_palette, _DATASET_METAINFO


def visualize_gt_dataset(dataset_type, gt_dir, output_dir, sample_rate=1.0):
    """
    可视化指定数据集的所有GT标签文件
    
    Args:
        dataset_type (str): 数据集类型
        gt_dir (str): GT标签所在目录
        output_dir (str): 输出目录
        sample_rate (float): 采样率，默认为1.0（即处理全部数据）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取调色板 (RGB格式)
    palette_rgb = _DATASET_METAINFO[dataset_type]['palette']
    palette_rgb = np.array(palette_rgb, dtype=np.uint8)
    
    print(f"开始处理 {dataset_type} 数据集，GT目录: {gt_dir}")
    
    # 获取所有png格式的GT文件
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
    
    if not gt_files:
        print(f"在 {gt_dir} 中未找到任何png格式的标签文件")
        return
    
    print(f"找到 {len(gt_files)} 个GT标签文件")
    
    # 根据采样率确定要处理的文件列表
    if sample_rate < 1.0:
        n_files_to_process = int(len(gt_files) * sample_rate)
        gt_files = gt_files[:n_files_to_process]
        print(f"根据采样率 {sample_rate}，将处理前 {n_files_to_process} 个文件")
    
    for i, gt_file in enumerate(gt_files):
        if i % max(1, int(10 / sample_rate)) == 0:  # 调整进度输出频率
            print(f"正在处理第 {i+1}/{len(gt_files)} 个文件: {gt_file}")
        
        gt_path = os.path.join(gt_dir, gt_file)
        
        # 读取标签图像
        try:
            # 使用PIL读取，保持标签的整数值不变
            label_img = np.array(Image.open(gt_path))
        except Exception as e:
            print(f"读取文件 {gt_path} 时出错: {e}")
            continue
        
        # 确保标签是二维的
        if label_img.ndim == 3:
            # 如果是三维的，取第一个通道
            label_img = label_img[:, :, 0]
        
        # 创建彩色图像
        h, w = label_img.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 为每个类别分配颜色
        unique_labels = np.unique(label_img)
        print(f"发现的唯一标签值: {sorted(unique_labels)}, 预期类别数: {len(palette_rgb)}")
        print(f"预定义类别: {_DATASET_METAINFO[dataset_type]['classes']}")
        
        for label in unique_labels:
            # 检查标签值是否从1开始，如果是，则减1来匹配色盘索引
            adjusted_label = label - 1 if label > 0 else label
            if 0 <= adjusted_label < len(palette_rgb):
                color_img[label_img == label] = palette_rgb[adjusted_label]
            elif label == 255:  # 特殊处理忽略标签
                # 通常255代表忽略的像素，在遥感数据中也可能有其他特殊含义
                black_color = np.array([0, 0, 0], dtype=np.uint8)  # 黑色
                color_img[label_img == label] = black_color
            else:
                # 如果标签超出了预定义的类别数量，使用黑色
                print(f"警告: 标签 {label} (调整后: {adjusted_label}) 超出 {dataset_type} 的预定义类别数量 ({len(palette_rgb)})，将显示为黑色")
                print(f"  - 该标签在图像中出现 {np.sum(label_img == label)} 次")
                black_color = np.array([0, 0, 0], dtype=np.uint8)  # 黑色
                color_img[label_img == label] = black_color
        
        # 保存可视化结果，使用与原始标签文件相同的命名
        output_path = os.path.join(output_dir, gt_file.replace('.png', '_gt_vis.png'))
        cv2.imwrite(output_path, color_img[:, :, ::-1])  # BGR to RGB conversion for cv2
    
    print(f"{dataset_type} 数据集可视化完成，结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化数据集的Ground Truth标签')
    parser.add_argument('--datasets', nargs='+', type=str, 
                        choices=['iSAID', 'LoveDA', 'Potsdam', 'Vaihingen'],
                        default=['iSAID', 'LoveDA', 'Potsdam', 'Vaihingen'],
                        help='要可视化的数据集列表')
    # 移除base-path参数，改为分别指定各数据集路径
    parser.add_argument('--isaid-path', type=str, default='E:\\iSAID',
                        help='iSAID数据集路径')
    parser.add_argument('--loveda-path', type=str, default='E:\\LoveDA',
                        help='LoveDA数据集路径')
    parser.add_argument('--potsdam-path', type=str, default='E:\\Potsdam',
                        help='Potsdam数据集路径')
    parser.add_argument('--vaihingen-path', type=str, default='E:\\Vaihingen',
                        help='Vaihingen数据集路径')
    
    args = parser.parse_args()
    
    # 定义数据集映射
    dataset_mapping = {
        'iSAID': {
            'type': 'iSAIDDataset',
            'dir': 'ann_dir/val',  # 相对路径部分
            'path': args.isaid_path,  # 完整路径
            'name': 'iSAID',
            'sample_rate': 0.1  # iSAID数据集只处理前10%
        },
        'LoveDA': {
            'type': 'LoveDADataset',
            'dir': 'ann_dir/val',  # 相对路径部分
            'path': args.loveda_path,  # 完整路径
            'name': 'LoveDA',
            'sample_rate': 1.0  # LoveDA数据集处理全部数据
        },
        'Potsdam': {
            'type': 'PotsdamDataset',
            'dir': 'ann_dir/val',  # 相对路径部分
            'path': args.potsdam_path,  # 完整路径
            'name': 'Potsdam',
            'sample_rate': 1.0  # Potsdam数据集处理全部数据
        },
        'Vaihingen': {
            'type': 'ISPRSDataset',  # Vaihingen dataset
            'dir': 'ann_dir/val',  # 相对路径部分
            'path': args.vaihingen_path,  # 完整路径
            'name': 'Vaihingen',
            'sample_rate': 1.0  # Vaihingen数据集处理全部数据
        }
    }
    
    for dataset_name in args.datasets:
        mapping = dataset_mapping[dataset_name]
        gt_dir = os.path.join(mapping['path'], mapping['dir'])  # 组合完整路径
        
        # 使用数据集路径作为根目录
        dataset_root = mapping['path']
        output_dir = os.path.join(dataset_root, 'vis_gt')  # 在数据集根目录下创建vis_gt文件夹
        
        if not os.path.exists(gt_dir):
            print(f"错误: 目录 {gt_dir} 不存在")
            continue
        
        # 使用相应的采样率
        sample_rate = mapping.get('sample_rate', 1.0)
        visualize_gt_dataset(mapping['type'], gt_dir, output_dir, sample_rate)


if __name__ == '__main__':
    main()