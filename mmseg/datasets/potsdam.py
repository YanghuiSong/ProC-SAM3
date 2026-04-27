# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PotsdamDataset(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.png',  # 修改为正确的图像后缀.png
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_annotations(self, data_root, data_prefix, ann_file=None, split=None, metainfo=None, **kwargs):
        """重写加载注释方法，处理RGB标签到类别索引的转换"""
        # 调用父类方法加载基本注释
        data_list = super().load_annotations(data_root, data_prefix, ann_file, split, metainfo, **kwargs)
        return data_list

    def convert_rgb_to_class_idxs(self, rgb_label):
        """将RGB标签图转换为类别索引图"""
        # 获取调色板信息
        palette = np.array(self.METAINFO['palette'])
        
        # 将RGB标签重塑为(H*W, 3)
        h, w, c = rgb_label.shape
        rgb_flat = rgb_label.reshape(-1, 3)
        
        # 计算每个像素与调色板中每个颜色的距离
        distances = np.linalg.norm(rgb_flat[:, None, :] - palette[None, :, :], axis=2)
        
        # 找到最近的颜色索引
        class_idxs = np.argmin(distances, axis=1)
        
        # 重塑回原始形状
        class_idxs = class_idxs.reshape(h, w)
        
        return class_idxs

    def get_gt_seg_map_by_idx(self, index):
        """获取GT分割图，如果需要则进行RGB到类别的转换"""
        # 获取标签路径
        seg_map_path = self.data_list[index].seg_map_path
        
        # 加载标签图像
        try:
            gt_seg_map = mmcv.imread(seg_map_path, flag='color', backend='pillow')
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation map from {seg_map_path}: {e}")
        
        # 如果标签图像是RGB格式，则将其转换为类别索引图
        if len(gt_seg_map.shape) == 3:
            gt_seg_map = self.convert_rgb_to_class_idxs(gt_seg_map)
        
        # 确保输出是2D张量
        if len(gt_seg_map.shape) != 2:
            raise ValueError(f"Expected 2D segmentation map, got {gt_seg_map.shape}")
        
        # 调试信息：输出标签图的形状和类型
        print(f"Loaded segmentation map shape: {gt_seg_map.shape}, dtype: {gt_seg_map.dtype}")
        
        return gt_seg_map