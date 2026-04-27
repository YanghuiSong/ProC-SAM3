# Optimized SAM3 configuration for Vaihingen dataset with LogSumExp aggregation
_base_ = './base_config.py'

# Model configuration
model = dict(
    type='BasicSAM3Segmentation',
    classname_path='./configs/cls_vaihingen.txt',  # 使用标准的类别文件路径
    device='cuda:0',  # 使用cuda:0设备（修正注释中的错误）
    prob_thd=0.1,  # 增加概率阈值以减少噪声
    bg_idx=0,      # Background index
    slide_stride=0,  # Slide stride for sliding window inference
    slide_crop=0,    # Crop size for sliding window inference
    confidence_threshold=0.4,  # 降低置信度阈值以捕获更多细节
    use_sem_seg=True,          # Use semantic segmentation head
    use_presence_score=True,   # Use presence score
    use_transformer_decoder=False,  # 现在改为False，只使用语义输出
    temperature=0.5  # 使用较低的温度参数以获得更确定性的预测
)

# Dataset settings for Vaihingen
dataset_type = 'ISPRSDataset'  # 使用MMSegmentation中已注册的ISPRS数据集类型
data_root = '/data/public/Vaihingen'  # 使用标准数据根目录

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Data configuration
test_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,  # Vaihingen doesn't typically use zero as ignore label
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline
    )
)

# Evaluation configuration - 更新为支持AP计算的评估器
test_evaluator = dict(
    type='CombinedMetrics',  # 使用新的组合评估器
    iou_metrics=['mIoU'],    # 继续计算IoU指标
    iou_thresholds=[0.5, 0.75],  # AP50 and AP75
    ignore_index=255
)

# Test configuration
test_cfg = dict(type='TestLoop')

# Training configuration (even though we're evaluating)
train_cfg = None
optim_wrapper = None
param_scheduler = None