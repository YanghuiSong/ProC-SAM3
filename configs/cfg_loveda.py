# base config
_base_ = ['./base_config.py']

# Dataset
data_root = '/data/public/LoveDA'
dataset_type = 'LoveDADataset'

# Class names for LoveDA dataset
class_names = [
    'background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'
]

# Model
model = dict(
    type='SegEarthOV3Segmentation',
    classname_path='./configs/cls_loveda.txt',
    model_type='SAM3',
    prob_thd=0.1,
    bg_idx=0,
    confidence_threshold=0.4,
    use_sem_seg=True,
    use_presence_score=True,
    use_transformer_decoder=True,
    slide_stride=0,
    device='cuda:1',
    slide_crop=0
)

# Data
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ])
)

# Evaluation
test_evaluator = dict(
    type='CombinedMetrics',  # 使用新的组合评估器
    iou_metrics=['mIoU'],
    iou_thresholds=[0.5, 0.75],  # AP50 and AP75
    ignore_index=255
)

# Test dataloader
test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]
    )
)

# Test configuration
test_cfg = dict(type='TestLoop')
