# config for Vaihingen dataset with cached text embeddings - full evaluation
_base_ = ['./base_config.py']

# Dataset
data_root = 'data/vaihingen'  # Ensure this path is correct for your environment
dataset_type = 'ISPRSDataset'

# Class names for Vaihingen dataset (for reference, actual classes loaded from classname_path)
class_names = [
    'impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'
]

# Model - Using the new cached segmentation model
model = dict(
    type='CachedSAM3OpenSegmentor',
    classname_path='./configs/cls_vaihingen.txt',
    model_type='CachedSAM3OpenSegmentor',
    prob_thd=0.1,
    bg_idx=5,
    confidence_threshold=0.4,
    use_sem_seg=True,
    use_presence_score=True,
    use_transformer_decoder=True,
    slide_stride=0,
    slide_crop=0,
    cache_text_embeddings=True,  # Enable text embedding caching
    enable_expanded_prompt=True,  # Enable expanded prompts
    expanded_prompt_pool_path=None,  # Will be set by eval script if needed
    optimize_method='guided'
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
