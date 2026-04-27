# config for Vaihingen dataset with cached text embeddings
_base_ = [
    '../mmseg/datasets/vaihingen.py',
    '../mmseg/models/deeplabv3plus_r50-d8.py',
    '../mmseg/default_runtime.py',
    '../mmseg/schedules/schedule_40k.py'
]

# dataset settings
dataset_type = 'ISPRSDataset'
data_root = 'data/vaihingen'

# Modify data pipeline for SAM3
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train',
            seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(test_mode=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# model settings - Using the new cached segmentation model
model = dict(
    type='CachedSAM3OpenSegmentor',
    classname_path='./configs/cls_vaihingen.txt',
    device='cuda:0',
    prob_thd=0.0,
    bg_idx=0,
    slide_stride=0,
    slide_crop=0,
    confidence_threshold=0.5,
    use_sem_seg=True,
    use_presence_score=True,
    use_transformer_decoder=True,
    expanded_prompt_pool_path=None,  # Will be set by eval script if needed
    optimize_method='guided',
    cache_text_embeddings=True,  # Enable text embedding caching
    enable_expanded_prompt=False,  # Set to True if you want to use expanded prompts
    model_type='CachedSAM3OpenSegmentor'
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]

# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# default runtime
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
