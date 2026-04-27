class_names = [
    'vegetation',
    'building',
    'road',
    'vehicle',
    'background',
]
data = dict(
    samples_per_gpu=1,
    val=dict(
        ann_dir='val/gt',
        data_root='/data/public/UDD/UDD5',
        img_dir='val/src',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='UDD5Dataset'),
    workers_per_gpu=4)
data_root = '/data/public/UDD/UDD5'
dataset_type = 'UDD5Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    bg_idx=4,
    classname_path='./configs/cls_udd5.txt',
    confidence_threshold=0.5,
    enable_expanded_prompt=True,
    expanded_prompt_pool_path=
    './prompt_pools/cfg_udd5_controlled_expanded_prompt_pool_hybrid_strict.pkl',
    model_type='SAM3',
    optimize_method='hybrid_strict',
    prob_thd=0.1,
    type='SegEarthOV3Segmentation',
    use_presence_score=True,
    use_sem_seg=True,
    use_transformer_decoder=True)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='val/src', seg_map_path='val/gt'),
        data_root='/data/public/UDD/UDD5',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='UDD5Dataset'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ignore_index=255,
    iou_metrics=[
        'mIoU',
    ],
    iou_thresholds=[
        0.5,
        0.75,
    ],
    type='CombinedMetrics')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    alpha=0.7,
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/cfg_udd5_controlled'
