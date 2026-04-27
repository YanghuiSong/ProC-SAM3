class_names = [
    'imprev',
    'building',
    'low_vegetation',
    'tree',
    'car',
    'clutter',
]
data = dict(
    samples_per_gpu=4,
    val=dict(
        ann_dir='ann_dir/val',
        data_root='/data/public/Potsdam',
        img_dir='img_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PotsdamDataset'),
    workers_per_gpu=8)
data_root = '/data/public/Potsdam'
dataset_type = 'PotsdamDataset'
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
    bg_idx=5,
    classname_path='./configs/cls_potsdam.txt',
    confidence_threshold=0.4,
    device='cuda:3',
    enable_expanded_prompt=True,
    expanded_prompt_pool_path=
    './prompt_pools/cfg_potsdam_controlled_expanded_prompt_pool_preset.pkl',
    model_type='SAM3',
    optimize_method='preset',
    prob_thd=0.1,
    slide_crop=0,
    slide_stride=0,
    type='SegEarthOV3Segmentation',
    use_presence_score=True,
    use_sem_seg=True,
    use_transformer_decoder=True)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/data/public/Potsdam',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PotsdamDataset'),
    num_workers=8,
    persistent_workers=True,
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
work_dir = './work_dirs/cfg_potsdam_controlled'
