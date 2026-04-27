checkpoint_config = dict(by_epoch=True, interval=50)
class_names = [
    'background',
    'ship',
    'store_tank',
    'baseball_diamond',
    'tennis_court',
    'basketball_court',
    'ground_track_field',
    'bridge',
    'large_vehicle',
    'small_vehicle',
    'helicopter',
    'swimming_pool',
    'roundabout',
    'soccer_ball_field',
    'plane',
    'harbor',
]
data = dict(
    samples_per_gpu=4,
    test=dict(
        ann_dir='ann_dir/val',
        data_root='/data/public/iSAID',
        img_dir='img_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='iSAIDDataset'),
    train=dict(
        ann_dir='ann_dir/train',
        data_root='/data/public/iSAID',
        img_dir='img_dir/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='iSAIDDataset'),
    val=dict(
        ann_dir='ann_dir/val',
        data_root='/data/public/iSAID',
        img_dir='img_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='iSAIDDataset'),
    workers_per_gpu=8)
data_root = '/data/public/iSAID'
dataset_type = 'iSAIDDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
dist_params = dict(backend='nccl')
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_config = dict(
    hooks=[
        dict(by_epoch=True, type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    bg_idx=0,
    classname_path='./configs/cls_iSAID.txt',
    confidence_threshold=0.4,
    expanded_prompt_pool_path=
    './prompt_pools/cfg_iSAID_expanded_prompt_pool_guided.pkl',
    model_type='SAM3',
    prob_thd=0.1,
    slide_crop=0,
    slide_stride=0,
    type='SegEarthOV3Segmentation',
    use_presence_score=True,
    use_sem_seg=True,
    use_transformer_decoder=True)
resume = False
resume_from = None
runner = dict(max_epochs=200, type='EpochBasedRunner')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/data/public/iSAID',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='iSAIDDataset'),
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
work_dir = './work_dirs/cfg_iSAID'
workflow = [
    (
        'test',
        1,
    ),
]
