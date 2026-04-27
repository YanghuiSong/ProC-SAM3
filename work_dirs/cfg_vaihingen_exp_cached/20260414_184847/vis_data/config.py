class_names = [
    'impervious_surface',
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
        data_root='/data/public/Vaihingen',
        img_dir='img_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ISPRSDataset'),
    workers_per_gpu=8)
data_root = '/data/public/Vaihingen'
dataset_type = 'ISPRSDataset'
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
    auto_tune_slot_batch=False,
    bg_idx=5,
    cache_text_embeddings=True,
    classname_path='./configs/cls_vaihingen_exp.txt',
    compile_model=False,
    confidence_threshold=0.4,
    enable_expanded_prompt=False,
    execution_mode='per_image',
    expanded_prompt_pool_path=None,
    group_images_by_size=True,
    image_batch_size=4,
    inference_dtype='bf16',
    mask_query_chunk_size=32,
    max_cross_image_slots=15,
    max_mask_tensor_mb=1024,
    model_type='SAM3',
    optimize_method='guided',
    prob_thd=0.1,
    processor_resolution=1008,
    prompt_batch_size=15,
    router_coarse_resolution=512,
    router_enabled=True,
    router_global_topk=4,
    router_inference_dtype='bf16',
    router_keep_per_class=1,
    slide_crop=0,
    slide_stride=0,
    slot_batch_size=0,
    slot_chunk_size=4,
    type='CachedSegEarthOV3Segmentation',
    use_presence_score=True,
    use_sem_seg=True,
    use_transformer_decoder=True)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/data/public/Vaihingen',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ISPRSDataset'),
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
work_dir = './work_dirs/cfg_vaihingen_exp_cached'
