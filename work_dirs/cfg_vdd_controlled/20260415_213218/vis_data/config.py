class_names = [
    'background',
    'facade',
    'road',
    'vegetation',
    'vehicle',
    'roof',
    'water',
]
data = dict(
    samples_per_gpu=1,
    val=dict(
        ann_dir='test/gt',
        data_root='/data/public/VDD',
        img_dir='test/src',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='VDDDataset'),
    workers_per_gpu=4)
data_root = '/data/public/VDD'
dataset_type = 'VDDDataset'
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
    bg_idx=0,
    cache_text_embeddings=True,
    ccpea_consensus_weight=0.3,
    ccpea_instance_weight=0.3,
    ccpea_pool_size=8,
    ccpea_presence_weight=0.7,
    ccpea_prompt_fusion_alpha=0.6,
    ccpea_topk=2,
    ccpea_uncertainty_weight=0.5,
    class_aggregation='max',
    class_aggregation_consensus_beta=8.0,
    class_aggregation_gem_power=6.0,
    class_aggregation_reliability_alpha=0.7,
    class_aggregation_temperature=0.4,
    class_aggregation_topk=2,
    classname_path='./configs/cls_vdd.txt',
    compile_model=False,
    confidence_threshold=0.5,
    enable_expanded_prompt=True,
    execution_mode='per_image',
    expanded_prompt_pool_path=
    './prompt_pools/cfg_vdd_controlled_expanded_prompt_pool_hybrid_strict.pkl',
    group_images_by_size=True,
    image_batch_size=1,
    image_encoder_dtype='bf16',
    inference_dtype='bf16',
    mask_query_chunk_size=16,
    max_cross_image_slots=0,
    max_mask_tensor_mb=1024,
    model_type='SAM3',
    optimize_method='hybrid_strict',
    prob_thd=0.3,
    processor_resolution=1008,
    prompt_batch_size=1,
    router_enabled=False,
    shared_image_encoder_batch=False,
    slide_crop=1024,
    slide_stride=768,
    slot_batch_size=0,
    slot_chunk_size=2,
    type='CachedSAM3OpenSegmentor',
    use_presence_score=True,
    use_sem_seg=True,
    use_transformer_decoder=True)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='test/src', seg_map_path='test/gt'),
        data_root='/data/public/VDD',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='VDDDataset'),
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
work_dir = './work_dirs/cfg_vdd_controlled'
