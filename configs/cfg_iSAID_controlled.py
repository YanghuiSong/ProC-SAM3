# Controlled config for iSAID dataset using hybrid_strict prompt optimization
_base_ = ['./base_config.py']

# Dataset
data_root = '/data/public/iSAID'
dataset_type = 'iSAIDDataset'

# Class names for iSAID dataset
class_names = [
    'background', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
    'basketball_court', 'ground_track_field', 'bridge', 'large_vehicle',
    'small_vehicle', 'helicopter', 'swimming_pool', 'roundabout',
    'soccer_ball_field', 'plane', 'harbor'
]

# Model
model = dict(
    type='CachedSAM3OpenSegmentor',
    classname_path='./configs/cls_iSAID.txt',
    optimize_method='hybrid_strict',  # "adaptive" or "guided" or "preset"
    prob_thd=0.1,
    bg_idx=0,
    confidence_threshold=0.4,
    use_sem_seg=True,
    use_presence_score=True,
    use_transformer_decoder=False,  # True or False
    slide_stride=0,
    slide_crop=0,
    prompt_batch_size=15,
    image_batch_size=4,
    slot_batch_size=0,
    auto_tune_slot_batch=False,
    execution_mode='per_image',
    mask_query_chunk_size=32,
    slot_chunk_size=4,
    max_cross_image_slots=15,
    max_mask_tensor_mb=1024,
    group_images_by_size=True,
    shared_image_encoder_batch=True,
    compile_model=False,
    processor_resolution=1008,
    inference_dtype='fp32',
    image_encoder_dtype='bf16',
    class_aggregation='max',  # 'consensus_gem', max，
    class_aggregation_topk=2,
    class_aggregation_temperature=0.4,
    class_aggregation_gem_power=6.0,
    class_aggregation_consensus_beta=8.0,
    class_aggregation_reliability_alpha=0.7,
    class_aggregation_coverage_penalty=0.2,
    class_aggregation_consensus_boost=0.3,
    class_aggregation_support_ratio=0.7,
    class_aggregation_boost_agreement_threshold=0.8,
    class_aggregation_boost_max_coverage=0.3,
    class_aggregation_keep_ratio=0.85,
    class_aggregation_max_selected=0,
    class_aggregation_second_boost_scale=0.18,
    class_aggregation_second_suppress_scale=0.0,
    ccpea_prompt_fusion_alpha=0.6,
    ccpea_presence_weight=0.7,
    ccpea_instance_weight=0.3,
    ccpea_uncertainty_weight=0.5,
    ccpea_consensus_weight=0.3,
    ccpea_pool_size=8,
    ccpea_topk=2,
    router_enabled=False,
    cache_text_embeddings=True,
    enable_expanded_prompt=True,
    expanded_prompt_pool_path=None,
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
    type='CombinedMetrics',
    iou_metrics=['mIoU'],
    iou_thresholds=[0.5, 0.75],
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
