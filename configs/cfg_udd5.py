_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_udd5.txt',
    confidence_threshold=0.5,
    prob_thd=0.1,
    bg_idx=4,
)

# dataset settings
dataset_type = 'UDD5Dataset'
data_root = '/data/public/UDD/UDD5'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/src',
            seg_map_path='val/gt'),
        pipeline=test_pipeline))