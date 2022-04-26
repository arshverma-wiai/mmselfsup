# dataset settings
data_source = 'NIH'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=90),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.2),
    dict(type='RandomGrayscale', p=0.2),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='"/scratchg/data/NIH-CXR/images"',
            ann_file=None,
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))
