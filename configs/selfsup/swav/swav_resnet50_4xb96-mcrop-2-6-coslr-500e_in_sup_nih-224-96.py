import os

_base_ = [
    'swav_resnet50_4xb128-mcrop-2-6-coslr-200e_nih-224-96.py'
]

split_folder = "/scratchg/data/NIH-CXR/splits/SSL-50k/"
data = dict(
    samples_per_gpu=96,  
    workers_per_gpu=4,
    train=dict(
        data_source=dict(
            split_folder=split_folder,
            ann_file=os.path.join(split_folder, "train.txt"),
        ),
    )) 

runner = dict(type='EpochBasedRunner', max_epochs=500)

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/scratchg/data/NIH-CXR/checkpoints/selfsup/resnet50-supervised-updated.pth',
            prefix='backbone',
        )),
    )