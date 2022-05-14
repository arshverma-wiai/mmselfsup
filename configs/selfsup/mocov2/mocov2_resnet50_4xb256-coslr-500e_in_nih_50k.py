import os

_base_ = [
    'mocov2_resnet50_4xb32-coslr-200e_nih.py'
]

split_folder = "/scratchg/data/NIH-CXR/splits/SSL-50k/"
data = dict(
    samples_per_gpu=256,  
    workers_per_gpu=10,
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
            checkpoint='/scratchg/data/NIH-CXR/checkpoints/selfsup/resnet50_imagenet_bs2k_epochs600.pth.tar',
            prefix='backbone',
        )),
    )