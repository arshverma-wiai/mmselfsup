import os

_base_ = 'simclr_resnet50_4xb32-coslr-200e_nih.py'

# dataset summary
split_folder = "/scratchg/data/NIH-CXR/splits/SSL-68k/"
data = dict(
    samples_per_gpu=128,  
    workers_per_gpu=4,
    train=dict(
        data_source=dict(
            split_folder=split_folder,
            ann_file=os.path.join(split_folder, "train.txt"),
        ),
    )) 

runner = dict(max_epochs=500)

optimizer = dict(lr=0.3)

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    )