import os

_base_ = 'simclr_resnet50_4xb32-coslr-200e_nih.py'

# dataset summary
split_folder = "/scratchg/data/NIH-CXR/splits/SSL-50k/"
data = dict(
    samples_per_gpu=128,  
    workers_per_gpu=4,
    train=dict(
        data_source=dict(
            split_folder=split_folder,
            ann_file=os.path.join(split_folder, "train.txt"),
        ),
    ))  # total 64*8

runner = dict(max_epochs=500)

optimizer = dict(lr=0.3)