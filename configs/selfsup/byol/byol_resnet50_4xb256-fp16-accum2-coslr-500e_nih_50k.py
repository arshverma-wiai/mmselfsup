import os

_base_ = 'byol_resnet50_8xb256-fp16-accum2-coslr-200e_nih_50k.py'

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=500)

# dataset summary
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

