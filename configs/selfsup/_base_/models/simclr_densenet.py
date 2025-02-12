# model settings
model = dict(
    type='SimCLR',
    backbone=dict(
        type='DenseNet',
        arch='161',
        in_channels=3,
        out_indices=0,  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=2208,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1))