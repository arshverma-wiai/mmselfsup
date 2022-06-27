from mmcls.models.backbones import DenseNet as _DenseNet
from mmcls.models.backbones.densenet import DenseLayer,DenseBlock,DenseTransition

from ..builder import BACKBONES

@BACKBONES.register_module()
class DenseNet(_DenseNet):
    """DenseNet.
    A PyTorch implementation of : `Densely Connected Convolutional Networks
    <https://arxiv.org/pdf/1608.06993.pdf>`_
    Modified from the `official repo
    <https://github.com/liuzhuang13/DenseNet>`_
    and `pytorch
    <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_.
    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``DenseNet.arch_settings``. And if dict, it
            should include the following two keys:
            - growth_rate (int): Each layer of DenseBlock produce `k` feature
              maps. Here refers `k` as the growth rate of the network.
            - depths (list[int]): Number of repeated layers in each DenseBlock.
            - init_channels (int): The output channels of stem layers.
            Defaults to '121'.
        in_channels (int): Number of input image channels. Defaults to 3.
        bn_size (int): Refers to channel expansion parameter of 1x1
            convolution layer. Defaults to 4.
        drop_rate (float): Drop rate of Dropout Layer. Defaults to 0.
        compression_factor (float): The reduction rate of transition layers.
            Defaults to 0.5.
        memory_efficient (bool): If True, uses checkpointing. Much more memory
            efficient, but slower. Defaults to False.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='ReLU')``.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict, optional): Initialization config dict.
    """
    arch_settings = {
        '121': {
            'growth_rate': 32,
            'depths': [6, 12, 24, 16],
            'init_channels': 64,
        },
        '169': {
            'growth_rate': 32,
            'depths': [6, 12, 32, 32],
            'init_channels': 64,
        },
        '201': {
            'growth_rate': 32,
            'depths': [6, 12, 48, 32],
            'init_channels': 64,
        },
        '161': {
            'growth_rate': 48,
            'depths': [6, 12, 36, 24],
            'init_channels': 96,
        },
    }
    def __init__(self,
                 arch='121',
                 in_channels=3,
                 bn_size=4,
                 drop_rate=0,
                 compression_factor=0.5,
                 memory_efficient=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 out_indices=0,
                 frozen_stages=0,
                 init_cfg=None):

        out_indices=-1

        super(DenseNet,self).__init__(
                 arch=arch,
                 in_channels=in_channels,
                 bn_size=bn_size,
                 drop_rate=drop_rate,
                 compression_factor=compression_factor,
                 memory_efficient=False,
                 norm_cfg=norm_cfg,
                 act_cfg=act_cfg,
                 out_indices=out_indices,
                 frozen_stages=frozen_stages,
                 init_cfg=init_cfg)

        temp_out_indices = self.out_indices
        self.out_indices=[idx+1 for idx in temp_out_indices]
    
    def forward(self, x):
        x = self.stem(x)
        outs = []
        if 0 in self.out_indices:
            outs.append(x)
        for i in range(self.num_stages):
            x = self.stages[i](x)
            x = self.transitions[i](x)
            if i+1 in self.out_indices:
                outs.append(x)

        return tuple(outs)