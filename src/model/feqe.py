from model import common

import torch.nn as nn


def make_model(args, parent=False):
    return FEQE(args)


class FEQE(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FEQE, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        compress = args.compress
        act = nn.ReLU(True)

        m_sub_mean = common.MeanShift(args.rgb_range)
        self.sub_mean = m_sub_mean
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [common.EDownsampler(conv, compress, n_feats)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = [
            common.EUpsampler(conv, compress, n_feats)
        ]

        setattr(self, "head%d" % compress, nn.Sequential(*m_head))
        self.head = [getattr(self, "head%d" % compress)]
        # self.body = nn.Sequential(*m_body)
        self.body_block_list = []
        for i, block in enumerate(m_body):
            setattr(self, "body.%d" % i, block)
            self.body_block_list.append(getattr(self, "body.%d" % i))
        setattr(self, "tail%d" % compress, nn.Sequential(*m_tail))
        self.tail = [getattr(self, "tail%d" % compress)]
        self.progress = 0.0

    def distilling(self, progress):
        self.progress = progress

    def forward(self, x):
        x = self.sub_mean(x)

        res = self.head[0](x)
        # res = self.body(res)
        for body_block in self.body_block_list:
            res = body_block(res)
        x = self.tail[0](res) + x

        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
