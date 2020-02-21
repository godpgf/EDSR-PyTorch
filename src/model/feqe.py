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
        scale = args.scale[0]
        act = nn.ReLU(True)
        enhance = args.enhance
        test_only = args.test_only

        m_sub_mean = common.MeanShift(args.rgb_range)
        self.sub_mean = m_sub_mean if enhance or test_only else nn.Sequential(*[m_sub_mean, nn.Upsample(scale_factor=scale, mode='bilinear')])
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [common.EDownsampler(conv, scale, n_feats)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = [
            common.EUpsampler(conv, scale, n_feats)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)

        res = self.head(x)
        res = self.body(res)
        x = self.tail(res) + x

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
