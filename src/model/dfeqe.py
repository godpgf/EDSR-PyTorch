# Distilling feqe
from model import common

import torch.nn as nn


def make_model(args, parent=False):
    return DFEQE(args)


class DFEQE(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DFEQE, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats // 4
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

        m_tail = [
            conv(n_feats, n_feats, kernel_size),
            common.EUpsampler(conv, compress, n_feats)
        ]

        setattr(self, "head%d" % compress, nn.Sequential(*m_head))
        head = [getattr(self, "head%d" % compress)]
        # self.body = nn.Sequential(*m_body)
        body_block_list = []
        for i, block in enumerate(m_body):
            setattr(self, "body.%d" % i, block)
            body_block_list.append(getattr(self, "body.%d" % i))
        setattr(self, "tail%d" % compress, nn.Sequential(*m_tail))
        tail = [getattr(self, "tail%d" % compress)]

        # 蒸馏
        m_head_x2 = [common.EDownsampler(conv, compress, n_feats * 2)]
        m_head_x4 = [common.EDownsampler(conv, compress, n_feats * 4)]
        setattr(self, "head%d_x2" % compress, nn.Sequential(*m_head_x2))
        head_x2 = [getattr(self, "head%d_x2" % compress)]
        setattr(self, "head%d_x4" % compress, nn.Sequential(*m_head_x4))
        head_x4 = [getattr(self, "head%d_x4" % compress)]

        m_tail_x2 = [conv(n_feats * 2, n_feats * 2, kernel_size), common.EUpsampler(conv, compress, n_feats * 2)]
        m_tail_x4 = [conv(n_feats * 4, n_feats * 4, kernel_size), common.EUpsampler(conv, compress, n_feats * 4)]
        setattr(self, "tail%d_x2" % compress, nn.Sequential(*m_tail_x2))
        tail_x2 = [getattr(self, "tail%d_x2" % compress)]
        setattr(self, "tail%d_x4" % compress, nn.Sequential(*m_tail_x4))
        tail_x4 = [getattr(self, "tail%d_x4" % compress)]

        m_body_x4 = [
            common.ResBlock(
                conv, n_feats * 4, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]

        body_x4_block_list = []
        for i, block in enumerate(m_body_x4):
            setattr(self, "body_x4.%d" % i, block)
            body_x4_block_list.append(getattr(self, "body_x4.%d" % i))

        m_body_x2 = [
            common.ResBlock(
                conv, n_feats * 2, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]

        body_x2_block_list = []
        for i, block in enumerate(m_body_x2):
            setattr(self, "body_x2.%d" % i, block)
            body_x2_block_list.append(getattr(self, "body_x2.%d" % i))

        m_body_x4_x2 = [conv(n_feats * 4, n_feats * 2, 1) for _ in range(n_resblocks + 1)]
        body_x4_x2_list = []
        for i, block in enumerate(m_body_x4_x2):
            setattr(self, "body_x4_x2.%d" % i, block)
            body_x4_x2_list.append(getattr(self, "body_x4_x2.%d" % i))


        m_body_x2_x1 = [conv(n_feats * 2, n_feats, 1) for _ in range(n_resblocks + 1)]
        body_x2_x1_list = []
        for i, block in enumerate(m_body_x2_x1):
            setattr(self, "body_x2_x1.%d" % i, block)
            body_x2_x1_list.append(getattr(self, "body_x2_x1.%d" % i))

        tmp_body_x4_x2 = []
        for i in range(n_resblocks + 1, -1, -1):
            cur_body = body_x4_block_list[0: i] + [body_x4_x2_list[i]]
            tmp_body_x4_x2.append(cur_body)

        tmp_body_x2_x1 = []
        for i in range(n_resblocks + 1, -1, -1):
            cur_body = body_x2_block_list[0: i] + [body_x2_x1_list[i]]
            tmp_body_x2_x1.append(cur_body)

        self.head_list = head_x4 + head_x4 * (n_resblocks + 1) + head_x2 * (n_resblocks + 1) + head
        self.body_list = [body_x4_block_list] + tmp_body_x4_x2 + tmp_body_x2_x1 + [body_block_list]
        self.tail_list = tail_x4 + tail_x2 * (n_resblocks + 1) + tail * (n_resblocks + 1) + tail
        self.progress = 0.0

    def distilling(self, progress):
        self.progress = progress

    def forward(self, x):
        x = self.sub_mean(x)

        # progress = (len(self.head_list) - 1) * self.progress
        # index = int(progress)
        # sub_progress = progress - index

        progress = len(self.head_list) * self.progress
        index = int(progress)

        def _forward(i):
            res = self.head_list[i](x)
            body_block_list = self.body_list[i]
            for body_block in body_block_list:
                res = body_block(res)
            return self.tail_list[i](res)

        # if sub_progress == 0:
        #     x = _forward(index) + x
        # else:
        #     x = (_forward(index) * (1.0 - sub_progress) + _forward(index+1) * sub_progress) + x
        x = _forward(index) + x

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
