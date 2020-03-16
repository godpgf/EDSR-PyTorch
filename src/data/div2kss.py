from data.div2k import DIV2K
from data import common
from torch import nn
import random


class DIV2KSS(DIV2K):
    def __init__(self, args, name='DIV2KSS', train=True, benchmark=False):
        super(DIV2KSS, self).__init__(args, name.replace("SS", ""), train, benchmark)
        self.upsample = nn.Upsample(scale_factor=self.scale[0], mode='bilinear', align_corners=True)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return self.upsample(pair_t[0].unsqueeze(0)).squeeze(0), pair_t[1], filename

