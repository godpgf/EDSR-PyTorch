from data.div2k import DIV2K
from data import common
import os
import pickle
import imageio
import cv2


class DIV2KSS(DIV2K):
    def __init__(self, args, name='DIV2KSS', train=True, benchmark=False):
        super(DIV2KSS, self).__init__(args, name.replace("SS", ""), train, benchmark)

    def _check_and_load(self, ext, img, f, verbose=True, scale=1):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                img = imageio.imread(img)
                if scale != 1:
                    x, y = img.shape[0:2]
                    img = cv2.resize(img, (y * scale, x * scale), interpolation=cv2.INTER_LINEAR)
                pickle.dump(img, _f)
                print(img.shape)

    def get_patch(self, lr, hr):
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=1,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih, 0:iw]

        return lr, hr

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair = common.add_noise(*pair, noise_type=self.args.noise_type, noise_param=self.args.noise_param)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

