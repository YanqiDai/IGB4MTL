import fnmatch
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

"""
Source: https://github.com/Cranial-XIX/CAGrad/blob/main/nyuv2/create_dataset.py
"""


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=None):
        if scale is None:
            scale = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(
            img[None, :, i: i + h, j: j + w],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        label_ = (
            F.interpolate(
                label[None, None, i: i + h, j: j + w],
                size=(height, width),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        )
        depth_ = F.interpolate(
            depth[None, :, i: i + h, j: j + w], size=(height, width), mode="nearest"
        ).squeeze(0)
        normal_ = F.interpolate(
            normal[None, :, i: i + h, j: j + w],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        return img_, label_, depth_ / sc, normal_


class NYUv2(Dataset):
    def __init__(self, root, mode="train", augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if mode == "train":
            self.data_path = root + "/train"
        elif mode == "val":
            self.data_path = root + "/val"
        else:
            self.data_path = root + "/test"

        # get data_files and calculate data length
        self.data_files = fnmatch.filter(os.listdir(self.data_path + "/image"), "*.npy")
        self.data_len = len(self.data_files)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(
            np.moveaxis(
                np.load(self.data_path + "/image/{}".format(self.data_files[index])), -1, 0
            )
        )
        semantic = torch.from_numpy(
            np.load(self.data_path + "/label/{}".format(self.data_files[index]))
        )
        depth = torch.from_numpy(
            np.moveaxis(
                np.load(self.data_path + "/depth/{}".format(self.data_files[index])), -1, 0
            )
        )
        normal = torch.from_numpy(
            np.moveaxis(
                np.load(self.data_path + "/normal/{}".format(self.data_files[index])), -1, 0
            )
        )

        # apply data augmentation if required
        if self.augmentation:
            image, semantic, depth, normal = RandomScaleCrop()(
                image, semantic, depth, normal
            )
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = -normal[0, :, :]

        return image.float(), semantic.float(), depth.float(), normal.float()

    def __len__(self):
        return self.data_len
