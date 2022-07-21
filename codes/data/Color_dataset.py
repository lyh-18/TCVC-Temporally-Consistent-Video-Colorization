import random
import numpy as np
import pickle
from skimage.transform import resize

import lmdb
import torch
import torch.utils.data as data
import data.util as util


class ColorDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(ColorDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt["data_type"]
        self.paths_GT = None
        self.sizes_GT = None
        self.GT_env = None  # envir onments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(
            self.data_type, opt["dataroot_GT"]
        )

        assert self.paths_GT, "Error: GT path is empty."
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.data_type == "lmdb" and (self.GT_env is None):
            self._init_lmdb()
        GT_path = None
        GT_size = self.opt["GT_size"]

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = (
            [int(s) for s in self.sizes_GT[index].split("_")]
            if self.data_type == "lmdb"
            else None
        )
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        H, W, _ = img_GT.shape
        if self.opt["phase"] == "train":
            # if the image size is too small
            if H < GT_size or W < GT_size:
                img_GT = resize(img_GT, (GT_size, GT_size))
            else:
                # randomly crop
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_GT = img_GT[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_GT = util.augment([img_GT], self.opt["use_flip"], self.opt["use_rot"])[
                0
            ]
        else:
            img_GT = resize(img_GT, (GT_size, GT_size))
            
        if img_GT.shape[2] == 3:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]  # rgb -> LAB
            img_LQ = img_GT[:, :, 0:1]
        else:
            img_LQ = img_GT
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))
        ).float()

        return {
            "LQ": img_LQ,
            "GT": img_GT,
            "LQ_path": GT_path,
            "GT_path": GT_path,
            "GT_HW": (H, W),
        }

    def __len__(self):
        return len(self.paths_GT)
