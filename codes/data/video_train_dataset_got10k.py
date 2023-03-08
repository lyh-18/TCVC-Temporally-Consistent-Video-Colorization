"""
DAVIS dataset
support reading images from lmdb, image folder and memcached
"""
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util

try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger("base")


def augment(img_list, hflip=True, vflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip
    vflip = rot
    rot90 = rot

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


class VideoTrainDataset(data.Dataset):
    """
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality (Gray image)
    support reading N LQ frames, N = 1, 3, 5, 7
    """

    def __init__(self, opt):
        super(VideoTrainDataset, self).__init__()
        assert opt["N_frames"] % 2 == 1, "[N_frames] must be an odd number."

        self.opt = opt

        #### keyframe interval
        self.keyframe_interval = opt["keyframe_interval"]  # integer

        # temporal augmentation
        self.interval_list = opt["interval_list"]
        self.random_reverse = opt["random_reverse"]
        logger.info(
            "Temporal augmentation interval list: [{}], with random reverse is {}.".format(
                ",".join(str(x) for x in opt["interval_list"]), self.random_reverse
            )
        )

        self.half_N_frames = opt["N_frames"] // 2
        self.GT_root = opt["dataroot_GT"]
        self.data_type = self.opt["data_type"]
        print(self.data_type)

        #### directly load image keys
        if self.data_type == "lmdb":
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt["dataroot_GT"])
            logger.info("Using lmdb meta info for cache keys.")
        elif opt["cache_keys"]:
            logger.info("Using cache keys: {}".format(opt["cache_keys"]))
            self.paths_GT = pickle.load(open(opt["cache_keys"], "rb"))["keys"]
        else:
            raise ValueError(
                "Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]"
            )

        assert self.paths_GT, "Error: GT path is empty."

        if self.data_type == "lmdb":
            self.GT_env = None
        elif self.data_type == "mc":  # memcached
            self.mclient = None
        elif self.data_type == "img":
            pass
        elif self.data_type == "npy":
            pass
        else:
            raise ValueError("Wrong data type: {}".format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file
            )

    def _read_img_mc(self, path):
        """ Return BGR, HWC, [0, 255], uint8"""
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        """ Read BGR channels separately and then combine for 1M limits in cluster"""
        img_B = self._read_img_mc(osp.join(path + "_B", name_a, name_b + ".jpg"))
        img_G = self._read_img_mc(osp.join(path + "_G", name_a, name_b + ".jpg"))
        img_R = self._read_img_mc(osp.join(path + "_R", name_a, name_b + ".jpg"))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == "mc":
            self._ensure_memcached()
        elif self.data_type == "lmdb" and (self.GT_env is None):
            self._init_lmdb()

        GT_size = self.opt["GT_size"]
        key = self.paths_GT[index]
        if self.opt["name"] == "GOT10k":
            name1, name2, name_a, name_b, num = key.split("_")
        else:
            name_a, name_b, num = key.split("_")
        #### determine the neighbor frames default=1
        interval = random.choice(self.interval_list)

        # determine the first and last keyframes, total frames = 2 keyframes + interval frames
        center_frame_idx_first = int(name_b)
        center_frame_idx_last = (
            center_frame_idx_first + self.keyframe_interval * interval + 1
        )

        # num is the max id of img squence, such as 00058.jpg in 480p/tuk-tuk
        while (center_frame_idx_last + self.half_N_frames * interval > int(num)) or (
            center_frame_idx_first - self.half_N_frames * interval < 0
        ):
            center_frame_idx_first = random.randint(
                1, int(num) - self.keyframe_interval * interval
            )
            center_frame_idx_last = (
                center_frame_idx_first + self.keyframe_interval * interval + 1
            )

        c_frame_idx_l = list(
            range(center_frame_idx_first, center_frame_idx_last + 1, interval)
        )

        # Get the frames
        LQ_l = []
        GT_l = []

        # Random reverse (determine once and apply to all frames)
        reverse = True if self.random_reverse and random.random() < 0.5 else False
        if reverse:
            c_frame_idx_l.reverse()

        # Get the frames within an interval
        for p in range(len(c_frame_idx_l)):
            center_frame_idx = c_frame_idx_l[p]
            if p > 0 and p < len(c_frame_idx_l) - 1:
                half_N_frames = 0
            else:
                half_N_frames = self.half_N_frames

            neighbor_list = list(
                range(
                    center_frame_idx - half_N_frames * interval,
                    center_frame_idx + half_N_frames * interval + 1,
                    interval
                )
            )
            if reverse:
                neighbor_list.reverse()

            #### get GT images
            if self.opt['name'] == 'DAVIS':
                GT_size_tuple = (3, 480, 854)
            elif self.opt['name'] == 'videvo':
                GT_size_tuple = (3, 480, 852)
            elif self.opt['name'] == 'DAVIS_videvo':
                GT_size_tuple = (3, 480, 852)
            elif self.opt['name'] == 'GOT10k':
                GT_size_tuple = (3, 360, 640)

            img_GT_l = []
            for v in neighbor_list:
                if self.data_type == "mc":
                    img_GT = self._read_img_mc(img_GT_path)
                elif self.data_type == "lmdb":
                    img_GT = util.read_img(
                        self.GT_env, "{}_{:05d}".format(name_a, v), GT_size_tuple
                    )
                elif self.data_type == "npy":
                    img_GT_path = osp.join(self.GT_root, name_a, "{:05d}.npy".format(v))
                    img_GT = util.read_img_npy(
                        None, img_GT_path
                    )
                else:
                    if self.opt["name"] == "GOT10k":
                        img_GT_path = osp.join(self.GT_root, name1+'_'+name2+'_'+name_a, "{:08d}.png".format(v))
                    else:
                        img_GT_path = osp.join(self.GT_root, name_a, "{:05d}.png".format(v))

#                     print(img_GT_path)
                    img_GT = util.read_img(None, img_GT_path)
                img_GT_l.append(img_GT)

            #### rgb2lab
            if self.data_type == "img":
                img_GT_l = util.channel_convert(3, self.opt["color"], img_GT_l)

            if self.opt["phase"] == "train":
                C, H, W = GT_size_tuple
                # randomly crop
                if p == 0:
                    rnd_h = random.randint(0, max(0, H - GT_size))
                    rnd_w = random.randint(0, max(0, W - GT_size))
                img_GT_l = [
                    v[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]
                    for v in img_GT_l
                ]

                # augmentation - flip, rotate
                if p == 0:
                    hflip_ = True if random.random() < 0.5 else False
                    vflip_ = True if random.random() < 0.5 else False
                    rot_ = True if random.random() < 0.5 else False
                img_GT_l = augment(img_GT_l, hflip_, vflip_, rot_)
                img_LQ_l = [v[:, :, :1] for v in img_GT_l]

            # stack LQ images to NHWC, N is the frame number
            img_GT = img_GT_l[half_N_frames]
            img_LQs = np.stack(img_LQ_l, axis=0)
            img_GT = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
            ).float()
            img_LQs = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))
            ).float()

            if self.half_N_frames == 0:
                img_LQs = img_LQs.squeeze(0)
            GT_l.append(img_GT)
            LQ_l.append(img_LQs)

        GT_l = torch.stack(GT_l, dim=0)
        return {"LQs": LQ_l, "GT": GT_l, "key": key, "GT_HW": [H, W]}

    def __len__(self):
        return len(self.paths_GT)
