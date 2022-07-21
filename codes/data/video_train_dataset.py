"""
DAVIS dataset
support reading images from lmdb, image folder
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

logger = logging.getLogger("base")


def augment(img, hflip=True, vflip=True, rot=True):
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

    return _augment(img)


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
        elif self.data_type == "img":
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

    def __getitem__(self, index):
        if self.data_type == "lmdb" and (self.GT_env is None):
            self._init_lmdb()

        GT_size = self.opt["GT_size"]
        key = self.paths_GT[index]
        if self.opt["name"] == "GOT10k":
            name1, name2, name_a, name_b, num = key.split("_")
            #print(name1, name2, name_a, name_b, num)
        else:
            name_a, name_b, num = key.split("_")
        #### determine the neighbor frames default=1
        interval = random.choice(self.interval_list)

        # determine the first and last keyframes, total frames = 2 keyframes + interval frames
        frame_idx_first = int(name_b)
        frame_idx_last = frame_idx_first + self.keyframe_interval * interval + 1

        # num is the max id of img squence, such as 00058.jpg in 480p/tuk-tuk
        while (frame_idx_last > int(num)) or (
            frame_idx_first  < 0
        ):
            frame_idx_first = random.randint(
                1, int(num) - self.keyframe_interval * interval
            )
            frame_idx_last = frame_idx_first + self.keyframe_interval * interval + 1

        frame_idx_l = list(
            range(frame_idx_first, frame_idx_last + 1, interval)
        )

        # Get the frames
        LQ_l = []
        GT_l = []

        # Random reverse (determine once and apply to all frames)
        reverse = True if self.random_reverse and random.random() < 0.5 else False
        if reverse:
            frame_idx_l.reverse()

        # Get the frames within an interval

        for p in range(len(frame_idx_l)):
            frame_idx = frame_idx_l[p]

            #### get GT images
            if self.opt['name'] == 'DAVIS':
                GT_size_tuple = (3, 480, 854)
                ext = '.jpg'
            elif self.opt['name'] == 'videvo':
                GT_size_tuple = (3, 480, 852)
                ext = '.png'
            elif self.opt['name'] == 'DAVIS_videvo':
                GT_size_tuple = (3, 480, 852)
                ext = '.png'
            elif self.opt['name'] == 'GOT10k':
                GT_size_tuple = (3, 360, 640)
                ext = '.png'
            elif self.opt['name'] == 'videvo_300':
                GT_size_tuple = (3, 300, 300)
                ext = '.png'
            elif self.opt['name'] == 'DAVIS_videvo_300':                
                GT_size_tuple = (3, 300, 300)
                ext = '.png'
            else:
                print('Error!')
                exit()
            
            if self.data_type == "lmdb":
                img_GT = util.read_img(
                    self.GT_env, "{}_{:05d}".format(name_a, v), GT_size_tuple
                )
            else:
                if self.opt["name"] == "GOT10k":
                    img_GT_path = osp.join(self.GT_root, name1+'_'+name2+'_'+name_a, "{:08d}{}".format(p+1, ext))
                    #print(img_GT_path)
                else:
                    img_GT_path = osp.join(self.GT_root, name_a, "{:05d}{}".format(p, ext))
                    
                #print(img_GT_path)
                img_GT = util.read_img(None, img_GT_path) / 255.
                
            if self.opt["phase"] == "train":
                C, H, W = GT_size_tuple
                # randomly crop
                if p == 0:
                    rnd_h = random.randint(0, max(0, H - GT_size))
                    rnd_w = random.randint(0, max(0, W - GT_size))
                img_GT = img_GT[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]
                
                # augmentation - flip, rotate
                if p == 0:
                    hflip_ = True if random.random() < 0.5 else False
                    vflip_ = True if random.random() < 0.5 else False
                    rot_ = True if random.random() < 0.5 else False
                img_GT = augment(img_GT, hflip_, vflip_, rot_)
            
            
            img_GT = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_GT.copy(), (2, 0, 1)))
            ).float()
            #print(img_GT.size())

            GT_l.append(img_GT)
        GT_l = torch.stack(GT_l, dim=0)
        #print('222:', GT_l.size())
        if self.data_type == "img":
            img_GT_lab = util.rgb2lab(GT_l)
            #print('222:', img_GT_lab.size())
            
        img_LQ = img_GT_lab[:,0,:,:]
        LQ_l = [img_LQ[i:i+1,...] for i in range(img_LQ.shape[0])]
        
#         import matplotlib.pyplot as plt
#         plt.imshow(GT_l[0,...].detach().cpu().numpy().transpose(1,2,0)/255.)
#         plt.show()
#         plt.imshow(LQ_l[0,0,...].detach().cpu().numpy()/255.)
#         plt.show()
        
        return {"LQs": LQ_l, "GT": img_GT_lab, "key": key, "GT_HW": [H, W]}

    def __len__(self):
        return len(self.paths_GT)
