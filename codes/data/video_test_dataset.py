import os.path as osp
import torch
import torch.utils.data as data
import torch.nn.functional as F
import data.util as util


class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    DAVIS4

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt["cache_data"]
        self.half_N_frames = opt["N_frames"] // 2
        self.keyframe_interval = opt["keyframe_interval"]
        self.GT_root = opt["dataroot_GT"]
        self.data_type = self.opt["data_type"]
        self.data_info = {"path_GT": [], "folder": [], "idx": [], "border": []}
        if self.data_type == "lmdb":
            raise ValueError("No need to use LMDB during validation/test.")
        #### Generate data info and cache data
        self.imgs_GT = {}
        self.imgs_LQ = {}
        if opt["name"].lower() in ["davis4"]:
            subfolders_GT = util.glob_file_list(self.GT_root)
            for subfolder_GT in subfolders_GT:
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_GT)

                self.data_info["path_GT"].extend(img_paths_GT)
                self.data_info["folder"].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info["idx"].append("{}/{}".format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info["border"].extend(border_l)

                if self.cache_data:
                    GT_size = opt["GT_size"]
                    (
                        self.imgs_LQ[subfolder_name],
                        self.imgs_GT[subfolder_name],
                    ) = util.read_img_lab_seq(img_paths_GT, opt["color"])
                    self.GT_HW = self.imgs_GT[subfolder_name].shape[-2:]
                    self.imgs_LQ[subfolder_name] = F.interpolate(
                        self.imgs_LQ[subfolder_name],
                        size=[GT_size, GT_size],
                        mode="bilinear",
                    )
        #                 print(self.imgs_LQ[subfolder_name].shape, self.imgs_GT[subfolder_name].shape)
        else:
            raise ValueError("Not support video test dataset. Support DAVIS4.")

    def __getitem__(self, index):
        folder = self.data_info["folder"][index]
        idx, max_idx = self.data_info["idx"][index].split("/")
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info["border"][index]
        if self.cache_data:
            select_idx = util.index_generation(
                idx, max_idx, self.opt["N_frames"], padding=self.opt["padding"]
            )
            img_GT = self.imgs_GT[folder][idx]
            imgs_LQ = self.imgs_LQ[folder][select_idx]
            if self.half_N_frames == 0:
                imgs_LQ = imgs_LQ.squeeze(0)
        else:
            pass  # TODO
        return {
            "LQs": [imgs_LQ],
            "GT": img_GT,
            "GT_HW": self.GT_HW,
            "folder": folder,
            "idx": index,
            "border": border,
        }

    def __len__(self):
        return len(self.data_info["path_GT"])
