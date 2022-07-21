"""
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
"""

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math

import utils.util as util
import data.util as data_util

import models.archs.TCVC_IDC_arch as TCVC_IDC_arch

from compute_hist import *


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_folders(input_path, GT_path, interval_length, logger):
    input_folder_list = os.listdir(input_path)
    input_folder_list.sort()
    
    avg_psnr_l = []
    key_avg_psnr_l = []
    inter_avg_psnr_l = []
    key_n_l = []
    inter_n_l = []
    
    for folder in input_folder_list:
        if not os.path.isdir(os.path.join(input_path, folder)):
            continue
      
        GT_img_path_l = sorted(glob.glob(osp.join(GT_path, folder, "*")))
        Input_img_path_l = sorted(glob.glob(osp.join(input_path, folder, "*")))
        
        max_idx = len(GT_img_path_l)
        keyframe_idx = list(range(0, max_idx, interval_length + 1))
        print(keyframe_idx)
        
        avg_psnr, N_im = 0, 0
        key_avg_psnr, inter_avg_psnr = 0, 0
        key_N_im, inter_N_im = 0, 0
        count = 0

        for img1_path, img2_path in zip(GT_img_path_l, Input_img_path_l):
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img_name = img1_path.split('/')[-1]
              
            psnr = calculate_psnr(img1, img2)        
            avg_psnr += psnr
            

                        
            if count in keyframe_idx or count == len(GT_img_path_l)-1:
                key_avg_psnr += psnr
                key_N_im += 1
                key_flag = True
                #print(img1_path)
            else:
                inter_avg_psnr += psnr
                inter_N_im += 1
                key_flag = False
   
            count += 1
            N_im += 1
            print(N_im)
            
            logger.info(
                "{:3d} - {:25} \tPSNR: {:.6f} dB   key frame: {}".format(
                    count, img_name, psnr, key_flag
                )
            )
            
        avg_psnr /= N_im
        avg_psnr_l.append(avg_psnr)
        
        key_avg_psnr /= key_N_im
        key_avg_psnr_l.append(key_avg_psnr)
        
        inter_avg_psnr /= inter_N_im
        inter_avg_psnr_l.append(inter_avg_psnr)
        
        key_n_l.append(key_N_im)
        inter_n_l.append(inter_N_im)
        
        message = "Folder {} - Average PSNR: {:.6f} dB for {} frames; AVG key PSNR: {:.6f} dB for {} key frames; AVG inter PSNR: {:.6f} dB for {} inter frames.".format(
                    folder, avg_psnr, N_im, key_avg_psnr, key_N_im, inter_avg_psnr, inter_N_im)
        logger.info(message)

         
    logger.info("################ Final Results ################")
    logger.info('Inter: {}'.format(str(interval_length)))
    
    
    message = "Total Average PSNR: {:.6f} dB for {} clips; AVG key PSNR: {:.6f} dB for {} key frames; AVG inter PSNR: {:.6f} dB for {} inter frames.".format(
        sum(avg_psnr_l) / len(avg_psnr_l), len(input_folder_list), 
        sum(key_avg_psnr_l) / len(key_avg_psnr_l), sum(key_n_l), 
        sum(inter_avg_psnr_l) / len(inter_avg_psnr_l), sum(inter_n_l), 
        )
    logger.info(message)

    
    return avg_psnr_l



def save_imglist(k, end_k, output_dir, img_list, logger, img_paths):
    """The color type of input img list is rgb"""
    count = 0
    for i in range(k, end_k):
        imname = img_paths[count].split('/')[-1]
        out_path = os.path.join(output_dir, imname)
        #logger.info("save img: {}".format(out_path))
        cv2.imwrite(out_path, img_list[count][:,:,::-1])
        count += 1

def main():
    #################
    # configurations
    #################
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
    data_mode = "DAVIS30"  # DAVIS30 | Videvo20
    key_net = "IDC"
    color_type = "LAB"
    GT_size = 256
    
    model_path = "../experiments/TCVC_IDC/models/80000_G.pth"  

    #### interval length (support only uniform interval here) (0, N, 2N, 3N, ...)
    interval_length = 17
    
    #### dataset path and model
    if data_mode == "DAVIS30":
        GT_dataset_folder = "/data2/yhliu/DATA/DAVIS-2017-trainval-480p/DAVIS30_GT_mod32_new/"
    elif data_mode == "Videvo20":
        GT_dataset_folder = "/data2/yhliu/DATA/videvo20_mod32/"
        
    
    save_folder = "../results/TCVC_{}_{}_interlen{}".format(key_net, data_mode, interval_length)
    
    
    if key_net == "IDC":           
        model = TCVC_IDC_arch.TCVC_IDC(nf=64, N_RBs=3, key_net="sig17", dataset="DAVIS4")
    else:
        raise NotImplementedError('Backbone [{}] is not yet ready!'.format(key_net))

    #### evaluation
    crop_border = 0

    # temporal padding mode
    padding = "new_info"
    save_imgs = True

    util.mkdirs(save_folder)
    util.setup_logger(
        "base", save_folder, "test", level=logging.INFO, screen=True, tofile=True
    )
    logger = logging.getLogger("base")

    #### log info
    logger.info("Data: {} - {}".format(data_mode, GT_dataset_folder))
    logger.info("Padding mode: {}".format(padding))
    logger.info("Model path: {}".format(model_path))
    logger.info("Save images: {}".format(save_imgs))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)


    video_list = sorted(os.listdir(GT_dataset_folder))
    video_list = [i for i in video_list if os.path.isdir(os.path.join(GT_dataset_folder, i))]
    avg_psnr_l = []
    
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
    
    for i in range(len(video_list)):
        video = video_list[i]
        ## mkdir output dir
        save_subfolder = osp.join(save_folder, video)
        if save_imgs:
            util.mkdirs(save_subfolder)
        
        video_dir_path = os.path.join(GT_dataset_folder, video)
        img_list = sorted(glob.glob(os.path.join(video_dir_path, "*.png"))) # you may change the suffix
        #print(img_list)
        imgs = [data_util.read_img(None, img_list[i])/255. for i in range(len(img_list))]
        
        keyframe_idx = list(range(0, len(imgs), interval_length+1))
        if keyframe_idx[-1] == (len(imgs)-1):
            keyframe_idx = keyframe_idx[:-1]
        print("Processing '{}'".format(video))
        print("Total images: {}  keyframe index: {}".format(len(imgs), keyframe_idx))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        count = 0
        avg_psnr, N_im = 0, 0
        for k in keyframe_idx:
            img_paths = img_list[k:k+interval_length+2]
            img_in = imgs[k:k+interval_length+2] # get input list
            img_in = np.stack(img_in, 0) # [9, H, W, 3] rgb
            img_tensor = torch.from_numpy(img_in.transpose(0,3,1,2)).float()
            img_lab_tensor = data_util.rgb2lab(img_tensor) # [9, 3, H, W] lab (-1, 1)
            img_l_tensor = img_lab_tensor[:,:1,:,:] # get l channel, original size (-0.5, 0.5)
            
            img_l_rs_tensor = F.interpolate(img_l_tensor, size=[GT_size, GT_size], mode="bilinear") # resize l channel to 256*256\
            img_l_rs_tensor_list = [img_l_rs_tensor[i:i+1,...].cuda() for i in range(img_l_rs_tensor.shape[0])] # generate input list
            
            with torch.no_grad():
                out_ab, _, _, _, _ = model(img_l_rs_tensor_list) # [1, 9, 3, H, W] rgb (0, 1)
#                 out_rgb = torch.cat((out_rgb[:,:1,:,:,:], w_rgb), 1)
            out_ab = out_ab.detach().cpu()[0,...]

            N, C, H, W = img_tensor.size() 
            out_a_rs = F.interpolate(out_ab[:,:1,:,:], size=[H, W], mode="bilinear") # resize ab channel to original size
            out_b_rs = F.interpolate(out_ab[:,1:2,:,:], size=[H, W], mode="bilinear")
#             out_ab_rs = F.interpolate(out_ab, size=[H, W], mode="bilinear")
            out_lab_origsize = torch.cat((img_l_tensor, out_a_rs, out_b_rs), 1) # concat
            out_rgb_origsize = data_util.lab2rgb(out_lab_origsize) # lab to rgb [9, 3, H, W] (0, 1)
            
            out_rgb_img = [util.tensor2img(np.clip(out_rgb_origsize[i,...]*255., 0, 255), np.uint8) for i in range(out_rgb_origsize.size(0))] # (0, 255)
            #import matplotlib.pyplot as plt
            #plt.imshow(out_rgb_img[0])
            #plt.show()
            
            
            save_imglist(k, k+len(out_rgb_img), save_subfolder, out_rgb_img, logger, img_paths)
            
    avg_psnr_l = calculate_psnr_folders(save_folder, GT_dataset_folder, interval_length, logger)
    
        
    dilation = [1,2,4]
    weight = [1/3, 1/3, 1/3]    
    JS_b_mean_list, JS_g_mean_list, JS_r_mean_list, JS_b_dict, JS_g_dict, JS_r_dict, CDC = calculate_folders_multiple(save_folder, data_mode, dilation=dilation, weight=weight)

    logger.info("################ Tidy Outputs ################")
    for (
        video,
        psnr,
    ) in zip(video_list, avg_psnr_l):
        logger.info("Folder {} - Average PSNR: {:.6f} dB.".format(video, psnr))
    logger.info("################ Final Results ################")
    logger.info("Data: {} - {}".format(data_mode, GT_dataset_folder))
    logger.info("Padding mode: {}".format(padding))
    logger.info("Model path: {}".format(model_path))
    logger.info("Save images: {}".format(save_imgs))
    logger.info(
        "Total Average PSNR: {:.6f} dB for {} clips.".format(
            sum(avg_psnr_l) / len(avg_psnr_l), len(video_list)
        )
    )
    logger.info("JS_b_mean: {:.6f} JS_g_mean: {:.6f} JS_r_mean: {:.6f}  CDC: {:.6f}".format(np.mean(JS_b_mean_list), np.mean(JS_g_mean_list), np.mean(JS_r_mean_list), CDC))
    
    
    with open('{}/val_log.txt'.format(save_folder), 'a') as f:
        f.write('AVG PSNR: {:.4f}  AVG JS_b: {:.6f}  AVG JS_g: {:.6f}  AVG JS_r: {:.6f} CDC: {:.6f}'.format(sum(avg_psnr_l) / len(avg_psnr_l), np.mean(JS_b_mean_list), np.mean(JS_g_mean_list), np.mean(JS_r_mean_list), CDC))
        f.write('\n')
    return sum(avg_psnr_l) / len(avg_psnr_l), np.mean(JS_b_mean_list), np.mean(JS_g_mean_list), np.mean(JS_r_mean_list)

if __name__ == '__main__':
    main()
