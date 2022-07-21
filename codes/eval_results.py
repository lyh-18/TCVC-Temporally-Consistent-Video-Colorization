import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import glob
import os.path as osp
import math

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_folders(input_path, GT_path, interval_length):
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
            
            print(
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
        print(message)

         
    print("################ Final Results ################")
    print('Inter: {}'.format(str(interval_length)))
    
    
    message = "Total Average PSNR: {:.6f} dB for {} clips; AVG key PSNR: {:.6f} dB for {} key frames; AVG inter PSNR: {:.6f} dB for {} inter frames.".format(
        sum(avg_psnr_l) / len(avg_psnr_l), len(input_folder_list), 
        sum(key_avg_psnr_l) / len(key_avg_psnr_l), sum(key_n_l), 
        sum(inter_avg_psnr_l) / len(inter_avg_psnr_l), sum(inter_n_l), 
        )
    print(message)

    
    return avg_psnr_l


def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)



def compute_JS_bgr(input_dir, dilation=1):
    input_img_list = os.listdir(input_dir)
    input_img_list.sort()
    #print(input_img_list)

    hist_b_list = []   # [img1_histb, img2_histb, ...]
    hist_g_list = []
    hist_r_list = []
    
    for img_name in input_img_list:
        #print(os.path.join(input_dir, img_name))
        img_in = cv2.imread(os.path.join(input_dir,  img_name))
        H, W, C = img_in.shape
        
        hist_b = cv2.calcHist([img_in],[0],None,[256],[0,256]) # B
        hist_g = cv2.calcHist([img_in],[1],None,[256],[0,256]) # G
        hist_r = cv2.calcHist([img_in],[2],None,[256],[0,256]) # R
        
        hist_b = hist_b/(H*W)
        hist_g = hist_g/(H*W)
        hist_r = hist_r/(H*W)
        
        hist_b_list.append(hist_b)
        hist_g_list.append(hist_g)
        hist_r_list.append(hist_r)
            
        '''
        plt.subplot(1,2,1)
        plt.imshow(img_in[:,:,[2,1,0]])
        plt.subplot(1,2,2)
        plt.plot(hist_b, color='b')
        plt.plot(hist_g, color='g')
        plt.plot(hist_r, color='r')
        plt.show()
        '''
    
    JS_b_list = []
    JS_g_list = []
    JS_r_list = []
    
    for i in range(len(hist_b_list)):
        if i+dilation > len(hist_b_list)-1:
            break
        hist_b_img1 = hist_b_list[i]
        hist_b_img2 = hist_b_list[i+dilation]     
        JS_b = JS_divergence(hist_b_img1, hist_b_img2)
        JS_b_list.append(JS_b)
        
        hist_g_img1 = hist_g_list[i]
        hist_g_img2 = hist_g_list[i+dilation]     
        JS_g = JS_divergence(hist_g_img1, hist_g_img2)
        JS_g_list.append(JS_g)
        
        hist_r_img1 = hist_r_list[i]
        hist_r_img2 = hist_r_list[i+dilation]     
        JS_r = JS_divergence(hist_r_img1, hist_r_img2)
        JS_r_list.append(JS_r)
        
        '''
        plt.subplot(1,2,1)
        plt.imshow(img_in[:,:,[2,1,0]])
        plt.subplot(1,2,2)
        plt.plot(hist_b_img1)
        plt.plot(hist_b_img2)
        plt.show()
        '''
        
        
    return JS_b_list, JS_g_list, JS_r_list


def draw_plot(y,  fig, marker, color, label):
    num_seq = len(y)
    if color != None:
        fig.scatter(np.arange(num_seq), y, marker=marker, color=color, label=label)
    else:
        fig.scatter(np.arange(num_seq), y, marker=marker, label=label)    
    fig.plot(np.arange(num_seq), y, linewidth=1, color=color)
    
    return fig

def draw_plot2(y,  fig, label):
    num_seq = len(y)

    fig.scatter(np.arange(num_seq), y, label=label)    
    fig.plot(np.arange(num_seq), y, linewidth=1)
    
    return fig    




def calculate_folders(input_folder, name, dilation=1):
    input_folder_list = os.listdir(input_folder)
    input_folder_list.sort()
    input_folder_list = [folder for folder in input_folder_list if os.path.isdir(os.path.join(input_folder, folder))]
    #input_folder_list = ['dance-twirl']
    print('##### {} #####'.format(name))
    JS_b_mean_list, JS_g_mean_list, JS_r_mean_list = [], [], []   # record mean JS
    JS_b_dict, JS_g_dict, JS_r_dict = {}, {}, {}    # record every JS sequence of each folder
    for i, folder in enumerate(input_folder_list):
        GT_folder = os.path.join(input_folder, folder)
        JS_b_list, JS_g_list, JS_r_list = compute_JS_bgr(GT_folder, dilation)
        JS_b_mean_list.append(np.mean(JS_b_list))
        JS_g_mean_list.append(np.mean(JS_g_list))
        JS_r_mean_list.append(np.mean(JS_r_list))
        
        JS_b_dict[str(i)] = JS_b_list
        JS_g_dict[str(i)] = JS_g_list
        JS_r_dict[str(i)] = JS_r_list
        
        
        #print('Folder: {}  AGV_JS_B: {:.6f}  AVG_JS_G: {:.6f}  AVG_JS_R: {:.6f}'.format(folder, float(np.mean(JS_b_list)), float(np.mean(JS_g_list)), float(np.mean(JS_r_list))))
        
    print('############ Sumarry ############')
    print('[{}] Total folders: {}  AGV_JS_B: {:.6f}  AVG_JS_G: {:.6f}  AVG_JS_R: {:.6f}'.format(name, len(input_folder_list), float(np.mean(JS_b_mean_list)), float(np.mean(JS_g_mean_list)), float(np.mean(JS_r_mean_list))))
    
    CDC = np.mean([float(np.mean(JS_b_mean_list)), float(np.mean(JS_g_mean_list)), float(np.mean(JS_r_mean_list))])
    print('Total AVG: {:.6f}'.format(CDC))
    print('#################################')
    
    
    return JS_b_mean_list, JS_g_mean_list, JS_r_mean_list, JS_b_dict, JS_g_dict, JS_r_dict, CDC

def calculate_folders_multiple(input_folder, name, dilation=[1,2,4], weight=[1/3, 1/3, 1/3]):
    input_folder_list = os.listdir(input_folder)
    input_folder_list.sort()
    input_folder_list = [folder for folder in input_folder_list if os.path.isdir(os.path.join(input_folder, folder))]
    #input_folder_list = ['blackswan', 'car-roundabout', 'dance-twirl', 'goat']
    print('##### {} #####'.format(name))
    JS_b_mean_list, JS_g_mean_list, JS_r_mean_list = [], [], []   # record mean JS
    JS_b_dict, JS_g_dict, JS_r_dict = {}, {}, {}    # record every JS sequence of each folder
    for i, folder in enumerate(input_folder_list):
        GT_folder = os.path.join(input_folder, folder)
        mean_b, mean_g, mean_r = 0, 0, 0
        JS_b_list, JS_g_list, JS_r_list = [], [], []
        for d, w in zip(dilation, weight):
            JS_b_list_one, JS_g_list_one, JS_r_list_one = compute_JS_bgr(GT_folder, d)
            mean_b += w*np.mean(JS_b_list_one)
            mean_g += w*np.mean(JS_g_list_one)
            mean_r += w*np.mean(JS_r_list_one)
        JS_b_list.append(mean_b)
        JS_g_list.append(mean_g)
        JS_r_list.append(mean_r)
            
        
        JS_b_mean_list.append(mean_b)
        JS_g_mean_list.append(mean_g)
        JS_r_mean_list.append(mean_r)
        
        JS_b_dict[str(i)] = {}
        JS_g_dict[str(i)] = {}
        JS_r_dict[str(i)] = {}
        
        
        print('Folder: {}  AGV_JS_B: {:.6f}  AVG_JS_G: {:.6f}  AVG_JS_R: {:.6f}'.format(folder, float(np.mean(JS_b_list)), float(np.mean(JS_g_list)), float(np.mean(JS_r_list))))
        
    print('############ Sumarry ############')
    print('[{}] Total folders: {}  AGV_JS_B: {:.6f}  AVG_JS_G: {:.6f}  AVG_JS_R: {:.6f}'.format(name, len(input_folder_list), float(np.mean(JS_b_mean_list)), float(np.mean(JS_g_mean_list)), float(np.mean(JS_r_mean_list))))
    CDC = np.mean([float(np.mean(JS_b_mean_list)), float(np.mean(JS_g_mean_list)), float(np.mean(JS_r_mean_list))])
    print('Total AVG: {:.6f}'.format(CDC))
    print('#################################')
    
    
    return JS_b_mean_list, JS_g_mean_list, JS_r_mean_list, JS_b_dict, JS_g_dict, JS_r_dict, CDC

    
if __name__ == '__main__':
    input_folder = '/home/yhliu/video_color_lyh/results/G03_80000_DAVIS30_mod32_ensemble17_19/'
    GT_folder = '/data2/yhliu/DATA/DAVIS-2017-trainval-480p/DAVIS30_GT_mod32_new/'
    
    avg_psnr_l = calculate_psnr_folders(input_folder, GT_folder, interval_length=17)
    
    dilation = [1,2,4]
    weight = [1/3, 1/3, 1/3]
    JS_b_mean_list_1, JS_g_mean_list_1, JS_r_mean_list_1, JS_b_dict_1, JS_g_dict_1, JS_r_dict_1, CDC = calculate_folders_multiple(input_folder, input_folder, dilation=dilation, weight=weight)
    
