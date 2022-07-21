import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats


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
        
        
        print('Folder: {}  AGV_JS_B: {:.6f}  AVG_JS_G: {:.6f}  AVG_JS_R: {:.6f}'.format(folder, float(np.mean(JS_b_list)), float(np.mean(JS_g_list)), float(np.mean(JS_r_list))))
        
    print('############ Sumarry ############')
    print('[{}] Total folders: {}  AGV_JS_B: {:.6f}  AVG_JS_G: {:.6f}  AVG_JS_R: {:.6f}'.format(name, len(input_folder_list), float(np.mean(JS_b_mean_list)), float(np.mean(JS_g_mean_list)), float(np.mean(JS_r_mean_list))))
    print('#################################')
    
    
    return JS_b_mean_list, JS_g_mean_list, JS_r_mean_list, JS_b_dict, JS_g_dict, JS_r_dict

    

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
    
