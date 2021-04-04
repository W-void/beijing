'''
Author: wang shuli
Date: 2020-12-20 20:31:02
LastEditTime: 2021-04-04 14:04:33
LastEditors: your name
Description: 
'''
#%%
import os
import numpy as np
from osgeo import gdal
from glob import glob
import matplotlib.pyplot as plt
import time_cut
import torch
from PIL import Image
import cv2

#%%
def time_plot(spectrum, flag, sort=False):
    '''
    sprectrum and flag should be 1-D vector with same lengh.
    '''
    if sort:
        idx = np.argsort(spectrum)
        idx = idx[2:-2]
        spectrum = spectrum[idx]
        flag = flag[idx]
    x_c, x_n = np.where(flag == 1)[0], np.where(flag == 0)[0] # 1 is cloud, 0 is non-cloud
    y_c, y_n = spectrum[x_c], spectrum[x_n]

    plt.figure()
    l1 = plt.scatter(x_c, y_c, marker='o', c='red')
    l2 = plt.scatter(x_n, y_n, marker='*', c='green')
    plt.legend(handles=[l1, l2], labels=['cloud', 'non-cloud'])
    plt.plot(spectrum)

#%%
def my_function(arr):
    normal = np.mean(arr[20:40], 0)
    idx = np.where(arr[40:] > normal * 5)[0][0] + 40 + 2
    return idx 

def my_avg_pooling(imgs):
    pooling = torch.nn.AvgPool2d(2, stride=2)
    imgs = pooling(torch.Tensor(imgs))
    imgs = imgs.numpy()
    return imgs

def my_max_pooling(imgs):
    pooling = torch.nn.MaxPool2d(2, stride=2)
    imgs = pooling(torch.Tensor(imgs))
    imgs = imgs.numpy()
    return imgs

def my_upsample(imgs, scale):
    upsample = torch.nn.Upsample(scale_factor=scale, mode='nearest')
    imgs = upsample(torch.Tensor(imgs[None, None]))
    imgs = imgs.numpy().squeeze()
    return imgs.astype('int64')

def plot_idx(idx, name, show=False):
    plt.figure()
    plt.imshow(idx)
    plt.colorbar()
    plt.savefig(name, bbox_inches='tight')
    if show:
        plt.show()
        
# def get_inflex(sorted):
#     dif = sorted[1:] - sorted[:-1]
#     dif3 = sorted[3:] - sorted[:-3]
#     # normal = np.mean(dif[20:40])
#     # idx = np.where(dif[40:] > normal * 5)[0][0] + 40 + 2
#     idx = np.where((dif[40:-2] > 0.02) | (dif[40:-2] > 0.01) & (dif3[40:] > 0.05))[0][0] + 40
#     return idx, sorted[idx]

def get_inflex(sorted):
    length = len(sorted)
    x0, y0 = 10, sorted[10]
    x1, y1 = length-10, sorted[-10]
    xx = np.arange(11, length-10)
    yy = sorted[11:-10]
    d = (xx-x1)*(y1-y0) + (yy-y1)*(x0-x1)
    idx = np.argmax(d) + x0
    return idx, sorted[idx]

def sort_and_plot(band, name=''):
    plt.figure()
    band = np.sort(band, 0)
    plt.plot(band)
    x_, y_ = get_inflex(band)
    plt.axhline(y=y_, ls=":", c='r')
    plt.axvline(x=x_, ls=":", c='r')
    plt.savefig(save_path+k+'_band_' + name +'.png', bbox_inches='tight')
    return band

def find_local_max(sorted):
    sorted_ = sorted[1:] - sorted[:-1]
    head = 10
    ans = [np.argmax(sorted_[:head])]
    tmp_max = sorted_[:head].max()
    for idx in range(head, len(sorted_)-10):
        grad = sorted_[idx]
        if  (grad > tmp_max * 1.2) and (idx > ans[-1] + 3):
            ans.append(idx)
            tmp_max = grad
    ans = ans + [len(sorted) - 2]
    return ans

def new_cloud_detection_function(sorted):
    inflexes = find_local_max(sorted)
    ave_grds = []
    for s, e in zip(inflexes[:-1], inflexes[1:]):
        ave_grd = (sorted[e] - sorted[s+1]) / (e-s-1)
        ave_grds.append(ave_grd)      
    # print('ave_grds is {}'.format(ave_grds))

    thres = np.where(ave_grds > 0.4 * ave_grds[-1])[0][0]
    inflex = inflexes[thres]
    return np.array(ave_grds), inflex

#%%
def plot_sort_and_segment(band, c):
    fig = plt.figure()
    ave_grds, _ = new_cloud_detection_function(band)

    band_ = band[1:] - band[:-1]
    ax1 = fig.add_subplot(111)
    # l1, = ax1.plot(np.array(band_) * 5, linewidth=1.0, label='diff', zorder=1)
    ax1.set_ylabel('diff')
    ax2 = ax1 # .twinx()  # this is the important function
    l2, = ax2.plot(band[1:], 'g--', marker='.', label='band', zorder=2)

    idx = find_local_max(band)
    # ax2.scatter(idx, band[1:][idx], color='red', marker='.', zorder=3)
    for s, e in zip(idx[:-1], idx[1:]):
        ax2.plot([s, e-1], [band[s+1], band[e]], 'r', marker='.', zorder=3)
    ax2.scatter(idx[:-1], ave_grds*20, color='black', marker='*', zorder=4)
    
    _, inflex = new_cloud_detection_function(band)
    x_, y_ = inflex, band[inflex]
    plt.axhline(y=y_, ls=":", c='r')
    plt.axvline(x=x_-1, ls=":", c='r')
    
    ax2.set_ylabel('reflence')
    # plt.legend(handles=[l1, l2])
    plt.savefig(save_path+k+'_sort_{}.png'.format(c), bbox_inches='tight')
    # plt.show()


def plot_time_sort_and_hvline(band, sorted, c, name=None):
    plt.figure()
    plt.scatter(range(len(band)), band, s=5, alpha=0.8)
    plt.plot(sorted, 'g--', marker='.')
    
    # x_, y_ = get_inflex(sorted)
    _, inflex = new_cloud_detection_function(sorted)
    x_, y_ = inflex, sorted[inflex]
    plt.axhline(y=y_, ls=":", c='r')
    plt.axvline(x=x_, ls=":", c='r')
    plt.ylabel('{} reflence'.format(name))
    plt.savefig(save_path+k+'_band_{}.png'.format(c), bbox_inches='tight')
    # plt.show()
    
#%%
win_size = 256
save_path = 'pics_time/'
miyun_water = (3000-1450, 4400-1200, win_size)
city = (5000-1450, 3000-1300, win_size)
mountion = (2000-1450, 5000-1300, win_size)
name = ['city', 'mountion', 'water']
sample = [city, mountion, miyun_water]
bands_name = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
spec_idx = 1

for k, s in zip(name, sample):
    cloud_name, oral_imgs, oral_flags = time_cut.main(*s)
    print(oral_imgs.shape) # [time_seq, channel, H, W]
    print(oral_flags.shape)
    oral_imgs = oral_imgs * 1e-4
    
    T, C, H, W = oral_imgs.shape
    center = oral_imgs[:, :, int(H/2), int(W/2)]
    
    for c in range(C):
        band = center[:, c]
        band_sort = np.sort(band, 0)[2:-2]
        # plot_sort_and_segment(band_sort, c)
        # plot_time_sort_and_hvline(band, band_sort, c, bands_name[c])
    
    # sort_and_plot(np.mean(center[:, :4], 1), 'mean_0-3')
    # sort_and_plot(np.mean(center[:, 4:], 1), 'mean_4-6')
    # sort_and_plot(np.mean(center, 1), 'mean')

    # r_b = center[:, 3] / center[:, 1]
    # sort_and_plot(r_b, 'r_b')


    all_idx = []
    imgs, flags = oral_imgs.copy(), oral_flags.copy()
    
    for i in range(3):
        t, c, h, w = imgs.shape
        # if i == 0:
        #     plt.figure()
        #     plt.scatter(range(t), imgs[:, 1, h//2, w//2]*1e-5, s=10, c='g')
        #     plt.ylabel('Blue reflectance')
        #     plt.savefig(save_path+k+'_{}.png'.format(h))
        imgs_sort = np.sort(imgs, 0)

        blue = imgs_sort[:, spec_idx][2:-12]  # [time_seq, H, W]
        blue_ = blue[1:] - blue[:-1]
        blue_mean = np.mean(blue_.reshape(blue_.shape[0], -1), -1) # [time_seq]

        # idx, _ = np.apply_along_axis(get_inflex, 0, blue) # [H, W]
        _, idx = np.apply_along_axis(new_cloud_detection_function, 0, blue)
        idx = idx + 2
        idx = idx.astype('int64')
        all_idx.append(my_upsample(idx, 2**i))
        plot_idx(idx, save_path+k+'_idx_{}.png'.format(h), False)
        
        xx, yy = np.where(idx)
        thread = np.reshape(blue[idx.flatten(), xx, yy], (h, w))
        pred = np.where(imgs[:, spec_idx] > thread, 1, 0)
        acc = np.sum(pred == flags) / (t * h * w)
        print(acc)

        imgs,flags = my_avg_pooling(imgs), my_max_pooling(flags)

    imgs_sort = np.sort(oral_imgs, 0)
    blue = imgs_sort[:, spec_idx]
    
    idx_mean = np.mean(np.stack(all_idx, 0), 0).astype('int64')
    xx, yy = np.where(idx_mean)
    thread = np.reshape(blue[idx_mean.flatten(), xx, yy], (H, W))
    pred = np.where(oral_imgs[:, spec_idx] > thread, 1, 0) # [seq_len, H, W]
    acc = np.sum(pred == oral_flags) / (T * H * W)
    print(acc)

    plot_idx(idx_mean, save_path+k+'_idx.png', False)

    result_path = 'bjcloud/'
    for i, name in enumerate(cloud_name):  
        cv2.imwrite(result_path + k +'_' + name + '_pred.png', pred[i]*255)
        cv2.imwrite(result_path + k +'_' + name + '_qa.png', oral_flags[i]*255)
        cv2.imwrite(result_path + k +'_' + name + '_color.png', np.clip(np.transpose(oral_imgs[i, 1:4], (1, 2, 0)) * 5 * 255, 0, 255))

#%%
# import matplotlib.pyplot as plt 
# img = np.transpose(imgs[5, 3:0:-1], (1, 2, 0))
# plt.imshow(img*5e-4)
