# %%
import re
import os
import cv2
import numpy as np
from osgeo import gdal
from glob import glob
# from libtiff import TIFF
from sklearn import tree
# import graphviz
from sklearn.ensemble import RandomForestClassifier 
from PIL import Image, ImageFilter
import skimage.filters.rank as sfr
from skimage.morphology import disk
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# %%
def write_images(bands, path):
    img_width = bands.shape[2]
    img_height = bands.shape[1]
    num_bands = bands.shape[0]
    datatype = gdal.GDT_UInt16

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
    if dataset is not None:
        for i in range(num_bands):
            dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")
    else:
        print("dataset is None!")

def crop_img(root='F:\\data\\landsat8\\BC\\', window_size=512):
    sences = os.listdir(root)
    for sence in sences:
        # print(sence)
        tifs = os.listdir(root + sence)
        valid_ext = ['.tif', '.TIF']
        tifs = [os.path.join(root, sence, tif) for tif in tifs if os.path.splitext(tif)[-1] in valid_ext]
        # tifs = ['B1', 'B10', 'B11', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'BQA']
        print("start read")
        tifs.sort()
        bandTifs = tifs[:11] # bandTifs = ['B1', 'B10', 'B11', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']
        
        # read mask
        maskImg = glob(root+sence+'/*.img')[0]
        mask_ds = gdal.Open(maskImg)
        M, N = mask_ds.RasterYSize, mask_ds.RasterXSize # M, N 分别是行和列

        # read bands, 特别耗时
        valid_band = [0, *range(3, 9), 10, 1, 2]
        num_of_bands = len(valid_band)

        print("get bands")
        # fill, shadow, land, thinCloud, cloud = [0, 64, 128, 192, 255]

        for i in range(5):
            while True:  # 根据mask是否为0过滤掉有填充值的图像
                x = np.random.randint(0, N-window_size)
                y = np.random.randint(0, M-window_size)
                label = mask_ds.ReadAsArray(x, y, window_size, window_size) # 参数必须是int型
                if np.sum(label == 0) == 0:
                    break
            img = []
            for band in valid_band:
                print(bandTifs[band])
                tif = gdal.Open(bandTifs[band]).ReadAsArray(x, y, window_size, window_size)
                img.append(tif)
            img = np.stack(img, 0)

            write_images(img, os.path.join('./image', sence + '_%05d.tiff'%(i)))
            cv2.imwrite(os.path.join('./label', sence + '_%05d.png'%(i)), np.uint8(label))

#%%
def feature_name(num_bands=10):
    all_bands = ['Coastal', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Cirrus', 'TIRS1', 'TIRS2']
    bands_name = all_bands[:num_bands]
    bands_comb = bands_name.copy()

    def add_frac(nume, denomi):
        return '$\\frac{\mathrm{' + nume + '}}{\mathrm{' + denomi + '}}$'

    for i in range(num_bands):
        for j in range(i+1, num_bands):
            numerator = bands_name[i] + '-' + bands_name[j]
            denominator = bands_name[i] + '+' + bands_name[j]
            bands_comb.append(add_frac(numerator, denominator))
    for i in range(num_bands):
        for j in range(i+1, num_bands):
            numerator = bands_name[i]
            denominator = bands_name[j]
            bands_comb.append(add_frac(numerator, denominator))
    
    bands_comb = np.array(bands_comb)
    return bands_comb


def make_feature(x):
    n_samples, n_features = x.shape
    new_feature = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            tmp = (x[:, i] - x[:, j]) / (x[:, i] + x[:, j])
            new_feature.append(np.where(np.isnan(tmp), 0, tmp))
    for i in range(n_features):
        for j in range(i+1, n_features):
            new_feature.append(x[:, i] / x[:, j])
    # for i in range(n_features):
    #     for j in range(i+1, n_features):
    #         new_feature.append(x[:, i]*x[:, i] / x[:, j])

    new_feature = np.stack(new_feature, 1)
    data_x = np.concatenate((x, new_feature), 1)
    data_x = np.where(data_x == np.inf, 0, data_x)
    data_x = np.nan_to_num(data_x)
    return data_x


def make_feature_(img):
    c, h, w = img.shape
    img_ = np.reshape(img, (c, -1))
    spect_feature = make_feature(img_.T)
    return spect_feature
    # return np.concatenate((spect_feature, otsu), 1)


def num2latex(feature_set=None):
    fn = feature_name()
    num2name_dict = dict(zip(range(len(fn)), fn))
    latex = []
    if feature_set is None:
        feature_set = list(range(len(fn))) 
    for ftr_num in feature_set:
        latex.append(num2name_dict[ftr_num])
    return ','.join(latex)

def num2latex2():
    for i in range(10):
        print('{}-{} & '.format(i*10, i*10+9), end='')
        print(num2latex(range(i*10, (i+1)*10)), end='\\\\\n')


#%%
def feature_selection():
    with open('log/vi.txt', 'w') as f:
        for k, v in img_dic.items():
            # if k in ['barren', 'forest', 'grass-crops', 'Shrubland']:
            #     continue
            print(k)
            f.write(k+'\n')
            data_x, data_y = [], []

            for img_path in v[::5]:
                img = gdal.Open(img_path).ReadAsArray() # [bands, H, W]
                bands = np.transpose(img[:num_bands].reshape((num_bands, -1)), (1, 0))
                bands = bands[::sample_rate]
                data = make_feature(bands)
                data_x.append(data)

                label = cv2.imread('./label'+img_path[7:-4]+'png', 0)
                label = label.flatten()[::sample_rate]
                data_y.append(label)

            data_x = np.vstack(data_x)
            data_y = np.hstack(data_y)
            data_y = np.where(data_y > 128, 1, 0)
            # data_x = make_feature(data_x.astype(np.float64))
            # clf = tree.DecisionTreeClassifier(max_depth=4)
            imps = []
            oob_score = []
            clf = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=True)
            for _ in range(10):
                clf.fit(data_x, data_y)
                imp = clf.feature_importances_
                oob_score.append(clf.oob_score_)
                imps.append(imp)
            imps = np.vstack(imps)
            imps_m, imps_v = np.mean(imps, 0), np.var(imps, 0)

            oob_score = np.mean(np.vstack(oob_score))
            print('oob score is {}'.format(oob_score))
            f.write('oob score is {} \n'.format(oob_score))

            # draw VI mean and var
            # fig = plt.figure()
            # ax1 = fig.add_subplot(111)
            # l1, = ax1.bar(range(len(imps_v)), imps_v, fc='coral', label='var')
            # ax1.set_ylabel('var')
            # ax2 = ax1.twinx()
            # l2, = ax2.plot(range(len(imps_m)), imps_m, 'g--', marker='^', label='mean')
            # ax2.set_ylabel('mean')
            # plt.legend(handles=[l1, l2])
            # plt.savefig(save_path+k+'.png', bbox_inches='tight')
            # plt.show()

            imps_sort = np.sort(imps_m)[::-1] # descend
            threds =  imps_sort[0] / 10
            num_threds = np.sum(imps_sort > threds)
            print('number of feature set is {}'.format(num_threds))

            # select feature subset using VI rank
            args = np.argsort(imp)[::-1].tolist()
            f.write('feature rank is {} \n'.format(args))
            f.write('feature VI is {} \n'.format(imps_m))
            args_n = args[:num_threds]
            f.write('feature rank is {} \n'.format(args_n))
            print('feature rank is {}'.format(args_n))

            clf = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=False)
            scores = cross_val_score(clf, data_x[:, :num_bands], data_y, cv=5)
            ora_score = scores.mean()
            clf = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=False)
            scores = cross_val_score(clf, data_x[:, args], data_y, cv=5)
            cmb_score = scores.mean()
            print('ora_score : {}, cmb_score : {}'.format(ora_score, cmb_score))
            f.write('ora_score : {}, cmb_score : {} \n'.format(ora_score, cmb_score))

            print('begin select feature...')
            clf = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=False)
            args = args_n.copy()
            feature_set = [args[0]]
            args.pop(0)
            scores = cross_val_score(clf, data_x[:, feature_set].reshape(-1, 1), data_y, cv=5)
            feature_score = [scores.mean()]
            print('No.1 feature acc is {}'.format(feature_score))

            while True:
                cur_scores = []
                for idx, ftr in enumerate(args):
                    clf = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=False)
                    scores = cross_val_score(clf, data_x[:, feature_set + [ftr]], data_y, cv=5)
                    score = scores.mean()
                    cur_scores.append(score)
                
                idx, score = np.argmax(cur_scores), np.max(cur_scores)
                if score - feature_score[-1] > 1e-4:
                    feature_score.append(score)
                    feature_set.append(args[idx])
                    args.pop(idx)
                else:
                    break
                print(feature_score)                    
                
            print('feature set score is {}'.format(feature_score))
            print('feature set is {}'.format(feature_set))
            f.write('feature set score is {} \n'.format(feature_score))
            f.write('feature set is {} \n'.format(feature_set))

            # plot sorted VI with selected feature
            fig = plt.figure()
            plt.plot(range(len(imps_sort)), imps_sort, 'g--', marker='.', zorder=1)
            plt.hlines(threds, 0, len(imps_sort)-1, color='blue', linestyles='dashed')
            plt.ylabel('Variable importance')
            args_new = [np.where(np.array(args_n) == a)[0][0] for a in feature_set]
            plt.scatter(args_new, imps_sort[args_new], c='red', marker='.', zorder=2)
            plt.savefig(save_path+k+'_sort.png')
            # plt.show()

            # plot acc curve with features added gradually
            fig = plt.figure()
            plt.plot(range(len(feature_score)), feature_score, 'g-', marker='^')
            plt.hlines(ora_score, 0, len(feature_score)-1, linestyles='dashed', label='ACC')
            plt.hlines(cmb_score, 0, len(feature_score)-1, linestyles='dashed', color='red', label='ACC1')
            plt.legend()
            plt.savefig(save_path+k+'_score.png', bbox_inches='tight')
            # plt.show()


#%%
def plot_noncloud_spectral():
    plt.rcParams['figure.figsize'] = (9.0, 6.0)
    plt.figure()
    for k, v in img_dic.items():
        print(k)
        noncloud = []

        for img_path in v[::2]:
            img = gdal.Open(img_path).ReadAsArray() # [bands, H, W]
            bands = np.transpose(img[:num_bands].reshape((num_bands, -1)), (1, 0))
            bands = bands[::sample_rate]

            label = cv2.imread('./label'+img_path[7:-4]+'png', 0)
            label = label.flatten()[::sample_rate]
            
            idx = np.where(label==128)
            non = bands[idx]
            idx = np.argsort(non[:, 0])[int(len(non)/3) : int(len(non)/3*2)]
            non = non[idx]
            noncloud.append(bands[idx])

        noncloud = np.vstack(noncloud)
        plt.plot(np.mean(noncloud, 0), marker='o', label=k)
        plt.ylabel('DN')
        plt.xlabel('Bands')
        plt.legend()
        plt.savefig(save_path+'all_spectral.png', bbox_inches='tight')
        plt.show()


#%%
def clf_rf():
    for k, v in img_dic.items():
        print(k)
        data_x, data_y = [], []

        for img_path in v[1::2]:
            img = gdal.Open(img_path).ReadAsArray() # [bands, H, W]
            data = np.transpose(img[:num_bands].reshape((num_bands, -1)), (1, 0))
            data_x.append(data[::sample_rate])

            label = cv2.imread('./label'+img_path[7:-4]+'png', 0)
            data_y.append(label.flatten()[::sample_rate])

        data_x = np.vstack(data_x)
        data_x = np.where(data_x == np.inf, 0, data_x)
        data_x = np.nan_to_num(data_x)
        data_y = np.hstack(data_y)
        data_y = np.where(data_y > 128, 1, 0)
        # data_x = make_feature(data_x.astype(np.float64))
        # clf = tree.DecisionTreeClassifier(max_depth=4)
        imps = []
        oob_score = []
        clf = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=True)
        for i in range(1):
            clf.fit(data_x, data_y)
            imp = clf.feature_importances_
            oob_score.append(clf.oob_score_)
            imps.append(imp)
        imps = np.vstack(imps)
        imps_m, imps_v = np.mean(imps, 0), np.var(imps, 0)

        oob_score = np.mean(np.vstack(oob_score))
        print('oob score is {}'.format(oob_score))

        fig = plt.figure()
        plt.plot(range(len(feature_score)), feature_score, 'g-', marker='^')
        plt.hlines(ora_score, 0, len(feature_score)-1, linestyles='dashed', label='ACC')
        plt.hlines(oob_score, 0, len(feature_score)-1, linestyles='dashed', color='red', label='ACC1')
        plt.legend()
        plt.savefig(save_path+k+'_score.png', bbox_inches='tight')
        plt.show()

#%%
def draw_result():
    f = open('log/vi.txt', 'r')
    lines = f.readlines()[4::5]
    for k, v in img_dic.items():
        if k != 'Snow-Ice':
            continue
        print(k)
        feature_set = lines[0].strip().split('[')[-1][:-1]
        feature_set = list(map(int, feature_set.split(',')))
        lines.pop(0)

        data_x, data_y = [], []
        for img_path in v[::2]:
            img = gdal.Open(img_path).ReadAsArray() # [bands, H, W]
            data = np.transpose(img[:num_bands].reshape((num_bands, -1)), (1, 0))
            data = make_feature(data[::sample_rate])
            data_x.append(data[:, feature_set])

            label = cv2.imread('./label'+img_path[7:-4]+'png', 0)
            data_y.append(label.flatten()[::sample_rate])

        data_x = np.vstack(data_x)
        data_y = np.hstack(data_y)
        data_y = np.where(data_y > 128, 1, 0)

        imps = []
        oob_score = []
        clf = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=False)
        clf.fit(data_x, data_y)

        for img_path in v[1::2]:
            img = gdal.Open(img_path).ReadAsArray() # [bands, H, W]
            _, H, W = img.shape
            data = np.transpose(img[:num_bands].reshape((num_bands, -1)), (1, 0))
            data = make_feature(data)

            y_pred = clf.predict(data[:, feature_set])
            y_pred = np.reshape(y_pred, (H, W))
            label = cv2.imread('./label'+img_path[7:-4]+'png', 0)
            cv2.imwrite('./rf_clf/'+k+'_'+img_path[8:-4]+'_color.png', np.transpose(img[[9, 7, 5]]*5e-5*255,  (1, 2, 0)))
            cv2.imwrite('./rf_clf/'+k+'_'+img_path[8:-4]+'_pred.png', y_pred*255)
            cv2.imwrite('./rf_clf/'+k+'_'+img_path[8:-4]+'_label.png', label)

        
# %%
if __name__ == '__main__':
    if not os.path.exists('./image'):
        crop_img()

    save_path = 'pics/'
    keys = ['barren', 'forest', 'grass-crops', 'Shrubland', 'Snow-Ice', 'Urban', 'Water', 'Wetlands']
    urls = []
    with open('./dataLoad/urls.txt') as f:
        for line in f.readlines():
            urls.append(re.split('[/_.]', line)[-3])
    values = [urls[i*12:(i+1)*12] for i in range(8)]
    dic = dict(zip(keys, values))

    num_bands = 10
    bands_comb = feature_name(num_bands)
    count = np.zeros((len(bands_comb)), np.int64)

    img_list = []
    for sences in values:
        tmp = []
        for sence in sences:
            tmp.extend(glob('./image/'+sence+'*'))
        img_list.append(tmp)
    img_dic = dict(zip(keys, img_list))
    sample_rate = 40

    feature_selection()
    # draw_result()
    # plot_noncloud_spectral()

    # k = 'Snow-Ice'
    # v = img_dic[k]
    # weight = np.array([5, 5, 5])
    # weight = weight[:, None, None]
    # for img_path in v[21:50:2]:
    #     print(img_path)
    #     img = gdal.Open(img_path).ReadAsArray() 
    #     data = np.transpose(img[:num_bands].reshape((num_bands, -1)), (1, 0))
    #     cv2.imwrite('./rf_clf/'+k+'_'+img_path[8:-4]+'_color.png', np.transpose(img[[9, 8, 1]]*weight*1e-5*255,  (1, 2, 0)))


#%%

# %%
