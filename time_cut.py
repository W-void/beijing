# %%
import gdal
import numpy as np
import os
import cv2 as cv


# %%
#定义读取函数
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_lines = dataset.RasterYSize    #栅格矩阵的行数
    im_samples = dataset.RasterXSize  #栅格矩阵的列数
    im_bands = dataset.RasterCount    #波段数
    im_geotrans = dataset.GetGeoTransform()     #获取仿射矩阵信息
    im_proj = dataset.GetProjection()     #获取投影信息
    im_data = dataset.ReadAsArray()
    return im_lines, im_samples, im_bands, im_geotrans, im_proj, im_data

#定义保存函数
def writeTiff(im_data, im_lines, im_samples, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
        #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_samples, im_lines, im_bands, datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
        for m in range(im_bands):
            dataset.GetRasterBand(m+1).WriteArray(im_data[m, :,:])
    del dataset

def coor2geo(x_pixel, y_pixel, geotran):
    '''
    geotran 六个参数：
    0：图像左上角的X坐标；
    1：图像东西方向分辨率；
    2：旋转角度，如果图像北方朝上，该值为0；
    3：图像左上角的Y坐标；
    4：旋转角度，如果图像北方朝上，该值为0；
    5：图像南北方向分辨率；
    '''
    px = geotran[0] + x_pixel * geotran[1] + y_pixel * geotran[2]
    py = geotran[3] + x_pixel * geotran[4] + y_pixel * geotran[5]
    return px, py

def get_xy_bias(mask_geo, oral_geo):
    assert mask_geo[1] == oral_geo[1]
    assert mask_geo[5] == oral_geo[5]
    px = (mask_geo[0] - oral_geo[0]) / mask_geo[1]
    py = (mask_geo[3] - oral_geo[3]) / mask_geo[5]
    return int(px), int(py)

def find_common_name(tif_root='E:/beijing/bjtif/', qa_root='E:/beijing/bjqa/'):
    tif_list = os.listdir(tif_root)
    qa_list = os.listdir(qa_root)

    tif_list.sort() # 升序排列
    qa_list.sort()
    i, j = 0, 0
    len_i, len_j = len(qa_list), len(tif_list)
    same_tif_list, same_qa_list = [], []
    while i < len_i or j < len_j:
        qa = qa_list[i][:25]
        tif = tif_list[j][:25]
        if qa == tif:
            same_tif_list.append(os.path.join(tif_root, tif_list[j]))
            same_qa_list.append(os.path.join(qa_root, qa_list[i]))
            i += 1
            j += 1
        elif qa > tif:
            j += 1
        else:
            i += 1
    common_list = list(zip(same_tif_list, same_qa_list))
    print('common_list length: %d' % len(common_list))
    return common_list

# %%
def crop_img_from_mask(tif_name, qa_name, ph=0, pw=0, win_size=400):
    bj_mask = 'E:/beijing/test.tif'
    mask_ds = gdal.Open(bj_mask)
    mask_geotran = mask_ds.GetGeoTransform()

    tif_ds = gdal.Open(tif_name)
    qa_ds = gdal.Open(qa_name)
    geotrans = tif_ds.GetGeoTransform()
    _geotrans = qa_ds.GetGeoTransform()
    assert geotrans == _geotrans
    dx, dy = get_xy_bias(mask_geotran, geotrans) # dx：东西方向；dy：南北方向
    assert (dx >= 0 and dy >= 0)
    x_s, y_s = dx + pw, dy + ph
    x_e, y_e = x_s + win_size, y_s + win_size
    tif = tif_ds.ReadAsArray(x_s, y_s, win_size, win_size) # [channel, win_size, win_size]
    qa = qa_ds.ReadAsArray(x_s, y_s, win_size, win_size) # [win_size, win_size]
    cloud = np.where(np.bitwise_and(qa, (1<<5)), 1, 0)  # 第5位是云标志位
    return tif, cloud
    
def main(ph=1400, pw=3200, win_size=100):
    bj_mask = 'E:/beijing/test.tif'
    mask_ds = gdal.Open(bj_mask)
    mask_geotran = mask_ds.GetGeoTransform()
    proj = mask_ds.GetProjection() 
    im_lines = mask_ds.RasterYSize    #栅格矩阵的行数
    im_samples = mask_ds.RasterXSize    #栅格矩阵的列数
    print('im_lines : %d, im_samples : %d' % (im_lines, im_samples))
    assert (ph + win_size <= im_lines) and (pw + win_size <= im_samples)

    data_qa, data_tif, cloud_name = [], [], []
    for k, (tif_name, qa_name) in enumerate(find_common_name()):
        tmp_name = tif_name.split('/')[-1]
        if not tmp_name.startswith('LC08_L1TP_123032') or not tmp_name.endswith('T1_sr_band1-7.tif'):
            continue
        cloud_name.append(tmp_name.split('_')[3])
        tif, cloud = crop_img_from_mask(tif_name, qa_name, ph, pw, win_size)
        data_tif.append(tif)
        data_qa.append(cloud)
    print("Done~")
    return cloud_name, np.stack(data_tif, 0), np.stack(data_qa)

# %%
if __name__ == '__main__':
    main()
