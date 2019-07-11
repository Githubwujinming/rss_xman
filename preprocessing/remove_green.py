import gdal
import os
import cv2
import numpy as np
def get_green(file_name,thread):
    dataset = gdal.Open(file_name)
    if dataset == None:
        print('the file does not exits!!')
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount

    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # B,R,G,NIR格式
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()

    im_buleBand = im_data[0, 0:im_height, 0:im_width]
    im_greenBand = im_data[1, 0:im_height, 0:im_width]
    im_redBand = im_data[2, 0:im_height, 0:im_width]
    im_nirBand = im_data[3, 0:im_height, 0:im_width]

    nir_m_red = im_nirBand.astype(np.float32) - im_redBand.astype(np.float32)
    nir_p_red = im_nirBand.astype(np.float32) + im_redBand.astype(np.float32)
    nir_p_red = np.where(nir_p_red == 0, 0.0001, nir_p_red)
    green = nir_m_red / nir_p_red

    green_avg = np.mean(green)
    green_mask = np.where(green < thread, 1, 0)
    im_redBand_green_mask = np.array(im_redBand) / 1100 * green_mask
    im_greenBand_green_mask = np.array(im_greenBand) / 1100 * green_mask
    im_buleBand_green_mask = np.array(im_buleBand) / 1100 * green_mask
    im_nirBand_green_mask = np.array(im_nirBand) / 1100 * green_mask
    out_png = cv2.merge([im_buleBand_green_mask, im_greenBand_green_mask, im_redBand_green_mask])
    # out_png = cv2.merge([im_buleBand, im_greenBand, im_redBand])
    out_png = (out_png) * 255
    # cv2.imwrite(file_name.split('/')[-1].replace('.tif','.png'),out_png)
    return out_png, green_mask


# data_path = '/home/ubuntu/PycharmProjects/datasets/tif'
# tif_2017 = os.path.join(data_path,'val','img_2017/image_2017_960_960_10.tif')
# tif_2018 = os.path.join(data_path,'val','img_2018/image_2018_960_960_10.tif')
# _, green_2017 = get_green(tif_2017)
# _, green_2018 = get_green(tif_2018)
# change = green_2018 - green_2017
# change *= 255
# cv2.imwrite('../change/{}'.format(tif_2018.split('/')[-1].replace('image','mask_2017')),change)