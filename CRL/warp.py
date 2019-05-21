import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from prepfm import *
IMAGE_SIZE_Y = 540
IMAGE_SIZE_X = 960

#DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/medical data/image'
DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/frames_cleanpass/35mm_focallength/scene_forwards/fast'
#DISP_DIR = 'C:/Users/HK/PycharmProjects/CRL/disparity_1/medical/'
DISP_DIR = 'C:/Users/HK/PycharmProjects/CRL/disparity_1/car/'
SYN_IMAGE_DATA = 'C:/Users/HK/PycharmProjects/CRL/syn_image/car/'

right_images = os.listdir(DATA_DIR + '/right/')
right_images.sort(key = lambda x: int(x[:-4]))
disparitys = os.listdir(DISP_DIR)
disparitys.sort(key = lambda x : int(x[:-4]))

image_num = len(right_images)
def warp(image_right,disparity):
    img_left_s = np.zeros([1,IMAGE_SIZE_Y,IMAGE_SIZE_X,3])
    for i in range(1):
        for j in range(IMAGE_SIZE_Y):
            for k in range(IMAGE_SIZE_X):
                #左图映射到右图中坐标
                SrcX = k - disparity[i][j][k]
                if SrcX<0: # 如果该坐标落在图像外，则将左图中该点像素设为0
                    img_left_s[i][j][k] = 0
                else:
                #使用线性差值的方法生成坐标值
                    u = SrcX - int(SrcX) #右图虚拟像素点横坐标的小数部分
                    x1 = int(SrcX) #x1、x2分别为右图中虚拟像素点左右两个像素点的横坐标
                    if x1<IMAGE_SIZE_X-1:
                        x2 = x1+1
                    else:
                        x2 = x1
                    img_left_s[i][j][k] = (1 - u) * image_right[i][j][x1] + u * image_right[i][j][x2]

    return img_left_s
if __name__ == '__main__':
    for i in range(0,300):
        right = Image.open(DATA_DIR + '/right/' + right_images[i])
        right = np.reshape(right,(1,IMAGE_SIZE_Y,IMAGE_SIZE_X,3))
        disp,scale = load_pfm(DISP_DIR+disparitys[i])
        disp = np.reshape(disp, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X))
        disp = np.reshape(disp, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
        syn_img = warp(right,disp)
        result = np.squeeze(syn_img)
        #plt.imshow(result)
        #plt.show()
        plt.imsave(SYN_IMAGE_DATA+str(i)+'.png',result.astype(np.uint8))