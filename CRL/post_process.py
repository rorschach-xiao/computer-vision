import tensorflow
import cv2
import numpy as np
import matplotlib.pyplot as plt
from prepfm import *
THRESHOLD = 2
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540
#read image

def disp_post_precess(img_l_disp,img_r_disp,index):
#LRC(left right consistency) check
    occlusion_region = np.ones([IMAGE_HEIGHT,IMAGE_WIDTH])
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            d1 = -img_l_disp[i][j]
            if (j-d1<0):
                d2 = 0
            else:
                d2 = img_r_disp[i][j-d1]
            if np.abs(d1-d2)>THRESHOLD:
                occlusion_region[i][j]=0
    #plt.imshow(occlusion_region,cmap='gray')
    #plt.show()
#Occlusion Filling
    pl=-1
    pr=-1
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            if(occlusion_region[i][j]==0):#(i,j)为遮挡点
                #向左搜寻第一个非遮挡点
                for k in range(j-1,-1,-1):
                    if(occlusion_region[i][k]==1):
                        pl = k
                        break
             #向右搜寻第一个非遮挡点
                for k in range(j,IMAGE_WIDTH):
                    if(occlusion_region[i][k]==1):
                        pr = k
                        break
                if(pl == -1 and pr != -1):
                    real_disp = -img_l_disp[i][pr]
                if(pr == -1 and pl != -1):
                    real_disp = -img_l_disp[i][pl]
                if(pl != -1 and pr != -1):
                    real_disp = min(-img_l_disp[i][pl],-img_l_disp[i][pr])
                img_l_disp[i][j] = -real_disp
    plt.imshow(-img_l_disp,cmap='gray')
    plt.show()

#Median Filtering
    img_l_disp_median = cv2.medianBlur(img_l_disp,5)
    for i in range(20):
        img_l_disp_median = cv2.medianBlur(img_l_disp_median,5)
    plt.imshow(-img_l_disp_median,cmap='gray')
    plt.show()
    writePFM('left_0'+str(index)+'.pfm',-img_l_disp_median.astype(np.float32))

def main():
    for i in range(500):
        #read image
        fs_l = cv2.FileStorage("left_disparity_"+str(i)+".xml", cv2.FileStorage_READ)
        fs_r = cv2.FileStorage("right_disparity_"+str(i)+".xml", cv2.FileStorage_READ)
        img_r_disp = fs_r.getNode('right_disparity_'+str(i)).mat()
        img_l_disp = fs_l.getNode('left_disparity_'+str(i)).mat()
        disp_post_precess(img_l_disp,img_r_disp,i)

if __name__ == '__main__':
    #main()
    pass











