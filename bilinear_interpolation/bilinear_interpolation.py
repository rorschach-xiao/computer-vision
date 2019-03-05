import numpy as np
from skimage import io

upsample_factor = 2  # 上采样两倍图像大小
def produce_image():
    size = 3
    x,y=np.ogrid[:size,:size]
    z=x+y
    #print(z)
    z=z[:,:,np.newaxis]
    #print(z)
    img=np.repeat(z,3,2)/12
    io.imshow(img,interpolation='none')
    io.show()
    return img

def bilinear_interpolation(image, factor):
    upsampling_image = np.zeros((image.shape[0] * factor, image.shape[1] * factor, image.shape[2]));
    image_size_X = image[0].shape[0]
    image_size_Y = image[0].shape[1]
    for k in range(3):
        for i in range(upsampling_image.shape[0]):
            for j in range(upsampling_image.shape[1]):
                # 映射到原图中的虚坐标
                SrcX = (i + 0.5) / factor - 0.5
                SrcY = (j + 0.5) / factor - 0.5
                if (SrcX > image_size_X - 1):
                    SrcX = image_size_X - 1
                if (SrcX < 0):
                    SrcX = 0
                if (SrcY > image_size_Y - 1):
                    SrcY = image_size_Y - 1
                if (SrcY < 0):
                    SrcY = 0
                # u,v分别为原图虚像素x,y坐标的小数部分
                u = SrcX - int(SrcX)
                v = SrcY - int(SrcY)
                # 虚像素周围的四个真实像素点坐标,其横纵坐标分别为(x1,y1).(x1,y2),(x2,y1),(x2,y2)
                x1 = int(SrcX)
                x2 = x1 + 1
                y1 = int(SrcY)
                y2 = y1 + 1
                # 当虚像素点位于源图像的右边界或者下边界上时
                if (SrcX == image_size_X - 1 and SrcY != image_size_Y - 1):
                    upsampling_image[i][j][k] = (1 - u) * (1 - v) * image[x1][y1][k] + (1 - u) * v * image[x1][y2][k]
                elif (SrcX != image_size_X - 1 and SrcY == image_size_Y - 1):
                    upsampling_image[i][j][k] = (1 - u) * (1 - v) * image[x1][y1][k] + (1 - v) * u * image[x2][y1][k]
                elif (SrcX != image_size_X - 1 and SrcY != image_size_Y - 1):
                    upsampling_image[i][j][k] = (1 - u) * (1 - v) * image[x1][y1][k] + (1 - u) * v * image[x1][y2][k] + (1 - v) * u * image[x2][y1][k] + u * v * image[x2][y2][k]
                else:
                    upsampling_image[i][j][k] = image[x1][y1][k]

    return upsampling_image

Img_i=produce_image()
up_img=bilinear_interpolation(Img_i,upsample_factor)
io.imshow(up_img,interpolation='none')
io.show()