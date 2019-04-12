import tensorflow as tf
import numpy as np
from math import ceil

BATCH_SIZE = 10
IMAGE_SIZE_X = 960
IMAGE_SIZE_Y = 540
def Weight(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    #initializer = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initializer(shape=shape), name="weight");

def Bilinear_Deconv_Weight(shape,name):
    """
    compute the bilinear filter and return it
    """
    filt_width = shape[0]
    filt_height= shape[1]
    half_width = ceil(filt_width /2.0)
    center = (2 * half_width - 1 - half_width % 2) / (2.0 * half_width) # 计算某点的权值  对这个点进行插值
    bilinear = np.zeros([filt_width, filt_height])
    for x in range(filt_width):
        for y in range(filt_height):
            value = (1 - abs(x / half_width - center)) * (1 - abs(y / half_width - center))
            bilinear[x, y] = value
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear
        #print(weights[:, :, i, i])
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        return tf.get_variable(name=name,initializer=init , shape=shape)


def Bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name="bias");

def Conv2d(x,W,strides):
    return tf.nn.conv2d(x, W, strides=strides, padding="SAME");

def TConv(x,W,output_shape):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME');

def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool');

def max_pool(x,factor):
    return tf.nn.max_pool(x,ksize=[1,factor,factor,1],stride=[1,factor,factor,1],padding='SAME',name='max_pool_x')

def loss(pre,gt):
    loss = tf.sqrt(tf.reduce_mean(tf.square(pre-gt)));
    return loss

def l1_loss(pre,gt):
    loss = tf.abs(pre-gt)
    return loss

def DispFulNet_model(concat_image, ground_truth, leftimg):

    #conv1
    with tf.name_scope('conv1'):
        W_conv1 = Weight([7, 7, 6, 64]); #[kernel_size,kernel_size,input_channel,output_channel]
        b_conv1 = Bias([64]);
        h_conv1 = tf.nn.leaky_relu(Conv2d(concat_image, W_conv1, [1, 2, 2, 1])+b_conv1, alpha=0.1);

    #conv2
    with tf.name_scope('conv2'):
        W_conv2 = Weight([5, 5, 64, 128]);
        b_conv2 = Bias([128]);
        h_conv2 = tf.nn.leaky_relu(Conv2d(h_conv1, W_conv2, [1, 2, 2, 1])+b_conv2, alpha=0.1);

    #conv3a
    with tf.name_scope('conv3a'):
        W_conv3a = Weight([5, 5, 128, 256]);
        b_conv3a = Bias([256]);
        h_conv3a = tf.nn.leaky_relu(Conv2d(h_conv2, W_conv3a, [1, 2, 2, 1])+b_conv3a, alpha=0.1);

    #conv3b
    with tf.name_scope('conv3b'):
        W_conv3b = Weight([3, 3, 256, 256]);
        b_conv3b = Bias([256]);
        h_conv3b = tf.nn.leaky_relu(Conv2d(h_conv3a, W_conv3b, [1, 1, 1,1])+b_conv3b, alpha=0.1);

    #conv4a
    with tf.name_scope('conv4a'):
        W_conv4a = Weight([3, 3, 256, 512]);
        b_conv4a = Bias([512]);
        h_conv4a = tf.nn.leaky_relu(Conv2d(h_conv3b, W_conv4a, [1, 2, 2, 1])+b_conv4a, alpha=0.1);

    # conv4b
    with tf.name_scope('conv4b'):
        W_conv4b = Weight([3, 3, 512, 512]);
        b_conv4b = Bias([512]);
        h_conv4b = tf.nn.leaky_relu(Conv2d(h_conv4a, W_conv4b, [1, 1, 1, 1]) + b_conv4b, alpha=0.1);

    # conv5a
    with tf.name_scope('conv5a'):
        W_conv5a = Weight([3, 3, 512, 512]);
        b_conv5a = Bias([512]);
        h_conv5a = tf.nn.leaky_relu(Conv2d(h_conv4b, W_conv5a, [1, 2, 2, 1]) + b_conv5a, alpha=0.1);

    # conv5b
    with tf.name_scope('conv5b'):
        W_conv5b = Weight([3, 3, 512, 512]);
        b_conv5b = Bias([512]);
        h_conv5b = tf.nn.leaky_relu(Conv2d(h_conv5a, W_conv5b, [1, 1, 1, 1]) + b_conv5b, alpha=0.1);

    # conv6a
    with tf.name_scope('conv6a'):
        W_conv6a = Weight([3, 3, 512, 1024]);
        b_conv6a = Bias([1024]);
        h_conv6a = tf.nn.leaky_relu(Conv2d(h_conv5b, W_conv6a, [1, 2, 2, 1]) + b_conv6a, alpha=0.1);

    # conv6b
    with tf.name_scope('conv6b'):
        W_conv6b = Weight([3, 3, 1024, 1024]);
        b_conv6b = Bias([1024]);
        h_conv6b = tf.nn.leaky_relu(Conv2d(h_conv6a, W_conv6b, [1, 1, 1, 1]) + b_conv6b, alpha=0.1);

    # pr6 _loss6
    with tf.name_scope('pr6_loss6'):
        W_pr6 = Weight([3, 3, 1024, 1]);
        b_pr6 = Bias([1]);
        pr6 = tf.nn.leaky_relu(Conv2d(h_conv6b, W_pr6, [1, 1, 1, 1])+b_pr6,alpha=0.1);
        gt6 = tf.nn.avg_pool(ground_truth, ksize = [1, 64, 64, 1], strides=[1, 64, 64, 1], padding='SAME',name='gt6');
        loss6 = loss(pr6, gt6);

    # upconv_pr6
    with tf.name_scope('upconv_pr6'):
        W_upconvpr6 = Bilinear_Deconv_Weight([4,4,1,1],"bili_weight6"); #采用双线性差值反卷积核
        b_upconvpr6 = Bias([1]);
        h_upconvpr6 = tf.nn.leaky_relu(TConv(pr6, W_upconvpr6, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32)+1, np.int32(IMAGE_SIZE_X / 32), 1]) + b_upconvpr6,alpha=0.1);

    # upconv5
    with tf.name_scope('upconv5'):
        W_upconv5 = Weight([4, 4, 512, 1024]); #[kernel_size,kernel_size,output_channel,input_channel]
        b_upconv5 = Bias([512]);
        h_upconv5 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_conv6b,W_upconv5,[BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32)+1, np.int32(IMAGE_SIZE_X / 32), 512]) + b_upconv5, center=True, scale=True, is_training=True),alpha=0.1)

    # iconv5
    with tf.name_scope('iconv5'):
        W_iconv5 = Weight([3, 3, 1025, 512]);
        b_iconv5 = Bias([512]);
        h_iconv5 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv5, h_conv5b, h_upconvpr6], 3), W_iconv5, [1, 1, 1, 1])+ b_iconv5,alpha = 0.1);

    # pr5 + loss5
    with tf.name_scope('pr5_loss5'):
        W_pr5 = Weight([3, 3, 512, 1]);
        b_pr5 = Bias([1]);
        pr5 = tf.nn.leaky_relu(Conv2d(h_iconv5, W_pr5, [1, 1, 1, 1]) + b_pr5,alpha=0.1);
        gt5 = tf.nn.avg_pool(ground_truth, ksize=[1, 32, 32, 1], strides=[1, 32, 32, 1], padding='SAME', name='gt5');
        loss5 = loss(pr5, gt5);

    #upconv4
    with tf.name_scope('upconv4'):
        W_upconv4 = Weight([4, 4, 256, 512]);
        b_upconv4 = Bias([256]);
        h_upconv4 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv5, W_upconv4, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 16)+1, np.int32(IMAGE_SIZE_X / 16), 256]) + b_upconv4, center=True, scale=True, is_training=True),alpha=0.1)

    # upconv_pr5
    with tf.name_scope('upconv_pr5'):
        W_upconvpr5 = Bilinear_Deconv_Weight([4,4,1,1],"bili_weight5");
        b_upconvpr5 = Bias([1]);
        h_upconvpr5 = tf.nn.leaky_relu(TConv(pr5, W_upconvpr5, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 16)+1, np.int32(IMAGE_SIZE_X / 16), 1]) + b_upconvpr5,alpha=0.1);

    # iconv4
    with tf.name_scope('iconv4'):
        W_iconv4 = Weight([3, 3, 769, 256]);
        b_iconv4 = Bias([256]);
        h_iconv4 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv4, h_conv4b, h_upconvpr5], 3), W_iconv4, [1, 1, 1, 1]) + b_iconv4, alpha=0.1);

    # pr4 + loss4
    with tf.name_scope('pr4_loss4'):
        W_pr4 = Weight([3, 3, 256, 1]);
        b_pr4 = Bias([1]);
        pr4 = tf.nn.leaky_relu(Conv2d(h_iconv4, W_pr4, [1, 1, 1, 1]) + b_pr4,alpha=0.1);
        gt4 = tf.nn.avg_pool(ground_truth, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME',name='gt4');
        loss4 = loss(pr4, gt4);

    # upconv3
    with tf.name_scope('upconv3'):
        W_upconv3 = Weight([4, 4, 128, 256]);
        b_upconv3 = Bias([128]);
        h_upconv3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv4, W_upconv3,[BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 8)+1,np.int32(IMAGE_SIZE_X / 8), 128]) + b_upconv3, center=True, scale=True,is_training=True),alpha=0.1)

    # upconv_pr4
    with tf.name_scope('upconv_pr4'):
        W_upconvpr4 = Bilinear_Deconv_Weight([4,4,1,1],"bili_weight4");
        b_upconvpr4 = Bias([1]);
        h_upconvpr4 = tf.nn.leaky_relu(TConv(pr4, W_upconvpr4, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 8)+1, np.int32(IMAGE_SIZE_X / 8), 1]) + b_upconvpr4,alpha=0.1);

    # iconv3
    with tf.name_scope('iconv3'):
        W_iconv3 = Weight([3, 3, 385, 128]);
        b_iconv3 = Bias([128]);
        h_iconv3 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv3, h_conv3b, h_upconvpr4], 3), W_iconv3, [1, 1, 1, 1]) + b_iconv3, alpha=0.1);

    # pr3 + loss3
    with tf.name_scope('pr3_loss3'):
        W_pr3 = Weight([3, 3, 128, 1]);
        b_pr3 = Bias([1]);
        pr3 = tf.nn.leaky_relu(Conv2d(h_iconv3, W_pr3, [1, 1, 1, 1]) + b_pr3,alpha=0.1);
        gt3 = tf.nn.avg_pool(ground_truth, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME',name='gt3');
        loss3 = loss(pr3, gt3);

    # upconv2
    with tf.name_scope('upconv2'):
        W_upconv2 = Weight([4, 4, 64, 128]);
        b_upconv2 = Bias([64]);
        h_upconv2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv3, W_upconv2, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 4),np.int32(IMAGE_SIZE_X / 4),64]) + b_upconv2,center=True, scale=True, is_training=True),alpha=0.1)

    # upconv_pr3
    with tf.name_scope('upconv_pr3'):
        W_upconvpr3 = Bilinear_Deconv_Weight([4,4,1,1],"bili_weight3");
        b_upconvpr3 = Bias([1]);
        h_upconvpr3 = tf.nn.leaky_relu(TConv(pr3, W_upconvpr3, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 4), np.int32(IMAGE_SIZE_X / 4),1]) + b_upconvpr3,alpha=0.1);

    # iconv2
    with tf.name_scope('iconv2'):
        W_iconv2 = Weight([3, 3, 193, 64]);
        b_iconv2 = Bias([64]);
        h_iconv2 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv2, h_conv2, h_upconvpr3], 3), W_iconv2, [1, 1, 1, 1]) + b_iconv2, alpha=0.1);

    # pr2 + loss2
    with tf.name_scope('pr2_loss2'):
        W_pr2 = Weight([3, 3, 64, 1]);
        b_pr2 = Bias([1]);
        pr2 = tf.nn.leaky_relu(Conv2d(h_iconv2, W_pr2, [1, 1, 1, 1]) + b_pr2,alpha=0.1);
        gt2 = tf.nn.avg_pool(ground_truth, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME',name='gt2');
        loss2 = loss(pr2, gt2);

    # upconv1
    with tf.name_scope('upconv1'):
        W_upconv1 = Weight([4, 4, 32, 64]);
        b_upconv1 = Bias([32]);
        h_upconv1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv2, W_upconv1, [BATCH_SIZE,np.int32(IMAGE_SIZE_Y / 2),np.int32(IMAGE_SIZE_X / 2),32]) + b_upconv1,center=True, scale=True, is_training=True),alpha=0.1)

    # upconv_pr2
    with tf.name_scope('upconv_pr2'):
        W_upconvpr2 = Bilinear_Deconv_Weight([4,4,1,1],"bili_weight2");
        b_upconvpr2 = Bias([1]);
        h_upconvpr2 = tf.nn.leaky_relu(TConv(pr2, W_upconvpr2, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 2), np.int32(IMAGE_SIZE_X / 2),1]) + b_upconvpr2,alpha=0.1);

    # iconv1
    with tf.name_scope('iconv1'):
        W_iconv1 = Weight([3, 3, 97, 32]);
        b_iconv1 = Bias([32]);
        h_iconv1 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv1, h_conv1, h_upconvpr2], 3), W_iconv1, [1, 1, 1, 1]) + b_iconv1, alpha=0.1);

    # pr1 + loss1
    with tf.name_scope('pr1_loss1'):
        W_pr1 = Weight([3, 3, 32, 1]);
        b_pr1 = Bias([1]);
        pr1 = tf.nn.leaky_relu(Conv2d(h_iconv1, W_pr1, [1, 1, 1, 1]) + b_pr1,alpha=0.1);
        gt1 = tf.nn.avg_pool(ground_truth, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='gt1');
        loss1 = loss(pr1, gt1);

    # upconv0
    with tf.name_scope('upconv0'):
        W_upconv0 = Weight([4, 4, 16, 32]);
        b_upconv0 = Bias([16]);
        h_upconv0 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv1, W_upconv0, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y), np.int32(IMAGE_SIZE_X), 16]) + b_upconv0, center=True, scale=True,is_training=True),alpha=0.1)

    # upconv_pr1
    with tf.name_scope('upconv_pr1'):
        W_upconvpr1 = Bilinear_Deconv_Weight([4,4,1,1],"bili_weight1");
        b_upconvpr1 = Bias([1]);
        h_upconvpr1 = tf.nn.leaky_relu(TConv(pr1, W_upconvpr1, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y), np.int32(IMAGE_SIZE_X ),1]) + b_upconvpr1,alpha=0.1);

    # iconv0
    with tf.name_scope('iconv0'):
        W_iconv0 = Weight([3, 3, 20, 32]);
        b_iconv0 = Bias([32]);
        h_iconv0 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv0, leftimg, h_upconvpr1], 3), W_iconv0, [1, 1, 1, 1]) + b_iconv0, alpha=0.1);

    # pr0+loss0
    with tf.name_scope('pr0_loss0'):
        W_pr0 = Weight([3, 3, 32, 1]);
        b_pr0 = Bias([1]);
        pr0 = tf.nn.leaky_relu(Conv2d(h_iconv0, W_pr0, [1, 1, 1, 1])+b_pr0,alpha=0.1);
        gt0 = tf.nn.avg_pool(ground_truth, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name='gt0');
        loss0 = loss(pr0,gt0);


    with tf.name_scope('loss'):
        total_loss=( loss0 + 1/2 * loss1 + 1/4 * loss2 + 1/8 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6);
        output_disparity = pr0


    return output_disparity, total_loss,loss0,loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, pr0,tf.is_inf(loss6)
def DispResNet_model(concate_input,ground_truth,pr1):

    #res_conv1
    with tf.name_scope('res_conv1'):
        W_res_conv1 = Weight([5,5,13,64]) #540*960 -> 540*960
        b_res_conv1 = Bias([64])
        h_res_conv1 = tf.nn.leaky_relu(Conv2d(concate_input,W_res_conv1,[1,1,1,1])+b_res_conv1,alpha=0.1)

    #res_conv2
    with tf.name_scope('res_conv2'):
        W_res_conv2 = Weight([5,5,64,128]) #540*960 -> 270*480
        b_res_conv2 = Bias([128])
        h_res_conv2 = tf.nn.leaky_relu(Conv2d(h_res_conv1,W_res_conv2,[1,2,2,1])+b_res_conv2,alpha=0.1)

    #res_conv2_1
    with tf.name_scope('res_conv2_1'):
        W_res_conv2_1 = Weight([3,3,128,128]) #270*480 -> 270*480
        b_res_conv2_1 = Bias([128])
        h_res_conv2_1 = tf.nn.leaky_relu(Conv2d(h_res_conv2,W_res_conv2_1,[1,1,1,1])+b_res_conv2_1,alpha=0.1)

    #res_conv3
    with tf.name_scope('res_conv3'):
        W_res_conv3 = Weight([3,3,128,256]) #270*480 -> 135*240
        b_res_conv3 = Bias([256])
        h_res_conv3 = tf.nn.leaky_relu(Conv2d(h_res_conv2_1,W_res_conv3,[1,2,2,1])+b_res_conv3,alpha=0.1)

    #res_conv3_1
    with tf.name_scope('res_conv3_1'):
        W_res_conv3_1 = Weight([3,3,256,256]) #135*240 -> 135*240
        b_res_conv3_1 = Bias([256])
        h_res_conv3_1 = tf.nn.leaky_relu(Conv2d(h_res_conv3,W_res_conv3_1,[1,1,1,1])+b_res_conv3_1,alpha=0.1)

    #res_conv4
    with tf.name_scope('res_conv4'):
        W_res_conv4 = Weight([3,3,256,512]) #135*240 -> 68*120
        b_res_conv4 = Bias([512])
        h_res_conv4 = tf.nn.leaky_relu(Conv2d(h_res_conv3_1,W_res_conv4,[1,2,2,1])+b_res_conv4,alpha = 0.1)

    #res_conv4_1
    with tf.name_scope('res_conv4_1'):
        W_res_conv4_1 = Weight([3,3,512,512]) #68*120 -> 68*120
        b_res_conv4_1 = Bias([512])
        h_res_conv4_1 = tf.nn.leaky_relu(Conv2d(h_res_conv4,W_res_conv4_1,[1,1,1,1])+b_res_conv4_1,alpha=0.1)

    #res_conv5
    with tf.name_scope('res_conv5'):
        W_res_conv5 = Weight([3,3,512,1024]) # 68*120 -> 34*60
        b_res_conv5 = Bias([1024])
        h_res_conv5 = tf.nn.leaky_relu(Conv2d(h_res_conv4_1,W_res_conv5,[1,2,2,1])+b_res_conv5,alpha=0.1)

    #res_conv5_1
    with tf.name_scope('res_conv5_1'):
        W_res_conv5_1 = Weight([3,3,1024,1024]) #34*60 -> 34*60
        b_res_conv5_1 = Bias([1024])
        h_res_conv5_1 = tf.nn.leaky_relu(Conv2d(h_res_conv5,W_res_conv5_1,[1,1,1,1])+b_res_conv5_1,alpha=0.1)

    #res_16
    with tf.name_scope('res_16'):
        W_res_16 = Weight([3,3,1024,1]) #34*60 -> 34*60
        b_res_16 = Bias([1])
        h_res_16 = Conv2d(h_res_conv5_1,W_res_16,[1,1,1,1])+b_res_16

    #pr_s1_16
    with tf.name_scope('pr_s1_16'):
        pr1_16 = max_pool(pr1,16) # 540*960 -> 34*60

    #pr_s2_16+loss16
    with tf.name_scope('pr_s2_16'):
        pr2_16 = tf.nn.leaky_relu(pr1_16 + h_res_16,alpha=0) #34*60
        gt_16 = max_pool(x,16)
        loss_16 = loss(pr2_16,gt_16)

    #res_upconv4
    with tf.name_scope('res_upconv4'):
        W_res_upconv4 = Weight([4,4,512,1024]) #34*60 -> 68*120
        b_res_upconv4 = Bias([512])
        h_res_upconv4 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_res_conv5_1,W_res_upconv4,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y/8)+1,np.int32(IMAGE_SIZE_X/8),512])+b_res_upconv4,center=True, scale=True, is_training=True),alpha=0.1)

    #res_upconv4_pr2_16to8
    with tf.name_scope('res_upconv4_pr2_16to8'):
        W_res_upconv4_pr2_16to8 = Weight([4,4,1,1]) # 34*60 -> 68*120
        b_res_upconv4_pr2_16to8 = Bias([1])
        pr2_16to8 = TConv(pr2_16,W_res_upconv4_pr2_16to8,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y/8)+1,np.int32(IMAGE_SIZE_X/8),1])+b_res_upconv4_pr2_16to8


    #res_iconv4
    with tf.name_scope('rse_iconv4'):
        W_res_iconv4 = Weight([3,3,1025,512]) #68*120 -> 68*120
        b_res_iconv4 = Bias([512])
        h_res_iconv4 = Conv2d(tf.concat(h_res_upconv4,h_res_conv4_1,pr2_16to8),W_res_iconv4,[1,1,1,1])+b_res_iconv4

    #res_8
    with tf.name_scope('res_8'):
        W_res_8 = Weight([3,3,512,1]) #68*120 -> 68*120
        b_res_8 = Bias([1])
        h_res_8 = Conv2d(h_res_iconv4,W_res_8,[1,1,1,1])+b_res_8

    #pr_s1_8
    with tf.name_scope('pr_s1_8'):
        pr1_8 = max_pool(pr1,8)  #540*960 -> 68*120

    #pr_s2_8 + loss_8
    with tf.name_scope('pr_s2_8'):
        pr2_8 = tf.nn.leaky_relu(pr1_8+h_res_8,alpha=0) # 68*120
        gt_8 = max_pool(ground_truth,8)
        loss_8 = loss(pr2_8,gt_8)

    #res_upconv3
    with tf.name_scope('res_upconv3'):
        W_res_upconv3 = Weight([4,4,256,512]) #68*120 -> 135*240
        b_res_upconv3 = Bias([256])
        h_res_upconv3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_res_iconv4,W_res_upconv3,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y/4),np.int32(IMAGE_SIZE_X/4),256])+b_res_upconv3,center=True, scale=True, is_training=True),alpha=0.1)

    #res_upconv3_pr2_8to4
    with tf.name_scope('res_upconv3_pr2_8to4'):
        W_res_upconv3_pr2_8to4 = Weight([4,4,1,1]) #68*120 ->135*240
        b_res_upconv3_pr2_8to4 = Bias([1])
        h_res_upconv3_pr2_8to4 = Tconv(pr2_8,W_res_upconv3_pr2_8to4,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y/4),np.int32(IMAGE_X/4),1])+b_res_upconv3_pr2_8to4

    #res_iconv3
    with tf.name_scope('res_iconv3'):
        W_res_iconv3 = Weight([3,3,513,256]) #135*240
        b_res_iconv3 = Bias([256])
        h_res_iconv3 = Conv2d(tf.concat(h_res_upconv3,h_res_conv3_1,h_res_upconv3_pr2_8to4),W_res_iconv3,[1,1,1,1])+b_res_iconv3

    #res_4
    with tf.name_scope('res_4'):
        W_res_4 = Weight([3,3,256,1]) #135*240
        b_res_4 = Bias([1])
        h_res_4 = Conv2d(h_res_iconv3,W_res_4,[1,1,1,1])+b_res_4

    #pr_s1_4
    with tf.name_scope('pr_s1_4'):
        pr1_4 = max_pool(pr1,4)  #135*240

    #pr_s2_4+loss_4
    with tf.name_scope('pr_s2_4'):
        pr2_4 = tf.nn.leaky_relu(pr1_4+h_res_4,alpha=0)
        gt4 = max_pool(ground_truth,4)
        loss_4 = loss(pr2_4,gt4)

    #res_upconv2
    with tf.name_scope('res_upconv2'):
        W_res_upconv2 = Weight([4,4,128,256]) #135*240 -> 270*480
        b_res_upconv2 = Bias([128])
        h_res_upconv2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_res_iconv3,W_res_upconv2,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y/2),np.int32(IMAGE_SIZE_X/2),128])+b_res_upconv2,center=True, scale=True, is_training=True),alpha=0.1)

    #res_upconv2_pr2_4to2
    with tf.name_scope('res_upconv2_pr2_4to8'):
        W_res_upconv2_pr2_4to2 = Weight([4,4,1,1]) #135*240 ->270*480
        b_res_upconv2_pr2_4to2 = Bias([1])
        h_res_upconv2_pr2_4to2 = TConv(pr2_4,W_res_upconv2_pr2_4to2,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y/2),np.int32(IMAGE_SIZE_X/2),1])+b_res_upconv2_pr2_4to2

    #res_iconv2
    with tf.name_scope('res_iconv2'):
        W_res_iconv2 = Weight([3,3,257,128])  #270*480 -> 270*480
        b_res_iconv2 = Bias([128])
        h_res_iconv2 = Conv2d(tf.concat(h_res_upconv2,h_res_conv2_1,h_res_upconv2_pr2_4to2),W_res_iconv2,[1,1,1,1])+b_res_iconv2

    #res_2
    with tf.name_scope('res_2'):
        W_res_2 = Weight([3,3,128,1]) #270*480 -> 270*480
        b_res_2 =Bias([1])
        h_res_2 = Conv2d(h_res_iconv2,W_res_2,[1,1,1,1])+b_res_2

    #pr_s1_2
    with tf.name_scope('pr_s1_2'):
        pr1_2 = max_pool(pr1,2)  #540*960 -> 270*480

    #pr_s2_2 +loss_2
    with tf.name_scope('pr_s2_2'):
        pr2_2 = tf.nn.leaky_relu(pr1_2+h_res_2,alpha=0) #270*480
        gt2 = max_pool(ground_truth,2)
        loss_2 =loss(pr2_2,gt2)

    #res_upconv1
    with tf.name_scope('res_upconv1'):
        W_res_upconv1 = Weight([4,4,64,128]) #270*480 -> 540*960
        b_res_upconv1 = Bias([64])
        h_res_upconv1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_res_iconv2,W_res_upconv1,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y),np.int32(IMAGE_SIZE_X),64])+b_res_upconv1,center=True, scale=True, is_training=True),alpha=0.1)

    #res_upconv1_pr2_2to1
    with tf.name_scope('res_upconv1_pr2_2to1'):
        W_res_upconv1_pr2_2to1 = Weight([4,4,1,1]) #270*480 -> 540*960
        b_res_upconv1_pr2_2to1 = Bias([1])
        h_res_upconv1_pr2_2to1 = TConv(pr2_2,W_res_upconv1_pr2_2to1,[BATCH_SIZE,np.int32(IMAGE_SIZE_Y),np.int32(IMAGE_SIZE_X),1])+b_res_upconv1_pr2_2to1

    #res_1
    with tf.name_scope('res_iconv1'):
        W_res_1 = Weight([3,3,129,1]) #540*960 -> 540*960
        b_res_1 = Bias([1])
        h_res_1 = Conv2d(tf.concat(h_res_upconv1,h_res_conv1,h_res_upconv1_pr2_2to1),W_res_1)+b_res_1

    #pr_s2
    with tf.name_scope('pr_s2'):
        pr2 = tf.nn.leaky_relu(pr1+h_res_1,alpha=0) #540*960
        loss0 = loss(pr2,ground_truth)

    with tf.name_scope('total loss'):
        total_loss = 1/16*loss_16 + 1/8*loss_8 + 1/4*loss_4 + 1/2*loss_2 + loss0

    return total_loss,loss0,pr2








