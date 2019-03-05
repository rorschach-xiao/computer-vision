import tensorflow as tf
import numpy as np

BATCH_SIZE = 12
IMAGE_SIZE_X = 768
IMAGE_SIZE_Y = 384
ROUND_STEP = 12
TRAINING_ROUNDS = 50
LEARNING_RATE = 1e-5

def Weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight");

def Bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name="bias");

def Conv2d(x,W,strides):
    return tf.nn.conv2d(x, W, strides=strides, padding="same");

def TConv(x,W,output_shape):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME');

def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool');

def loss(pre,gt):
    loss = tf.sqrt(tf.reduce_mean(tf.square(pre-gt)));
    return loss

def DispFulNet_model(concat_image, ground_truth, leftimg):
    #conv1
    with tf.name_scope('conv1'):
        W_conv1 = Weight([7, 7, 6, 64]);
        b_conv1 = Bias([64]);
        h_conv1 = tf.nn.leaky_relu(Conv2d(concat_image, W_conv1, [1, 2, 2, 1])+b_conv1, alpha=0.1);

    #conv2
    with tf.name_scope('conv2'):
        W_conv2 = Weight([5, 5, 64, 128]);
        b_conv2 = Bias([64]);
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

    # pr6 +loss6
    with tf.name_scope('pr6_loss6'):
        W_pr6 = Weight([3, 3, 1024, 1]);
        b_pr6 = Bias([1]);
        pr6 = tf.nn.leaky_relu(Conv2d(h_conv6b, W_pr6, [1, 1, 1, 1])+b_pr6);
        gt6 = tf.nn.avg_pool(ground_truth, ksize = [1, 64, 64, 1], strides=[1, 64, 64, 1], padding='SAME',name='gt6');
        loss6 = loss(pr6, gt6);

    # upconv_pr6
    with tf.name_scope('upconv_pr6'):
        W_upconvpr6 = Weight([4, 4, 1, 1]);
        b_upconvpr6 = Bias([1]);
        h_upconvpr6 = tf.nn.leaky_relu(TConv(pr6, W_upconvpr6, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32), 1]) + b_upconvpr6);

    # upconv5
    with tf.name_scope('upconv5'):
        W_upconv5 = Weight([4, 4, 1024, 512]);
        b_upconv5 = Bias([512]);
        h_upconv5 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_conv6b,W_upconv5,[BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32), 512]) + b_upconv5, center=True, scale=True, is_training=True))

    # iconv5
    with tf.name_scope('iconv5'):
        W_iconv5 = Weight([3, 3, 1025, 512]);
        b_iconv5 = Bias([512]);
        h_iconv5 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv5, h_conv5b, h_upconvpr6], 3), W_iconv5, [1, 1, 1, 1])+ b_iconv5,alpha = 0.1);

    # pr5 + loss5
    with tf.name_scope('pr5_loss5'):
        W_pr5 = Weight([3, 3, 512, 1]);
        b_pr5 = Bias([1]);
        pr5 = tf.nn.leaky_relu(Conv2d(h_iconv5, W_pr5, [1, 1, 1, 1]) + b_pr5);
        gt5 = tf.nn.avg_pool(ground_truth, ksize=[1, 32, 32, 1], strides=[1, 32, 32, 1], padding='SAME', name='gt5');
        loss5 = loss(pr5, gt5);

    #upconv4
    with tf.name_scope('upconv4'):
        W_upconv4 = Weight([4, 4, 512, 256]);
        b_upconv4 = Bias([256]);
        h_upconv4 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv5, W_upconv4, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 16), np.int32(IMAGE_SIZE_X / 16), 256]) + b_upconv4, center=True, scale=True, is_training=True))

    # upconv_pr5
    with tf.name_scope('upconv_pr5'):
        W_upconvpr5 = Weight([4, 4, 1, 1]);
        b_upconvpr5 = Bias([1]);
        h_upconvpr5 = tf.nn.leaky_relu(TConv(pr5, W_upconvpr5, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32), 1]) + b_upconvpr5);

    # iconv4
    with tf.name_scope('iconv4'):
        W_iconv4 = Weight([3, 3, 769, 256]);
        b_iconv4 = Bias([256]);
        h_iconv4 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv4, h_conv4b, h_upconvpr5], 3), W_iconv4, [1, 1, 1, 1]) + b_iconv4, alpha=0.1);

    # pr4 + loss4
    with tf.name_scope('pr4_loss4'):
        W_pr4 = Weight([3, 3, 256, 1]);
        b_pr4 = Bias([1]);
        pr4 = tf.nn.leaky_relu(Conv2d(h_iconv4, W_pr4, [1, 1, 1, 1]) + b_pr4);
        gt4 = tf.nn.avg_pool(ground_truth, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME',name='gt4');
        loss4 = loss(pr4, gt4);

    # upconv3
    with tf.name_scope('upconv3'):
        W_upconv3 = Weight([4, 4, 256, 128]);
        b_upconv3 = Bias([128]);
        h_upconv3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv4, W_upconv3,[BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 8),np.int32(IMAGE_SIZE_X / 8), 128]) + b_upconv3, center=True, scale=True,is_training=True))

    # upconv_pr4
    with tf.name_scope('upconv_pr4'):
        W_upconvpr4 = Weight([4, 4, 1, 1]);
        b_upconvpr4 = Bias([1]);
        h_upconvpr4 = tf.nn.leaky_relu(TConv(pr4, W_upconvpr4, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 8), np.int32(IMAGE_SIZE_X / 8), 1]) + b_upconvpr4);

    # iconv3
    with tf.name_scope('iconv3'):
        W_iconv3 = Weight([3, 3, 385, 128]);
        b_iconv3 = Bias([128]);
        h_iconv3 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv3, h_conv3b, h_upconvpr4], 3), W_iconv3, [1, 1, 1, 1]) + b_iconv3, alpha=0.1);

    # pr3 + loss3
    with tf.name_scope('pr3_loss3'):
        W_pr3 = Weight([3, 3, 128, 1]);
        b_pr3 = Bias([1]);
        pr3 = tf.nn.leaky_relu(Conv2d(h_iconv3, W_pr3, [1, 1, 1, 1]) + b_pr3);
        gt3 = tf.nn.avg_pool(ground_truth, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME',name='gt3');
        loss3 = loss(pr3, gt3);

    # upconv2
    with tf.name_scope('upconv2'):
        W_upconv2 = Weight([4, 4, 128, 64]);
        b_upconv2 = Bias([64]);
        h_upconv2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv3, W_upconv2, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 4),np.int32(IMAGE_SIZE_X / 4),64]) + b_upconv2,center=True, scale=True, is_training=True))

    # upconv_pr3
    with tf.name_scope('upconv_pr3'):
        W_upconvpr3 = Weight([4, 4, 1, 1]);
        b_upconvpr3 = Bias([1]);
        h_upconvpr3 = tf.nn.leaky_relu(TConv(pr3, W_upconvpr3, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 4), np.int32(IMAGE_SIZE_X / 4),1]) + b_upconvpr3);

    # iconv2
    with tf.name_scope('iconv2'):
        W_iconv2 = Weight([3, 3, 193, 64]);
        b_iconv2 = Bias([64]);
        h_iconv2 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv2, h_conv2, h_upconvpr3], 3), W_iconv2, [1, 1, 1, 1]) + b_iconv2, alpha=0.1);

    # pr2 + loss2
    with tf.name_scope('pr2_loss2'):
        W_pr2 = Weight([3, 3, 64, 1]);
        b_pr2 = Bias([1]);
        pr2 = tf.nn.leaky_relu(Conv2d(h_iconv2, W_pr2, [1, 1, 1, 1]) + b_pr2);
        gt2 = tf.nn.avg_pool(ground_truth, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME',name='gt2');
        loss2 = loss(pr2, gt2);

    # upconv1
    with tf.name_scope('upconv1'):
        W_upconv1 = Weight([4, 4, 64, 32]);
        b_upconv1 = Bias([32]);
        h_upconv1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv2, W_upconv1, [BATCH_SIZE,np.int32(IMAGE_SIZE_Y / 2),np.int32(IMAGE_SIZE_X / 2),32]) + b_upconv1,center=True, scale=True, is_training=True))

    # upconv_pr2
    with tf.name_scope('upconv_pr2'):
        W_upconvpr2 = Weight([4, 4, 1, 1]);
        b_upconvpr2 = Bias([1]);
        h_upconvpr2 = tf.nn.leaky_relu(TConv(pr2, W_upconvpr2, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32),1]) + b_upconvpr2);

    # iconv1
    with tf.name_scope('iconv1'):
        W_iconv1 = Weight([3, 3, 97, 32]);
        b_iconv1 = Bias([32]);
        h_iconv1 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv1, h_conv1, h_upconvpr2], 3), W_iconv1, [1, 1, 1, 1]) + b_iconv1, alpha=0.1);

    # pr1 + loss1
    with tf.name_scope('pr1_loss1'):
        W_pr1 = Weight([3, 3, 32, 1]);
        b_pr1 = Bias([1]);
        pr1 = tf.nn.leaky_relu(Conv2d(h_iconv1, W_pr1, [1, 1, 1, 1]) + b_pr1);
        gt1 = tf.nn.avg_pool(ground_truth, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='gt1');
        loss1 = loss(pr1, gt1);

    # upconv0
    with tf.name_scope('upconv0'):
        W_upconv0 = Weight([4, 4, 32, 16]);
        b_upconv0 = Bias([16]);
        h_upconv0 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_iconv1, W_upconv0, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y), np.int32(IMAGE_SIZE_X), 16]) + b_upconv0, center=True, scale=True,is_training=True))

    # upconv_pr1
    with tf.name_scope('upconv_pr1'):
        W_upconvpr1 = Weight([4, 4, 1, 1]);
        b_upconvpr1 = Bias([1]);
        h_upconvpr1 = tf.nn.leaky_relu(TConv(pr1, W_upconvpr1, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y), np.int32(IMAGE_SIZE_X ),1]) + b_upconvpr1);

    # iconv0
    with tf.name_scope('iconv0'):
        W_iconv0 = Weight([3, 3, 20, 32]);
        b_iconv0 = Bias([32]);
        h_iconv0 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv0, leftimg, h_upconvpr1], 3), W_iconv0, [1, 1, 1, 1]) + b_iconv0, alpha=0.1);

    # pr0+loss0
    with tf.name_scope('pr0+loss0'):
        W_pr0 = Weight([3, 3, 32, 1]);
        b_pr0 = Bias([1]);
        pr0 = tf.nn.leaky_relu(Conv2d(h_iconv0, W_pr0, [1, 1, 1, 1])+b_pr0);
        gt0 = tf.nn.avg_pool(ground_truth, ksize=[1, 2 ,2, 1], strides=[1, 2, 2, 1], padding='SAME', name='gt0');
        loss0 = loss(pr0,gt0);

    output_disparity=pr0

    with tf.name_scope('loss'):
        total_loss=( loss0+ 1/2 * loss1 + 1/4 * loss2 + 1/8 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6);
    return output_disparity, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, tf.is_inf(loss6)


