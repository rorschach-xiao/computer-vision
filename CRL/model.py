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

def upsampling_bilinear():

def DispFulNet_model(concat_image, ground_truth):
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

    # upconv5
    with tf.name_scope('upconv5'):
        W_upconv5 = Weight([4, 4, 1024, 512]);
        b_upconv5 = Bias([512]);
        h_upconv5 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(TConv(h_conv6b,W_upconv5,[BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32), 512]) + b_upconv5, center=True, scale=True, is_training=True))

    # iconv5
    with tf.name_scope('iconv5'):
        W_iconv5 = Weight([3, 3, 1024, 512]);
        b_iconv5 = Bias([512]);
        h_iconv5 = tf.nn.leaky_relu(Conv2d(tf.concat([h_upconv5, h_conv5b], 3), W_iconv5, [1, 1, 1, 1])+ b_iconv5,alpha = 0.1);


    #



