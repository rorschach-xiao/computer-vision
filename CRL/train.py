from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import time
from model import *
from datetime import date
import _pickle as cPickle
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from prepfm import *

BATCH_SIZE = 3
IMAGE_SIZE_X = 960
IMAGE_SIZE_Y = 540
ROUND_STEP = 3
TRAINING_ROUNDS = 1000
LEARNING_RATE_BASE = 1e-5
LEARNING_RATE_DECAY = 0.95

MODEL_DIR = 'C:/Users/HK/PycharmProjects/CRL/model'
DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/frames_cleanpass/35mm_focallength/scene_forwards/fast'
#DATA_DIR = 'E:/FlyingThings3D_subset/train/image_clean'
#DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/medical data/image'
LOGS_DIR = 'C:/Users/HK/PycharmProjects/CRL/logs'
RUNNING_LOGS_DIR = 'C:/Users/HK/PycharmProjects/CRL/running_logs'
OUTPUT_DIR = 'C:/Users/HK/PycharmProjects/CRL/output/train_14/car/'
GT_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/disparity/35mm_focallength/scene_forwards/fast/left/'
#GT_DIR = 'E:/FlyingThings3D_subset/train/disparity/right/'
#GT_DIR = 'C:/Users/HK/PycharmProjects/CRL/medical data/disp/medical_gt/'
#DISP_DIR = 'C:/Users/HK/PycharmProjects/CRL/disparity_1/medical/'
DISP_DIR = 'C:/Users/HK/PycharmProjects/CRL/disparity_1/car/'
SYN_IMAGE_DIR = 'C:/Users/HK/PycharmProjects/CRL/syn_image/car/'
#TRAIN_SERIES = list(range(0,27))+list(range(30,57))+list(range(60,87))+list(range(90,117))+list(range(120,147))+list(range(150,177))+list(range(180,207))+list(range(210,237))+list(range(240,267))+list(range(270,297))
#image_num = np.size(TRAIN_SERIES)
#TEST_SERIES = list(set(list(range(300)))^set(TRAIN_SERIES))
TRAIN_SERIES = list(range(300))
image_num = np.size(TRAIN_SERIES)


def py_avg_pool(value, strides):
    batch_size , height , width , channel_size = value.shape
    res_height = int(height/strides[1])
    res_width  = int(width/strides[2])
    print(res_height , res_width)
    result = np.zeros((batch_size, res_height, res_width, 1))
    for i in range(res_height):
        for j in range(res_width):
            for k in range(batch_size):
                result[k, i, j, 0] = np.mean(value[k, i * int(strides[1]): (i+1) * int(strides[1]), j * int(strides[2]):(j+1) * int(strides[2]), :])
    return result

def _norm(img):
    return (img - np.mean(img)) / np.std(img)

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
                    img_left_s [i][j][k] = (1-u)*image_right[i][j][x1] + u*image_right[i][j][x2]

    return img_left_s




def train_first_stage():

    image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_left')
    image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_right')
    ground_truth = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1], name='ground_truth')
    combine_image = tf.concat([image_left, image_right], 3)
    final_output, epe, total_loss,loss0, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1,pr0, loss6_inf = DispFulNet_model(concat_image=combine_image, ground_truth=ground_truth,leftimg=image_left)
    tf.summary.scalar('loss', total_loss)
    tf.summary.image('disparity',final_output)
    with tf.name_scope('train'):
        #global_step = tf.Variable(0, trainable=False)
        #LEARNING_RATE = tf.train.exponential_decay(LEARNING_RATE_BASE ,global_step,100, LEARNING_RATE_DECAY,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE).minimize(total_loss)

    # important step
    # reload the model

    sess = tf.Session()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)

    left_images = os.listdir(DATA_DIR + '/left/')
    left_images.sort(key=lambda x: int(x[:-4]))
    right_images = os.listdir(DATA_DIR + '/right/')
    right_images.sort(key=lambda x: int(x[:-4]))
    ground_truths = sorted(os.listdir(GT_DIR))
    #ground_truths = os.listdir(GT_DIR)
    #ground_truths.sort(key = lambda x : int(x[:-4]))
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=1) # save the minimum-loss generate model
    minimum_loss = 10
    sess.run(init)
    # saver.restore(sess, MODEL_PATH)
    with open(RUNNING_LOGS_DIR + "/log" + date.isoformat(date.today()) + str(time.time()) + ".txt","w+") as file:
        file.write('BATCH_SIZE ' + str(BATCH_SIZE) + '\n'  + ' TRAINING_ROUNDS ' + str(TRAINING_ROUNDS) + '\n' + ' image_num ' + str(image_num) + '\n' + ' LEARNING_RATE ' + str(LEARNING_RATE_BASE) + '\n')
        real_loss_record=[]
        for round in range(TRAINING_ROUNDS):
            for i in range(0, image_num - BATCH_SIZE, ROUND_STEP):
                for j in range(BATCH_SIZE):
                    # input data
                    full_pic_name = GT_DIR + ground_truths[TRAIN_SERIES[i + j]]
                    input_one_image, scale = load_pfm(full_pic_name)
                    # input_one_image = (np.array(input_one_image))[:, :, [0]]
                    input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X))
                    input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
                    if (j == 0):
                        input_gts = input_one_image
                    else:
                        input_gts = np.concatenate((input_gts, input_one_image), axis=0)
                    full_pic_name = DATA_DIR + '/left/' + left_images[TRAIN_SERIES[i + j]]
                    input_one_image = Image.open(full_pic_name)
                    input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                    if (j == 0):
                        input_left_images = input_one_image
                    else:
                        input_left_images = np.concatenate((input_left_images, input_one_image), axis=0)
                    full_pic_name = DATA_DIR + '/right/' + right_images[TRAIN_SERIES[i + j]]
                    input_one_image = Image.open(full_pic_name)
                    input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                    if (j == 0):
                        input_right_images = input_one_image
                    else:
                        input_right_images = np.concatenate((input_right_images, input_one_image), axis=0)
                    full_pic_name = GT_DIR + ground_truths[TRAIN_SERIES[i + j]]
                    input_one_image, scale = load_pfm(full_pic_name)
                    # input_one_image = (np.array(input_one_image))[:, :, [0]]

                result, optimizer_res, epe_loss, total_loss_res, loss0_res, loss1_res, loss2_res, loss3_res, loss4_res, loss5_res, loss6_res, pr6_res, pr5_res, pr4_res, pr3_res, pr2_res, pr1_res,pr0_res, loss6_inf_res = sess.run([merged, optimizer, epe, total_loss,loss0, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2,pr1,pr0, loss6_inf],feed_dict={image_left: input_left_images, image_right: input_right_images, ground_truth: input_gts})
                writer.add_summary(result, round)
                if round == TRAINING_ROUNDS - 1:
                    final_result = (sess.run(final_output, feed_dict={image_left: input_left_images, image_right: input_right_images, ground_truth: input_gts}))
                    for k in range(10):
                        result = np.squeeze(final_result[k])
                        #result = result.astype(np.uint32)
                        writePFM(OUTPUT_DIR + '/first_stage/' +str(i + k) + '.pfm', result)
                        #plt.imsave(OUTPUT_DIR + '/' + str(i+k) + '.png', result, format='png',cmap='gray')
                        file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(
                            total_loss_res) + ' loss0 ' + str(loss0_res) + '\n')
                if i == 0:
                    pr0_real_loss = loss0_res
                    real_loss_record.append(pr0_real_loss)
                    print('Epoch %d :'%(round)+' pr0_real_loss ' + str(pr0_real_loss)+'  total_loss '+str(total_loss_res) + ' EPE ' +str(epe_loss))
                    if(pr0_real_loss<minimum_loss):
                        minimum_loss=pr0_real_loss
                        saver.save(sess, MODEL_DIR+'/DispFulNet.ckpt',global_step=round+1)
                    file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(
                        total_loss_res) + ' loss1 ' + str(loss1_res) + '\n')

    # show loss curve
    x_axis = list(range(3000))
    plt.title('Learning rate=1e-5 ')
    plt.plot(x_axis, real_loss_record, color='blue', label='train loss')
    plt.savefig("C:/Users/HK/PycharmProjects/CRL/train_loss_graph/FIG_12.jpg")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.show()

def train_second_stage():
    image_left = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE_Y,IMAGE_SIZE_X,3], name='image_left')
    image_right = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE_Y,IMAGE_SIZE_X,3],name = 'image_right')
    ground_truth = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE_Y,IMAGE_SIZE_X,1],name = 'ground_truth')
    disparity_1 = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1], name='disparity_1')
    image_left_s = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE_Y,IMAGE_SIZE_X,3],name='image_left_syn')
    err = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE_Y,IMAGE_SIZE_X,3],name='error')
    combine_input = tf.concat([image_left,image_right,image_left_s,err,disparity_1],3)
    total_loss , loss0, final_output , epe_loss = DispResNet_model(combine_input,ground_truth,disparity_1)

    tf.summary.scalar('smooth_l1_loss',total_loss)
    tf.summary.image('disparity',final_output)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE).minimize(total_loss)

    sess = tf.Session()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGS_DIR,sess.graph)

    left_images = os.listdir(DATA_DIR + '/left/')
    left_images.sort(key=lambda x: int(x[:-4]))
    right_images = os.listdir(DATA_DIR + '/right/')
    right_images.sort(key=lambda x: int(x[:-4]))
    ground_truths = sorted(os.listdir(GT_DIR))
    disparitys = os.listdir(DISP_DIR)
    disparitys.sort(key = lambda x:int(x[:-4]))
    syn_images = os.listdir(SYN_IMAGE_DIR)
    syn_images.sort(key = lambda x:int(x[:-4]))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)

    minimum_loss = 2
    sess.run(init)

    with open(RUNNING_LOGS_DIR + "/log" + date.isoformat(date.today()) + str(time.time())+'_2' + ".txt","w+") as file:
        file.write('BATCH_SIZE ' + str(BATCH_SIZE) + '\n'  + ' TRAINING_ROUNDS ' + str(TRAINING_ROUNDS) + '\n' + ' image_num ' + str(image_num) + '\n' + ' LEARNING_RATE ' + str(LEARNING_RATE_BASE) + '\n')
        real_loss_record=[]
        for round in range(TRAINING_ROUNDS):
            for i in range(0, image_num - BATCH_SIZE, ROUND_STEP):
                for j in range(BATCH_SIZE):
                    # input data
                    full_pic_name = DATA_DIR + '/left/' + left_images[TRAIN_SERIES[i + j]]
                    input_one_image_l = Image.open(full_pic_name)
                    input_one_image_l = _norm(np.reshape(input_one_image_l, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                    if (j == 0):
                        input_left_images = input_one_image_l
                    else:
                        input_left_images = np.concatenate((input_left_images, input_one_image_l), axis=0)
                    full_pic_name = DATA_DIR + '/right/' + right_images[TRAIN_SERIES[i + j]]
                    input_one_image_r = Image.open(full_pic_name)
                    input_one_image_r = _norm(np.reshape(input_one_image_r, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                    if (j == 0):
                        input_right_images = input_one_image_r
                    else:
                        input_right_images = np.concatenate((input_right_images, input_one_image_r), axis=0)
                    full_pic_name = GT_DIR + ground_truths[TRAIN_SERIES[i + j]]
                    input_one_image_g, scale = load_pfm(full_pic_name)
                    # input_one_image = (np.array(input_one_image))[:, :, [0]]
                    input_one_image_g = np.reshape(input_one_image_g, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X))
                    input_one_image_g = np.reshape(input_one_image_g, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
                    if (j == 0):
                        input_gts = input_one_image_g
                    else:
                        input_gts = np.concatenate((input_gts, input_one_image_g), axis=0)
                    full_pic_name = DISP_DIR + disparitys[TRAIN_SERIES[i + j]]
                    input_one_image_d ,scale = load_pfm(full_pic_name)
                    input_one_image_d = np.reshape(input_one_image_d,(1,IMAGE_SIZE_Y,IMAGE_SIZE_X))
                    input_one_image_d = np.reshape(input_one_image_d,(1,IMAGE_SIZE_Y,IMAGE_SIZE_X,1))
                    if (j == 0):
                        input_pr = input_one_image_d
                    else:
                        input_pr = np.concatenate((input_pr,input_one_image_d),axis=0)

                    #input_one_image_s = warp(input_one_image_r,input_one_image_d)
                    full_pic_name = SYN_IMAGE_DIR  + syn_images[TRAIN_SERIES[i + j]]
                    input_one_image_s = plt.imread(full_pic_name)[:,:,0:3]
                    input_one_image_s = _norm(np.reshape(input_one_image_s, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))

                    if (j == 0):
                        input_syn = input_one_image_s
                    else:
                        input_syn = np.concatenate((input_syn,input_one_image_s),axis=0)
                    input_one_image_e = np.abs(input_one_image_l-input_one_image_s)
                    if(j == 0):
                        input_err = input_one_image_e
                    else:
                        input_err = np.concatenate((input_err,input_one_image_e),axis=0)

                result , optimizer_res , total_loss_res , loss0_res , epe_loss_res = sess.run([merged,optimizer,total_loss,loss0,epe_loss],
                                                                                              feed_dict={image_left:input_left_images,image_right:input_right_images,ground_truth:input_gts,disparity_1:input_pr,image_left_s:input_syn,err:input_err})
                writer.add_summary(result, round)
                if round == TRAINING_ROUNDS - 1:
                    final_result = sess.run(final_output, feed_dict={image_left:input_left_images,image_right:input_right_images,ground_truth:input_gts,disparity_1:input_pr,image_left_s:input_syn,err:input_err})
                    for k in range(3):
                        result = np.squeeze(final_result[k])
                        #result = result.astype(np.uint32)
                        writePFM(OUTPUT_DIR  + '000' +str(i + k) + '.pfm', result)
                        #plt.imsave(OUTPUT_DIR + '/' + str(i+k) + '.png', result, format='png',cmap='gray')
                        file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(
                            total_loss_res) + ' loss0 ' + str(loss0_res) + '\n')
                if i == 0:
                    pr0_real_loss = loss0_res
                    real_loss_record.append(pr0_real_loss)
                    print('Epoch %d :'%(round)+' pr0_real_loss ' + str(pr0_real_loss)+'  total_loss '+str(total_loss_res) + ' EPE ' +str(epe_loss_res))
                    if(pr0_real_loss<minimum_loss):
                        minimum_loss=pr0_real_loss
                        saver.save(sess, MODEL_DIR+'/DispResNet.ckpt',global_step=round+1)
                    file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(
                        total_loss_res) + ' loss0 ' + str(loss0_res) + '\n')
    # show loss curve
    x_axis = list(range(1000))
    plt.title('Learning rate=1e-5')
    plt.plot(x_axis, real_loss_record, color='blue', label='train loss')
    plt.savefig("C:/Users/HK/PycharmProjects/CRL/train_loss_graph/FIG_14.jpg")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    #train_first_stage()
    train_second_stage()

