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

BATCH_SIZE = 10
IMAGE_SIZE_X = 960
IMAGE_SIZE_Y = 540
ROUND_STEP = 10
TRAINING_ROUNDS = 2000
LEARNING_RATE_BASE = 1e-5
LEARNING_RATE_DECAY = 0.95

MODEL_DIR = 'C:/Users/HK/PycharmProjects/CRL/model'
DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/frames_cleanpass/35mm_focallength/scene_forwards/fast'
LOGS_DIR = 'C:/Users/HK/PycharmProjects/CRL/logs'
RUNNING_LOGS_DIR = 'C:/Users/HK/PycharmProjects/CRL/running_logs'
OUTPUT_DIR = 'C:/Users/HK/PycharmProjects/CRL/output/train_08'
GT_DIR = 'C:/Users/HK/PycharmProjects/CRL/ground_truth/train/35mm-focallength/forward/fast/left/'
TRAIN_SERIES = list(range(0,27))+list(range(30,57))+list(range(60,87))+list(range(90,117))+list(range(120,147))+list(range(150,177))+list(range(180,207))+list(range(210,237))+list(range(240,267))+list(range(270,297))
image_num = np.size(TRAIN_SERIES)
TEST_SERIES = list(set(list(range(300)))^set(TRAIN_SERIES))
GT_SERIES =list(range(270))

def load_pfm(fname):
    file=open(fname,'rb')
    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encoding='utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape))

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

def main():
    #with open(GT_DIR, 'rb') as f:
    #    buf = cPickle.load(f)
    #    for i in range(len(buf)):
    #        for j in range(1, 15):
    #            buf[i][525 + j] = 255
    image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_left')
    image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_right')
    ground_truth = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1], name='ground_truth')
    combine_image = tf.concat([image_left, image_right], 3)
    final_output, total_loss,loss0, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1,pr0, loss6_inf = DispFulNet_model(concat_image=combine_image, ground_truth=ground_truth,leftimg=image_left)
    tf.summary.scalar('loss', total_loss)
    with tf.name_scope('train'):
        #global_step = tf.Variable(0, trainable=False)
        #LEARNING_RATE = tf.train.exponential_decay(LEARNING_RATE_BASE ,global_step,100, LEARNING_RATE_DECAY,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE).minimize(total_loss)

    # important step
    # reload the model
    #tf.reset_default_graph()
    sess = tf.Session()
    #saver = tf.train.import_meta_graph(MODEL_DIR + '/DispNet.ckpt-970.meta')
    #saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
    #graph= tf.get_default_graph()


    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)

    left_images = sorted(os.listdir(DATA_DIR + '/left/'))
    right_images = sorted(os.listdir(DATA_DIR + '/right/'))
    ground_truths = os.listdir(GT_DIR)
    ground_truths.sort(key = lambda x : int(x[:-4]))
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
                    full_pic_name = GT_DIR + ground_truths[GT_SERIES[i + j]]
                    input_one_image = Image.open(full_pic_name)
                    input_one_image = (np.array(input_one_image))[:,:,[0]]
                    input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
                    if (j == 0):
                        input_gts = input_one_image
                    else:
                        input_gts = np.concatenate((input_gts, input_one_image), axis=0)
                result, optimizer_res, total_loss_res, loss0_res, loss1_res, loss2_res, loss3_res, loss4_res, loss5_res, loss6_res, pr6_res, pr5_res, pr4_res, pr3_res, pr2_res, pr1_res,pr0_res, loss6_inf_res = sess.run([merged, optimizer, total_loss,loss0, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2,pr1,pr0, loss6_inf],feed_dict={image_left: input_left_images, image_right: input_right_images, ground_truth: input_gts})
                writer.add_summary(result, round)
                if round == TRAINING_ROUNDS - 1:
                    final_result = (sess.run(final_output, feed_dict={image_left: input_left_images, image_right: input_right_images, ground_truth: input_gts}))
                    for k in range(10):
                        result = np.squeeze(final_result[k])
                        #result = result.astype(np.uint32)
                        writePFM(OUTPUT_DIR + '/' + str(i + k) + '.pfm', result)
                        #plt.imsave(OUTPUT_DIR + '/' + str(i+k) + '.png', result, format='png',cmap='gray')
                        file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(
                            total_loss_res) + ' loss0 ' + str(loss0_res) + '\n')
                if i == 0:
                    pr0_real_loss = np.sqrt(np.mean(np.square(input_gts-pr0_res)))
                    real_loss_record.append(pr0_real_loss)
                    print('Epoch %d :'%(round)+' pr0_real_loss ' + str(pr0_real_loss)+'  total_loss '+str(total_loss_res))
                    if(pr0_real_loss<minimum_loss):
                        minimum_loss=pr0_real_loss
                        saver.save(sess, MODEL_DIR+'/DispNet.ckpt',global_step=round+1)
                    file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(
                        total_loss_res) + ' loss1 ' + str(loss1_res) + '\n')

    # show lose curve
    x_axis = list(range(2000))
    plt.title('Learning rate=1e-5 ')
    plt.plot(x_axis, real_loss_record, color='blue', label='train loss')
    plt.savefig("train_loss_graph/FIG_09.jpg")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    main()

