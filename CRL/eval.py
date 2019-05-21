import os
import tensorflow as tf
import numpy as np
from PIL import Image
from model import *
import _pickle as cPickle
import matplotlib.pyplot as plt
from prepfm import *
import datetime
import re
MODEL_DIR = 'C:/Users/HK/PycharmProjects/CRL/model'
#DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/frames_cleanpass/35mm_focallength/scene_forwards/fast'
#DATA_DIR = 'E:/FlyingThings3D_subset/train/image_clean'
DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/medical data/image'
#EVAL_DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/frames_cleanpass/35mm_focallength/scene_forwards/fast'
EVAL_DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/medical data/rec_image'
#GT_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/disparity/35mm_focallength/scene_forwards/fast/left/'
GT_TEST_DIR = 'C:/Users/HK/PycharmProjects/CRL/ground_truth/eval/35mm_focallength/forward/fast/left/'
GT_DIR = 'E:/FlyingThings3D_subset/val/disparity/right/'
#GT_DIR = 'C:/Users/HK/PycharmProjects/CRL/medical data/disp/'
LOGS_DIR = 'C:/Users/HK/PycharmProjects/CRL/logs'
EVAL_DIR = 'C:/Users/HK/PycharmProjects/CRL/eval_result/eval_07'
OUTPUT_DIR = 'C:/Users/HK/PycharmProjects/CRL/output/train_14/second_stage/medical/'
#DISP_DIR = 'C:/Users/HK/PycharmProjects/CRL/disparity_1/car/'
DISP_DIR = 'C:/Users/HK/PycharmProjects/CRL/disparity_1/medical/'
SYN_IMAGE_DIR = 'C:/Users/HK/PycharmProjects/CRL/syn_image/medical/'


#TRAIN_SERIES = list(range(0,27))+list(range(30,57))+list(range(60,87))+list(range(90,117))+list(range(120,147))+list(range(150,177))+list(range(180,207))+list(range(210,237))+list(range(240,267))+list(range(270,297))
#TEST_SERIES = list(set(list(range(300)))^set(TRAIN_SERIES))
TEST_SERIES = list(range(450))
GT_TEST_SERIES = list(range(450))
LEARNING_RATE_BASE = 1e-5
LEARNING_RATE_DECAY = 0.95
TRAIN_ROUND = 2000
TRAIN_SERIES = list(range(600))
image_num = np.size(TRAIN_SERIES)
#GT_SERIES = list(range(270))

def _norm(img):
    return (img-np.mean(img))/np.std(img)

def epe_loss(pre,gt):
    return tf.abs(pre-gt)

def smooth_l1_loss(pre,gt):
    # smooth_l1_loss = 0.5x^2 ,|x|<1
    #                = |x|-0.5, otherwise
    loss = tf.reduce_mean(tf.where(tf.greater(1.0,tf.abs(pre-gt)),0.5*(pre-gt)**2,tf.abs(pre-gt)-0.5))
    return loss

def retrain():
    #load the graph and the variable
    tf.reset_default_graph()
    saver_restore = tf.train.import_meta_graph('C:/Users/HK/PycharmProjects/CRL/model/DispFulNet.ckpt-1957.meta')
    graph = tf.get_default_graph()

    image_left_1 = graph.get_tensor_by_name("image_left:0")
    image_right_1 = graph.get_tensor_by_name("image_right:0")
    ground_truth_1 = graph.get_tensor_by_name("ground_truth:0")
    #bili_weight_1 = graph.get_tensor_by_name("bili_weight:0")
    final_output = graph.get_tensor_by_name("pr0_loss0/LeakyRelu:0")
    #multiscale disparity
    pr6 = graph.get_tensor_by_name("pr6_loss6/LeakyRelu:0")
    pr5 = graph.get_tensor_by_name("pr5_loss5/LeakyRelu:0")
    pr4 = graph.get_tensor_by_name("pr4_loss4/LeakyRelu:0")
    pr3 = graph.get_tensor_by_name("pr3_loss3/LeakyRelu:0")
    pr2 = graph.get_tensor_by_name("pr2_loss2/LeakyRelu:0")
    pr1 = graph.get_tensor_by_name("pr1_loss1/LeakyRelu:0")
    pr0 = graph.get_tensor_by_name("pr0_loss0/LeakyRelu:0")

    # loss
    # loss6 = graph.get_tensor_by_name("pr6_loss6/Sqrt:0")
    # loss5 = graph.get_tensor_by_name("pr5_loss5/Sqrt:0")
    # loss4 = graph.get_tensor_by_name("pr5_loss5/Sqrt:0")
    # loss3 = graph.get_tensor_by_name("pr4_loss4/Sqrt:0")
    # loss2 = graph.get_tensor_by_name("pr3_loss3/Sqrt:0")
    # loss1 = graph.get_tensor_by_name("pr2_loss2/Sqrt:0")
    # loss0 = graph.get_tensor_by_name("pr1_loss1/Sqrt:0")
    loss6 = graph.get_tensor_by_name("pr6_loss6/Mean:0")
    loss5 = graph.get_tensor_by_name("pr5_loss5/Mean:0")
    loss4 = graph.get_tensor_by_name("pr4_loss4/Mean:0")
    loss3 = graph.get_tensor_by_name("pr3_loss3/Mean:0")
    loss2 = graph.get_tensor_by_name("pr2_loss2/Mean:0")
    loss1 = graph.get_tensor_by_name("pr1_loss1/Mean:0")
    loss0 = graph.get_tensor_by_name("pr0_loss0/Mean:0")
    EPE = graph.get_tensor_by_name("loss/Mean:0")

    real_loss_record = []
    total_loss = loss0+1/2*loss1+1/4*loss2+1/8*loss3+1/16*loss4+1/32*loss5+1/32*loss6
    tf.summary.scalar('total_loss',total_loss)
    #print(image_left_1)
    #print(image_right_1)
    #print(ground_truth_1)
    #print(final_output)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGS_DIR, tf.Session().graph)

    left_images = sorted(os.listdir(DATA_DIR + '/left/'))
    right_images = sorted(os.listdir(DATA_DIR + '/right/'))
    ground_truths = sorted(os.listdir(GT_DIR))
    #ground_truths = os.listdir(GT_DIR)
    #ground_truths.sort(key = lambda x:int(x[:-4]))

    with tf.name_scope('train_1'):
        #_step = tf.Variable(0, trainable=False)
        #LEARNING_RATE = tf.train.exponential_decay(LEARNING_RATE_BASE, _step, 100, LEARNING_RATE_DECAY)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE,name="retrain").minimize(total_loss)
    init= tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver_restore.restore(sess, tf.train.latest_checkpoint("C:/Users/HK/PycharmProjects/CRL/model"))
        saver = tf.train.Saver(max_to_keep=1)
        minimum_loss = 2.2
        for round in range(TRAIN_ROUND):
            #_step=round
            for i in range(0,image_num,BATCH_SIZE):
                for j in range(BATCH_SIZE):
                    full_pic_name = DATA_DIR + '/left/' + left_images[TRAIN_SERIES[i+j]]
                    input_one_image = Image.open(full_pic_name)
                    input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                    if (j == 0):
                        input_left_images = input_one_image
                    else:
                        input_left_images = np.concatenate((input_left_images, input_one_image), axis=0)
                    full_pic_name = DATA_DIR + '/right/' + right_images[TRAIN_SERIES[i+j]]
                    input_one_image = Image.open(full_pic_name)
                    input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                    if (j == 0):
                        input_right_images = input_one_image
                    else:
                        input_right_images = np.concatenate((input_right_images, input_one_image), axis=0)
                    full_pic_name = GT_DIR + ground_truths[TRAIN_SERIES[i + j]]
                    input_one_image ,scale = load_pfm(full_pic_name)
                    #input_one_image = (np.array(input_one_image))[:, :, [0]]
                    input_one_image = np.reshape(input_one_image,(1,IMAGE_SIZE_Y,IMAGE_SIZE_X))
                    input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
                    if (j == 0):
                        input_gts = input_one_image
                    else:
                        input_gts = np.concatenate((input_gts, input_one_image), axis=0)
                feed_dict = {image_left_1: input_left_images, image_right_1: input_right_images,ground_truth_1: input_gts}
                result, optimizer_res , pr0_res , total_loss_res,loss0_res,epe_res = sess.run([merged,optimizer,pr0,total_loss,loss0,EPE],feed_dict=feed_dict)
                writer.add_summary(result, round)
                if round == TRAIN_ROUND-1:
                    output = sess.run(final_output,feed_dict=feed_dict)
                    for k in range(10):
                        result = np.squeeze(output[k])
                        #result = result.astype(np.uint32)
                        writePFM(EVAL_DIR + '/' + str(i + k) + '.pfm', result)
                        #plt.imsave(OUTPUT_DIR + '/' + str(i+k) + '.png', result, format='png', cmap='gray')
                if i == 0:
                    pr0_real_loss = loss0_res
                    real_loss_record.append(pr0_real_loss)
                    print('Epoch %d :' % (round) + ' pr0_real_loss ' + str(pr0_real_loss) + '  total_loss ' + str(total_loss_res)+ '  EPE ' + str(epe_res))
                    if (pr0_real_loss < minimum_loss):
                        minimum_loss = pr0_real_loss
                        saver.save(sess, MODEL_DIR + '/DispFulNet.ckpt', global_step=round + 1)
    # show lose curve
        x_axis = list(range(2000))
        plt.title('Learning rate=1e-5 ')
        plt.plot(x_axis, real_loss_record, color='blue', label='train loss')
        plt.savefig("C:/Users/HK/PycharmProjects/CRL/train_loss_graph/FIG_11.jpg")
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.show()

def eval():
    tf.reset_default_graph()
    saver_restore = tf.train.import_meta_graph("C:/Users/HK/PycharmProjects/CRL/model/DispFulNet.ckpt-1957.meta")
    graph = tf.get_default_graph()
    #get the parameter
    image_left = graph.get_tensor_by_name("image_left:0")
    image_right = graph.get_tensor_by_name("image_right:0")
    ground_truth = graph.get_tensor_by_name("ground_truth:0")
    final_output = graph.get_tensor_by_name("pr0_loss0/LeakyRelu:0")

    left_images = os.listdir(EVAL_DATA_DIR + '/left/')[1:]
    left_images.sort(key=lambda x: int(x[2:-4]))
    right_images = os.listdir(EVAL_DATA_DIR + '/right/')[1:]
    right_images.sort(key=lambda x: int(x[2:-4]))
    # ground_truths = os.listdir(GT_DIR)
    # ground_truths.sort(key=lambda x: int(x[5:-4]))
    # left_images = sorted(os.listdir(EVAL_DATA_DIR+ '/left/'))[1:]
    # right_images = sorted(os.listdir(EVAL_DATA_DIR+'/right/'))[1:]
    ground_truths = sorted(os.listdir(GT_DIR))


    with tf.Session() as sess:
        saver_restore.restore(sess,tf.train.latest_checkpoint("C:/Users/HK/PycharmProjects/CRL/model"))
        for i in range(0,450,10):
            for j in range(10):
                full_pic_name = EVAL_DATA_DIR + '/left/' + left_images[TEST_SERIES[i + j]]
                input_one_image = Image.open(full_pic_name)
                input_one_image = _norm(np.reshape(input_one_image,(1,IMAGE_SIZE_Y,IMAGE_SIZE_X,3)))
                if j==0:
                    input_left_image = input_one_image
                else:
                    input_left_image = np.concatenate((input_left_image,input_one_image),axis=0)

                full_pic_name = EVAL_DATA_DIR + '/right/' + right_images[TEST_SERIES[i + j]]
                input_one_image = Image.open(full_pic_name)
                input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                if j == 0:
                    input_right_image = input_one_image
                else:
                    input_right_image = np.concatenate((input_right_image, input_one_image), axis=0)
                full_pic_name = GT_DIR + ground_truths[GT_TEST_SERIES[i + j]]
                input_one_image, scale = load_pfm(full_pic_name)
                # input_one_image = (np.array(input_one_image))[:, :, [0]]
                input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X))
                input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
                if j==0:
                    input_gts = input_one_image
                else:
                    input_gts = np.concatenate((input_gts,input_one_image),axis=0)
            feed_dict = {image_left:input_left_image,image_right:input_right_image,ground_truth:input_gts}
            output = sess.run(final_output , feed_dict=feed_dict)
            for k in range(10):
                result = np.squeeze(output[k])
                #result = result.astype(np.uint16)
                writePFM(EVAL_DIR + '/' + str(i + k) + '.pfm',result)
                #plt.imsave(EVAL_DIR + '/' + str(i + k) + '.png', result, format='png', cmap='gray')

def eval2():
    tf.reset_default_graph()
    saver_restore = tf.train.import_meta_graph("C:/Users/HK/PycharmProjects/CRL/model/DispResNet.ckpt-989.meta")
    graph = tf.get_default_graph()
    # get the parameter
    image_left = graph.get_tensor_by_name("image_left:0")
    image_right = graph.get_tensor_by_name("image_right:0")
    ground_truth = graph.get_tensor_by_name("ground_truth:0")
    syn_left = graph.get_tensor_by_name("image_left_syn:0")
    disp1 = graph.get_tensor_by_name("disparity_1:0")
    err = graph.get_tensor_by_name("error:0")
    final_output = graph.get_tensor_by_name("pr_s2/LeakyRelu:0")

    left_images = os.listdir(EVAL_DATA_DIR + '/left/')[1:]
    left_images.sort(key=lambda x: int(x[2:-4]))
    right_images = os.listdir(EVAL_DATA_DIR + '/right/')[1:]
    right_images.sort(key=lambda x: int(x[2:-4]))
    disparitys = os.listdir(DISP_DIR)
    disparitys.sort(key=lambda x: int(x[:-4]))
    syn_images = os.listdir(SYN_IMAGE_DIR)
    syn_images.sort(key=lambda x: int(x[:-4]))
    ground_truths = sorted(os.listdir(GT_DIR))

    with tf.Session() as sess:
        saver_restore.restore(sess, tf.train.latest_checkpoint("C:/Users/HK/PycharmProjects/CRL/model"))
        for i in range(0, 270, 3):
            for j in range(3):
                full_pic_name = EVAL_DATA_DIR + '/left/' + left_images[TRAIN_SERIES[i + j]]
                input_one_image_l = Image.open(full_pic_name)
                input_one_image_l = _norm(np.reshape(input_one_image_l, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                if (j == 0):
                    input_left_images = input_one_image_l
                else:
                    input_left_images = np.concatenate((input_left_images, input_one_image_l), axis=0)
                full_pic_name = EVAL_DATA_DIR + '/right/' + right_images[TRAIN_SERIES[i + j]]
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
                input_one_image_d, scale = load_pfm(full_pic_name)
                input_one_image_d = np.reshape(input_one_image_d, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X))
                input_one_image_d = np.reshape(input_one_image_d, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
                if (j == 0):
                    input_pr = input_one_image_d
                else:
                    input_pr = np.concatenate((input_pr, input_one_image_d), axis=0)

                # input_one_image_s = warp(input_one_image_r,input_one_image_d)
                full_pic_name = SYN_IMAGE_DIR + syn_images[TRAIN_SERIES[i + j]]
                input_one_image_s = plt.imread(full_pic_name)[:, :, 0:3]
                input_one_image_s = _norm(np.reshape(input_one_image_s, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))

                if (j == 0):
                    input_syn = input_one_image_s
                else:
                    input_syn = np.concatenate((input_syn, input_one_image_s), axis=0)
                input_one_image_e = np.abs(input_one_image_l - input_one_image_s)
                if (j == 0):
                    input_err = input_one_image_e
                else:
                    input_err = np.concatenate((input_err, input_one_image_e), axis=0)
            feed_dict = {image_left: input_left_images, image_right: input_right_images, ground_truth: input_gts,
                         disp1: input_pr, syn_left: input_syn, err: input_err}

            output = sess.run(final_output, feed_dict=feed_dict)
            for k in range(3):
                result = np.squeeze(output[k])
                # result = result.astype(np.uint16)
                writePFM(OUTPUT_DIR  + str(i + k) + '.pfm', result)
                # plt.imsave(EVAL_DIR + '/' + str(i + k) + '.png', result, format='png', cmap='gray')


if __name__== '__main__':
    #retrain()
    begin=datetime.datetime.now()
    eval2()
    end=datetime.datetime.now()
    print ((end-begin).seconds)
