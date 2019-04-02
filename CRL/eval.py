import os
import tensorflow as tf
import numpy as np
from PIL import Image
from model import *
import _pickle as cPickle
import matplotlib.pyplot as plt
from prepfm import *
import re
MODEL_DIR = 'C:/Users/HK/PycharmProjects/CRL/model'
DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/frames_cleanpass/35mm_focallength/scene_forwards/fast'
EVAL_DATA_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/frames_cleanpass/35mm_focallength/scene_forwards/fast'
GT_DIR = 'C:/Users/HK/PycharmProjects/CRL/data/disparity/35mm_focallength/scene_forwards/fast/left/'
GT_TEST_DIR = 'C:/Users/HK/PycharmProjects/CRL/ground_truth/eval/35mm_focallength/forward/fast/left/'
LOGS_DIR = 'C:/Users/HK/PycharmProjects/CRL/logs'
EVAL_DIR = 'C:/Users/HK/PycharmProjects/CRL/eval_result/eval_04'
OUTPUT_DIR = 'C:/Users/HK/PycharmProjects/CRL/output/train_09'
TRAIN_SERIES = list(range(0,27))+list(range(30,57))+list(range(60,87))+list(range(90,117))+list(range(120,147))+list(range(150,177))+list(range(180,207))+list(range(210,237))+list(range(240,267))+list(range(270,297))
TEST_SERIES = list(set(list(range(300)))^set(TRAIN_SERIES))
GT_TEST_SERIES = list(range(30))
LEARNING_RATE_BASE = 5e-6
LEARNING_RATE_DECAY = 0.95
TRAIN_ROUND = 500
GT_SERIES = list(range(270))

def _norm(img):
    return (img-np.mean(img))/np.std(img)

def loss(pr,gt):
    return tf.sqrt(tf.reduce_mean(tf.square(pr-gt)))
def retrain():
    #load the graph and the variable
    tf.reset_default_graph()
    saver_restore = tf.train.import_meta_graph('./model/DispNet.ckpt-998.meta')
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
    loss6 = graph.get_tensor_by_name("pr6_loss6/Sqrt:0")
    loss5 = graph.get_tensor_by_name("pr5_loss5/Sqrt:0")
    loss4 = graph.get_tensor_by_name("pr5_loss5/Sqrt:0")
    loss3 = graph.get_tensor_by_name("pr4_loss4/Sqrt:0")
    loss2 = graph.get_tensor_by_name("pr3_loss3/Sqrt:0")
    loss1 = graph.get_tensor_by_name("pr2_loss2/Sqrt:0")
    loss0 = graph.get_tensor_by_name("pr1_loss1/Sqrt:0")

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
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE,name="retrain_Adam_03").minimize(total_loss)
    init= tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver_restore.restore(sess, tf.train.latest_checkpoint("./model"))
        saver = tf.train.Saver(max_to_keep=1)
        minimum_loss = 3.2
        for round in range(TRAIN_ROUND):
            #_step=round
            for i in range(0,270,10):
                for j in range(10):
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
                result, optimizer_res , pr0_res , total_loss_res = sess.run([merged,optimizer,pr0,total_loss],feed_dict=feed_dict)
                writer.add_summary(result, round)
                if round == TRAIN_ROUND-1:
                    output = sess.run(final_output,feed_dict=feed_dict)
                    for k in range(10):
                        result = np.squeeze(output[k])
                        #result = result.astype(np.uint32)
                        writePFM(EVAL_DIR + '/' + str(i + k) + '.pfm', result)
                        #plt.imsave(OUTPUT_DIR + '/' + str(i+k) + '.png', result, format='png', cmap='gray')
                if i == 0:
                    pr0_real_loss = np.sqrt(np.mean(np.square(input_gts - pr0_res)))
                    real_loss_record.append(pr0_real_loss)
                    print('Epoch %d :' % (round) + ' pr0_real_loss ' + str(pr0_real_loss) + '  total_loss ' + str(total_loss_res)+ '  learning_rate ' + str(LEARNING_RATE_BASE))
                    if (pr0_real_loss < minimum_loss):
                        minimum_loss = pr0_real_loss
                        saver.save(sess, MODEL_DIR + '/DispNet.ckpt', global_step=round + 1)
    # show lose curve
        x_axis = list(range(500))
        plt.title('Learning rate=5e-6 ')
        plt.plot(x_axis, real_loss_record, color='blue', label='train loss')
        plt.savefig("train_loss_graph/FIG_10.jpg")
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.show()

def eval():
    tf.reset_default_graph()
    saver_restore = tf.train.import_meta_graph("./model/DispNet.ckpt-998.meta")
    graph = tf.get_default_graph()
    #get the parameter
    image_left = graph.get_tensor_by_name("image_left:0")
    image_right = graph.get_tensor_by_name("image_right:0")
    ground_truth = graph.get_tensor_by_name("ground_truth:0")
    final_output = graph.get_tensor_by_name("pr0_loss0/LeakyRelu:0")

    left_images = sorted(os.listdir(EVAL_DATA_DIR + '/left/'))
    right_images = sorted(os.listdir(EVAL_DATA_DIR + '/right/'))
    ground_truths = os.listdir(GT_DIR)
    ground_truths.sort(key=lambda x: int(x[:-4]))


    with tf.Session() as sess:
        saver_restore.restore(sess,tf.train.latest_checkpoint("./model"))
        for i in range(0,30,10):
            for j in range(10):
                full_pic_name = DATA_DIR + '/left/' + left_images[TEST_SERIES[i + j]]
                input_one_image = Image.open(full_pic_name)
                input_one_image = _norm(np.reshape(input_one_image,(1,IMAGE_SIZE_Y,IMAGE_SIZE_X,3)))
                if j==0:
                    input_left_image = input_one_image
                else:
                    input_left_image = np.concatenate((input_left_image,input_one_image),axis=0)

                full_pic_name = DATA_DIR + '/right/' + right_images[TEST_SERIES[i + j]]
                input_one_image = Image.open(full_pic_name)
                input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
                if j == 0:
                    input_right_image = input_one_image
                else:
                    input_right_image = np.concatenate((input_right_image, input_one_image), axis=0)
                full_pic_name = GT_TEST_DIR + ground_truths[GT_TEST_SERIES[i + j]]
                input_one_image = Image.open(full_pic_name)
                input_one_image = (np.array(input_one_image))[:, :, [0]]
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
                plt.imsave(EVAL_DIR + '/' + str(i + k) + '.png', result, format='png', cmap='gray')

if __name__== '__main__':
    retrain()
    #eval()
