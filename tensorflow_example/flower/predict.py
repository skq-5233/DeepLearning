#coding:utf-8
import os, cv2
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu
 
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import glob
 
import model
 
 
N_CLASSES = 5
IMG_W = 224
IMG_H = IMG_W
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0
label_dict = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}
label_dict_res = {v:k for k,v in label_dict.items()}
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
 
 
def init_tf(logs_train_dir = './model_save/model.ckpt-24000'):
    global sess, pred, x
    # process image
    x = tf.placeholder(tf.float32, shape=[IMG_W, IMG_W, 3])
    x_norm = tf.image.per_image_standardization(x)
    x_4d = tf.reshape(x_norm, [1, IMG_W, IMG_W, 3])
    # predict
    logit = model.model4(x_4d, N_CLASSES, is_trian=False)
    #logit = model.model2(x_4d, batch_size=1, n_classes=N_CLASSES)
    pred = tf.nn.softmax(logit)
 
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, logs_train_dir)
    print('load model done...')
 
def evaluate_image(img_dir):
    # read image
    im = cv2.imread(img_dir)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (IMG_W, IMG_W))
    image_array = np.array(im)
 
    prediction = sess.run(pred, feed_dict={x: image_array})
    max_index = np.argmax(prediction)
    print("%s, predict: %s, prob: %f" %(os.path.basename(img_dir), label_dict_res[max_index], prediction[0][max_index]))
    
 
if __name__ == '__main__':
    init_tf()
    # data_path = 'flowers/flower_photos'
    # label = os.listdir(data_path)
    # for l in label:
    #     if os.path.isfile(os.path.join(data_path, l)):
    #         continue
    #     for img in glob.glob(os.path.join(data_path, l, "*.jpg")):
    #         print(img)
    #         evaluate_image(img_dir=img)
    for img in glob.glob("./jpg/*.jpg"):
        evaluate_image(img)
    sess.close()
