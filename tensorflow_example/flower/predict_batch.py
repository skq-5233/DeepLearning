#coding:utf-8
import os, cv2, time
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
batch_size = 8
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0
label_dict = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}
label_dict_res = {v:k for k,v in label_dict.items()}
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
 
 
def get_imgpath(path):
    img_list = []
    for fpath , dirs , fs in os.walk(path):
        for f in fs:
            img_path = os.path.join(fpath , f)
            if os.path.dirname(img_path) == os.getcwd():
                continue
            if not os.path.isfile(img_path):
                continue
            if os.path.basename(img_path)[-3:] == "jpg":
                img_list.append(img_path)
    return img_list
 
 
def init_tf(logs_train_dir = './model_save/model.ckpt-24000'):
    global sess, pred, x
    # process image
    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_W, 3])
    # predict
    logit = model.model4(x, N_CLASSES, is_trian=False)
    #logit = model.model2(x_4d, batch_size=1, n_classes=N_CLASSES)
    pred = tf.nn.softmax(logit)
 
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, logs_train_dir)
    print('load model done...')
 
def evaluate_image(img_dir):
    # read and process image
    batch_img = []
    for img in img_dir:    
        im = cv2.imread(img)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (IMG_W, IMG_W))
        im_mean = np.mean(im)
        stddev = max(np.std(im), 1.0/np.sqrt(IMG_W*IMG_H*3))
        im = (im - im_mean) / stddev
        image_array = np.array(im)
        batch_img.append(image_array)
    # output sotfmax
    prediction = sess.run(pred, feed_dict={x: batch_img})
    for i in range(len(img_dir)):
        img = img_dir[i]
        max_index = np.argmax(prediction[i])
        print("img:%s, predict: %s, prob: %f" % (img, label_dict_res[max_index], prediction[i][max_index]))
    
 
if __name__ == '__main__':
    init_tf()
    data_path = './jpg'
    img_list = get_imgpath(data_path)
    total_batch = len(img_list)/batch_size
    start = time.time()
    for i in range(int(total_batch)):
        print(str(i) + "-"*50)
        batch_img = img_list[i*batch_size: (i+1)*batch_size]
        evaluate_image(batch_img)
    print("time cost:", time.time()-start)
    sess.close()
