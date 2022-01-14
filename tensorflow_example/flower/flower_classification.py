#coding:utf-8
import os
import numpy as np
import tensorflow as tf
import glob
 
import model
 
 
init_lr = 0.001
decay_steps = 10000
MAX_STEP = 200000
N_CLASSES = 5
IMG_W = 224
IMG_H = 224
BATCH_SIZE = 32
CAPACITY = 2000
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # gpu编号
label_dict = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4} # 手动指定一个名字到label的映射关系,必须从0开始
 
train_dir = './flower_photos'  # 该文件下放着各类图像的子文件夹这里有5个
logs_train_dir = './model_save'
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 设置最小gpu使用量
 
 
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 数据增强
    #image = tf.image.resize_image_with_pad(image, target_height=image_W, target_width=image_H)
    image = tf.image.resize_images(image, (image_W, image_H))
    # 随机左右翻转
    image = tf.image.random_flip_left_right(image)
    # 随机上下翻转
    image = tf.image.random_flip_up_down(image)
    # 随机设置图片的亮度
    image = tf.image.random_brightness(image, max_delta=32/255.0)
    # 随机设置图片的对比度
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # 随机设置图片的色度
    image = tf.image.random_hue(image, max_delta=0.3)
    # 随机设置图片的饱和度
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # 标准化,使图片的均值为0，方差为1
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
 
 
def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        for img in glob.glob(os.path.join(file_dir, label, "*.jpg")):
            image_list.append(img)
            label_list.append(label_dict[label])
    print('There are %d data' %(len(image_list)))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list
 
 
def main():    
    global_step = tf.Variable(0, name='global_step', trainable=False)    
    # dataset
    train, train_label = get_files(train_dir)
    # label without one-hot
    batch_train, batch_labels = get_batch(train,
                                          train_label,
                                          IMG_W,
                                          IMG_H,
                                          BATCH_SIZE, 
                                          CAPACITY)
    # network
    #logits = model.model2(batch_train, BATCH_SIZE, N_CLASSES)
    logits = model.model4(batch_train, N_CLASSES, is_trian=True)
    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('train_loss', loss)
    # optimizer
    lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step, decay_steps=decay_steps, decay_rate=0.1)
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    # accuracy
    correct = tf.nn.in_top_k(logits, batch_labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('train_acc', accuracy)
    
    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver.restore(sess, logs_train_dir+'/model.ckpt-174000') 
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                    break
            _, learning_rate, tra_loss, tra_acc = sess.run([optimizer, lr, loss, accuracy])
            if step % 50 == 0:
                print('Step %4d, lr %f, train loss = %.2f, train accuracy = %.2f%%' %(step, learning_rate, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
 
if __name__ == '__main__':
    main()
