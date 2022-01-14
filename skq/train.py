# coding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.python.keras.layers import Flatten, Dropout, Reshape
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import pandas as pd
import random
import cv2


# 图片路径
DATATRAIN_PATH = 'D:/software/DL information/sun/img/train'  # 训练数据集文件夹
DATATEST_PATH = 'D:/software/DL information/sun/img/test'  # 测试数据集文件夹

# 图片大小
IMAGE_SIZE = (256, 512)

# 若CPU/GPU性能不足，可降低batch size
BATCH_SIZE = 8

# Epoch 迭代次数
NUM_EPOCHS = 15

train_datagen = ImageDataGenerator(rescale=1 / 255.0)  # 数据增强会影响图片解码效果，只进行归一化处理

x_train = train_datagen.flow_from_directory(DATATRAIN_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  color_mode='rgb',
                                                  class_mode='input',
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True)



valid_datagen = ImageDataGenerator(rescale=1 / 255.0)

x_test = valid_datagen.flow_from_directory(DATATEST_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  color_mode='rgb',
                                                  class_mode='input',
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False)

noise_factor = 0.7
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

plt.figure(figsize=(20, 2))
for i in range(1, 5 + 1):
    ax = plt.subplot(1, 5, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
