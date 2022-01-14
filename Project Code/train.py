# coding=utf-8
import os
os.environ["CUDA.VISIABLE.DEVICES"] = "0"

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3Large
from tensorflow.python.keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard

import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

# 图片路径
DATASET_PATH = 'train'  # 训练数据集文件夹
DATATEST_PATH = 'test'  # 测试数据集文件夹

# 图片大小
IMAGE_SIZE = (224, 224)

# 图片类别数
NUM_CLASSES = 1

# 若CPU/GPU性能不足，可降低batch size或冻结更多网络层
BATCH_SIZE = 16

# 冻结网络层数
FREEZE_LAYERS = 2

# Epoch 数
NUM_EPOCHS = 60

TRAIN_TYPE = 0  # 0：重新训练，1:在预训练权重基础上继续训练


def my_lr_scheduler(epoch, lr):
    if epoch < 50:
        return 1e-4
    elif epoch < 100:
        return 2e-4
    else:
        return lr * 0.9 + 1e-5 * 0.1


# 模型输出文件名
WEIGHTS_FINAL = './models/mobilenet_v3.hdf5'

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='binary',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()

valid_batches = valid_datagen.flow_from_directory(DATATEST_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='binary',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# 输出各类别的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# 以训练好的的MobileNet为基础来建立模型，
# 拾取 MobileNet 顶层的 fully connected layers
net = MobileNetV3Large(alpha=1.0, include_top=False, weights=None,
                       input_tensor=None,
                       input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

net.load_weights('weights_mobilenet_v3_large_224_1.0_float_no_top.h5', by_name=True)

x = net.output
x = Flatten()(x)

x = Dense(64, activation='relu')(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加Dense layer，以 softmax 产生各类别的几率值
output_layer = Dense(NUM_CLASSES, activation='sigmoid', name='sigmoid')(x)

# 设定冻结与要进行训练的网络层
net_final = Model(inputs=net.input, outputs=output_layer)

for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
# ======================================================================================================================

filepath = './models/weights.best.hdf5'

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max',
                             period=1)

learning_rate_scheduler = LearningRateScheduler(my_lr_scheduler)

reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_accuracy',
                                         factor=0.1,
                                         patience=5,
                                         min_delta=1e-3)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=10,
                               mode='max',
                               min_delta=1e-2)

tensorboard = TensorBoard(log_dir='./logs',
                          update_freq=NUM_EPOCHS,
                          histogram_freq=1,
                          write_graph=False)

# callbacks_list = [checkpoint, learning_rate_scheduler, early_stopping, tensorboard]
callbacks_list = [checkpoint]

# 使用 Adam optimizer，以较低的 learning rate 进行 fine-tuning
net_final.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=1e-4))
# ======================================================================================================================
# 输出整个网络结 构
print(net_final.summary())

# 训练模型
if TRAIN_TYPE:  # 0：重新训练，1:在预训练权重基础上继续训练
    del net_final
    net_final = load_model('./models/mobilenet_v3.hdf5')

hist = net_final.fit_generator(train_batches,
                               steps_per_epoch=train_batches.samples // BATCH_SIZE,
                               validation_data=valid_batches,
                               validation_steps=valid_batches.samples // BATCH_SIZE,
                               epochs=NUM_EPOCHS,
                               callbacks=callbacks_list)


def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b-')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy(blue:acc,red:val_acc)')
    plt.savefig('acc_l.png')
    plt.figure()
    plt.plot(epochs, loss, 'b-')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss(blue:loss,red:val_loss)')
    plt.savefig('loss_l.png')


plot_training(hist)

# 保存训练好的模型
# net_final.save(WEIGHTS_FINAL)  # 此种保存方式转换为pb格式时报错
save_model_to_hdf5(net_final, WEIGHTS_FINAL, overwrite=True, include_optimizer=True)
