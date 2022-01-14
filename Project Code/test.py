# coding=utf-8
import cv2

from tensorflow.python.keras.models import load_model
import numpy as np
from datetime import datetime
import utils
# 上面的是导入相关的依赖库，包含numpy opencv tensorflow等依赖库

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

font = cv2.FONT_HERSHEY_SIMPLEX

CLASSES = (
    'NG', 'OK')
# class 类别

model = load_model('./models/mobilenet_v3.hdf5')
# 加载训练好的模型，修改该模型的名称可以加载不同的模型，model文件夹下面有两个模型

path = 'test/NG/0353.png'
path2 = 'test/NG/0353.png'
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# self.detection = self.img
# 缩放图片到指定的大小
img = cv2.resize(img, (680, 680), interpolation=cv2.INTER_AREA)

# 图片预处理
code = utils.ImageEncode(path)
code2 = utils.ImageEncode(path2)

# 图片预测
ret2 = model.predict(code2)   # 第一次推理时间较长

access_start = datetime.now()
print(access_start)

ret = model.predict(code)
print(ret)

# 打印处理时间
access_end = datetime.now()
print(access_end)
strTime = 'funtion time use:%fms' % (
        (access_end - access_start).seconds * 1000 + (access_end - access_start).microseconds / 1000)
print(strTime)

# 输入最大相似度的类别
if ret[0] > 0.8:
    res = 1
else:
    res = 0

# 打印最大相似度的类别
print('result:', CLASSES[res])

# 在图片上绘制出类别相似度
cv2.putText(img, str(float('%.4f' % np.max(ret[0, :])) * 100) + '%', (1, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
            thickness=2, lineType=2)

# 在图片上绘制出类别
cv2.putText(img, str(CLASSES[res]), (1, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
            thickness=2, lineType=2)

cv2.imshow('out', img)
cv2.imwrite("result.png",img)
cv2.waitKey(0)
