#coding:utf-8
# 导入本次需要的模块
###0810

from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  ## 解决中文乱码问题；
import numpy as np
# import pylab as pl

# 设置图片大小；dpi：图像每英寸长度内的像素点数；
fig = plt.figure(figsize=(8,6),dpi=80)

x=range(1,14)
y= [8,4,6,1,30,25,20,27,24,22,18,16,17,]

x1=range(1,14)
y1=[19,16,16,12,10,6,3,30,29,25,23,18,14]

x2=range(1,14)
y2=[12,9,7,7,5,3,1,29,25,19,21,16,14]

x3=range(1,3)
y3=[10,7]

plt.plot(x,y,'b',label='2019')
plt.plot(x1,y1,'r',label='2020')
plt.plot(x2,y2,'g',label='2021')
plt.plot(x3,y3,'y',label='2022')

plt.title("Teacher's_Li Menstrual Cycle Curve",fontsize=18)
plt.xlabel('Month',fontsize=14)
plt.ylabel('Data',fontsize=14)
# pl.xlim(1,13)
# pl.ylim(1,31)
plt.legend(loc='best')  ## 出现图标；
## 设置x轴刻度；
#plt.xticks(range(1,14))  ### 间隔为1；

plt.xticks(x[::1])  ### 间隔为1；
## 保存图片
plt.savefig("李老师曲线图2019-2022.png")
plt.show()


