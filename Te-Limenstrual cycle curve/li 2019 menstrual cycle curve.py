#coding:utf-8
# 导入本次需要的模块
###0706
#  数据可视化（画图）0730
from matplotlib import pyplot as plt
x = range(1,14,1)
y = [8,4,6,1,30,25,20,27,24,22,18,16,17,]
plt.plot(x,y)
# 标题
plt.title('Teacher li 2019 menstrual cycle curve')
plt.show()
