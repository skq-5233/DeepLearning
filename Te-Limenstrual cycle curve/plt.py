#coding:utf-8
# 导入本次需要的模块
#  数据可视化（画图）0730
from matplotlib import pyplot as plt
# 绘图；
fig = plt.figure(figsize=(8,6),dpi=80)  ## 设置图片大小；dpi：图像每英寸长度内的像素点数；
x = range(2,14)   # [2,26]间隔为2；
y = [15,13,14.5,17,20,25,26,26,24,22,18,15]
plt.plot(x,y)
# 添加标题
plt.title('temperature')
## 设置 X轴刻度；
plt.xticks(x[::1]) ### 间隔为1；
#plt.xticks(range(2,26))  ### 间隔为1；

#设置更加紧凑的 X轴刻度；
# xtick_labels = [i/2 for i in range(2,49)]  ### x刻度为0.5；
# plt.xticks(xtick_labels)

#### 设置y轴刻度；
plt.yticks(range(min(y),max(y)+1))

## 保存图片
# plt.savefig("./picture_1.png")
# 展示图形；
plt.show()




