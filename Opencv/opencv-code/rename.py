# 这是一个示例 Python 批量修改文件名脚本；
### 记得备份，运行完程序后，path的图像会消失；
import os

# 需要改变的图像文件的路径，我放于桌面了
path ='E:\\Deep Learning\\DL Book\\Opencv\\img'
# 改变后存放图片的文件夹路径，我也放于桌面了
path1 = 'E:\\Deep Learning\\DL Book\\Opencv\\img1'
filelist = os.listdir(path)

count = 0

for i in filelist:
    # 判断该路径下的文件是否为图片
    if i.endswith('.png'):#png可以改为jpg
        # 打开图片
        src = os.path.join(os.path.abspath(path), i)
        # 重命名
        # 0>s的意思是 图片的名称没有0，例如1.png，如果改为0>3s，则结果为001_.png
        dst = os.path.join(os.path.abspath(path1), format(str(count), '0>3s')+ '-BJLG-2-2-1009-1117'+'.png')

        # 执行操作
        os.rename(src, dst)
        count += 1
print("一共修改了"+str(count)+"个文件")
print("继续工作，奥里给！！！")