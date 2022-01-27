# -*- coding: utf-8 -*-
# 000
# import numpy as np
# import cv2

# # Load an color image in grayscale
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0) # cv2.imread(读取图像)；
# cv2.imshow('image',img)  # imshow(显示图像)；
# cv2.waitKey(0)&0xFF  # 64 位系统，使用&0xFF;
# cv2.destroyAllWindows()
# cv2.imwrite("D:/skq/DL/DL informaton/Opencv/VR-Gray.jpg",img) # cv2.imwrite(保存图像)；
#
# print(img)

# 001
# import numpy as np
# import cv2
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0) # cv2.imread(读取图像)；
# cv2.namedWindow('image',cv2.WINDOW_NORMAL) # cv2.WINDOW_NORMAL,可手动调整显示图像的窗口大小;
# cv2.imshow('image',img)
# cv2.waitKey(0)&0xFF  # 64 位系统，使用&0xFF;
# cv2.destroyAllWindows()
#
# cv2.imwrite("D:/skq/DL/DL informaton/Opencv/VR-Gray1.png",img)   # cv2.imwrite(保存图像);


# 002
# import numpy as np
# import cv2
#
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0)
# cv2.imshow('image',img)
# k = cv2.waitKey(0)&0xFF  # 64 位系统，使用&0xFF;
# if k == 27: # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('D:/skq/DL/DL informaton/Opencv/VR002.png',img)
# cv2.destroyAllWindows()

# 003
# 使用 Matplotlib

# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0)
#
# plt.imshow(img,cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
# plt.show()
# cv2.imwrite('D:/skq/DL/DL informaton/Opencv/VR003.png',img)


# 004
# 视频
# 004_1 用摄像头捕获视频

# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read() # cap.read() 返回一个布尔值（True/False）;
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

# 004_2 从文件中播放视频;
# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         frame = cv2.flip(frame, 0)
#         # write the flipped frame
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # 播放视频；
# import cv2
# cv =cv2.VideoCapture("Fall.mp4")
# # 检查视频是否正确打开；
# if cv.isOpened():
#     open,frame = cv.read()
# else:
#     open = False
# while open:
#     ret,frame = cv.read()
#     if frame is None:
#         break
#     if ret == True:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         cv2.imshow("result",gray)
#         if cv2.waitKey(10) &0xFF == 27:
#             break
# cv.release()
# cv2.destroyAllWindows()


# 004_3 保存视频;
# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
# # Define the codec and create VideoWriter object
# fourcc = cv2.cv.FOURCC(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         frame = cv2.flip(frame, 0)
#         # write the flipped frame
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# 005 OpenCV 中的绘图函数;
# 005_1 画线；
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# # Create a black image
# img=np.zeros((512,512,3), np.uint8)
# # print(img)
# # Draw a diagonal blue line with thickness of 5 px
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# cv2.imshow("line",img)
# cv2.imwrite("E:\\Deep Learning\\DeepLearning\\Opencv\\Blue_Line.jpg",img)
# cv2.waitKey(0)
# plt.show()

# 005_2 画矩形；
# import numpy as np
# import cv2
# img=np.zeros((512,512,3), np.uint8)
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
# cv2.imshow("rectangle",img)
# cv2.imwrite("E:\\Deep Learning\\DeepLearning\\Opencv\\rectangle.jpg",img)
# cv2.waitKey(0)

# # 005_3 画圆；
# import numpy as np
# import cv2
# img=np.zeros((512,512,3), np.uint8)
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# cv2.circle(img,(447,63), 63, (0,0,255), -1)
# cv2.imshow("circle",img)
# cv2.imwrite("E:\\Deep Learning\\DeepLearning\\Opencv\\circle.jpg",img)
# cv2.waitKey(0)

# 005_4 画椭圆；
# import numpy as np
# import cv2
# img=np.zeros((512,512,3), np.uint8)
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# cv2.circle(img,(447,63), 63, (0,0,255), -1)
# cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# cv2.imshow("ellipse",img)
# cv2.imwrite("E:\\Deep Learning\\DeepLearning\\Opencv\\ellipse.jpg",img)
# cv2.waitKey(0)

# 005_5 画多边形；
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# 创建一个黑色图片 np.zeros()返回一个填充为0的数组
# img = np.zeros((512,512,3),np.uint8)
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts=pts.reshape((-1,1,2))  # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# cv2.circle(img,(447,63), 63, (0,0,255), -1)
# cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# cv2.polylines(img,[pts],True,(255,0,255))  # 如果第三个参数是 False，我们得到的多边形是不闭合的（首尾不相连）。
# cv2.imshow("polylines",img)
# cv2.imwrite("E:\\Deep Learning\\DeepLearning\\Opencv\\polygon.jpg",img)
# cv2.waitKey(0)


# 005_6 在图片上添加文字;
# 在图像上绘制白色的 OpenCV；
# import numpy as np
# import cv2
# img=np.zeros((512,512,3), np.uint8)
# font=cv2.FONT_HERSHEY_SIMPLEX
#
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts=pts.reshape((-1,1,2))  # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# cv2.circle(img,(447,63), 63, (0,0,255), -1)
# cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# cv2.polylines(img,[pts],True,(255,0,255))  # 如果第三个参数是 False，我们得到的多边形是不闭合的（首尾不相连）。
#
# cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)
#
# winname = 'example'
# cv2.namedWindow(winname)
# cv2.imshow(winname, img)
# cv2.waitKey(0)
# cv2.destroyWindow(winname)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\OpenCV.jpg',img)

# 005_7 把鼠标当画笔;
# import cv2
# events=[i for i in dir(cv2) if 'EVENT'in i]
# print (events)
# 打印['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY',
# 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY',
# 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK',
# 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

# # 005_8 鼠标事件回调函数只用做一件事：在双击过的地方绘制一个圆圈。
# import numpy as np
# import cv2
# img=np.zeros((512,512,3), np.uint8)
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
# # 创建图像与窗口并将窗口与回调函数绑定
# img = np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
# while(1):
#     cv2.imshow('image', img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

# 005_9 高级一点的示例;

# import numpy as np
# import cv2
# # 当鼠标按下时变为 True
# drawing=False
# # 如果 mode 为 true 绘制矩形。按下'm' 变成绘制曲线。
# mode=True
# ix,iy=-1,-1
# # 创建回调函数
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
# # 当按下左键是返回起始位置坐标
#     if event==cv2.EVENT_LBUTTONDOWN:
#         drawing=True
#         ix,iy=x,y
# # 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
#     elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
#         if drawing==True:
#             if mode==True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:# 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细
#                 cv2.circle(img,(x,y),3,(0,0,255),-1) # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
#                 # r=int(np.sqrt((x-ix)**2+(y-iy)**2))
#                 # cv2.circle(img,(x,y),r,(0,0,255),-1)
#                 # 当鼠标松开停止绘画。
#     elif event==cv2.EVENT_LBUTTONUP:
#         drawing==False
#         # if mode==True:
#         # cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         # else:
#         # cv2.circle(img,(x,y),5,(0,0,255),-1)
#         img = np.zeros((512, 512, 3), np.uint8)
#         cv2.namedWindow('image')
#         cv2.setMouseCallback('image', draw_circle)
#         while (1):
#             cv2.imshow('image', img)
#             k = cv2.waitKey(1) & 0xFF
#             if k == ord('m'):
#                 mode = not mode
#             elif k == 27:
#                 break


# 005_10 用滑动条做调色板;

# import cv2
# import numpy as np
# def nothing(x):
#     pass
# # 创建一副黑色图像
# img=np.zeros((300,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('G','image',0,255,nothing)
# cv2.createTrackbar('B','image',0,255,nothing)
# switch='0:OFF\n1:ON'
# cv2.createTrackbar(switch,'image',0,1,nothing)
# while(1):
#     cv2.imshow('image',img)
#     k=cv2.waitKey(1)&0xFF
#     if k==27:
#         break
#     r=cv2.getTrackbarPos('R','image')
#     g=cv2.getTrackbarPos('G','image')
#     b=cv2.getTrackbarPos('B','image')
#     s=cv2.getTrackbarPos(switch,'image')
#     if s==0:
#         img[:]=0
#     else:
#         img[:]=[b,g,r]
# cv2.destroyAllWindows()


# 005_11 获取并修改像素值;
# import numpy as np
# import cv2
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\OpenCV.jpg')
#
# px=img[100,100]
# print (px) # [0 0 0];
# blue=img[100,100,0]
# print (blue) # 0;

# 005_12 获取图像属性;

# import numpy as np
# import cv2
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\OpenCV.jpg')
# print(img.shape)  # (512, 512, 3)--img.shape 可以获取图像的形状;
# print(img.size)   # 786432--img.size 可以返回图像的像素数目(H*W*C);
# print(img.dtype)  # uint8--img.dtype 返回的是图像的数据类型;


# 005_13 图像 ROI;

# import numpy as np
# import cv2
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\OpenCV.jpg')
# ball=img[280:340,330:390]
# img[273:333,100:160]=ball

# 005_14 拆分及合并图像通道;

# import numpy as np
# import cv2
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\OpenCV.jpg')
#
# b,g,r=cv2.split(img)
# img=cv2.merge(b,g)
#
# b=img[:,:,0]
# img[:,:,2]=0
# print(img)

# 005_15 为图像扩边（填充）;
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# BLUE=[255,0,0]
# img1=cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan.jpg')
#
# replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
# constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan_splicing.jpg')
# plt.show()

# 如何保存多幅拼接图像；

# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan.png',all)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan_img1.png',img1)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan_replicate.png',replicate)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan_reflect.png',reflect)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan_reflect101.png',reflect101)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan_wrap.png',wrap)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan_constant.png',constant)


# 10.1 图像上的算术运算;
# 函数 cv2.add() 将两幅图像进行加法运算;
# import numpy as np
# import cv2
# x = np.uint8([250])
# y = np.uint8([10])
# print(cv2.add(x,y))  # 250+10 = 260 => 255;
# print(x+y)           # 250+10 = 260 % 256 = 4;

# 10.2 图像混合--(图像大小需一致)(g (x) = (1 − α) f0 (x) + αf1 (x));
# import cv2
# import numpy as np

# img1=cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan.jpg')
# img2=cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\VR002.png')
# dst= cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
# cv2.imshow('dst',dst)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\yy.jpg',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 10.3 按位运算;
# import cv2
# import numpy as np
# 加载图像
# img1=cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\yanyan.jpg')
# img2=cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\me.jpg') # I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols ]
# # Now create a mask of logo and create its inverse mask also
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# # Now black-out the area of logo in ROI
# # 取 roi 中与 mask 中不为零的值对应的像素的值，其他值为 0 # 注意这里必须有 mask=mask 或者 mask=mask_inv, 其中的 mask= 不能忽略
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
# # 取 roi 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0。
# # Take only region of logo from logo image.
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
# # Put logo in ROI and modify the main image
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
# cv2.imshow('res',img1)
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\logo.png',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 11.1 使用 OpenCV 检测程序效率;
# cv2.getTickCount 函数返回从参考点到这个函数被执行的时钟数;cv2.getTickFrequency 返回时钟频率;

# import cv2
# import numpy as np
# e1 = cv2.getTickCount()
# # your code execution
# e2 = cv2.getTickCount()
# time = (e2 - e1)/ cv2.getTickFrequency()


# import cv2
# import numpy as np
# img1 = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg')
# e1 = cv2.getTickCount()
# for i in range(5,49,2):
#     img1 = cv2.medianBlur(img1,i)
# e2 = cv2.getTickCount()
# t = (e2 - e1)/cv2.getTickFrequency()
# print (t)

# 11.2 OpenCV 中的默认优化

# import numpy as np
# import cv2
# cv2.useOptimized()  # return True or False;

# 11.3 在 IPython 中检测程序效率
# 11.4 更多 IPython 的魔法命令(profiling， line profiling，内存使用等)
#
# 1. Python Optimization Techniques
# 2. Scipy Lecture Notes - Advanced Numpy
# 3. Timing and Profiling in IPython

# OpenCV 中的图像处理
# 13.1 转换颜色空间
# cv2.cvtColor(input_image， flag),flag为转换类型；
# 对于 BGR<->Gray 的转换，我们要使用的 flag 就是 cv2.COLOR_BGR2GRAY；
# cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

# 对于 BGR<->HSV 的转换，我们用的 flag 就是 cv2.COLOR_BGR2HSV；
# CV2.cvtColor(input_image,cv2.COLOR_BRG2HSV);

# 注意： 在 OpenCV 的 HSV 格式中， H（色彩/色度）的取值范围是 [0， 179]，S（饱和度）的取值范围 [0， 255]， V（亮度）的取值范围 [0， 255]。但是不
# 同的软件使用的值可能不同。所以当你需要拿 OpenCV 的 HSV 值与其他软件的 HSV 值进行对比时，一定要记得归一化。

# -*- coding: utf-8 -*-
# 13.2 物体跟踪(提取带有某个特定颜色的物体，比如：在蓝色物体周围画一个圈。)
# import cv2
# import numpy as np
# cap=cv2.VideoCapture(0)
# while(1):
#     # 获取每一帧
#     ret, frame = cap.read()
#     # 转换到 HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # 设定蓝色的阈值
#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#     # 根据阈值构建掩模
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     # 对原图像和掩模进行位运算
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#     # 显示图像
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# # 关闭窗口；
# cv2.destroyAllWindows()


# 13.3 怎样找到要跟踪对象的 HSV 值
# -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# green=np.uint8([0,255,0])
# hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# error: /builddir/build/BUILD/opencv-2.4.6.1/
# modules/imgproc/src/color.cpp:3541:
# error: (-215) (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F)
# in function cvtColor
# scn (the number of channels of the source),
# i.e. self.img.channels(), is neither 3 nor 4.
#
# depth (of the source),
# i.e. self.img.depth(), is neither CV_8U nor CV_32F.
# 所以不能用 [0,255,0]，而要用 [[[0,255,0]]]
# 这里的三层括号应该分别对应于 cvArray， cvMat， IplImage
# green=np.uint8([[[0,255,0]]])
# hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# print (hsv_green)
# [[[60,255,255]]]

# 14 几何变换 (cv2.warpAffine 接收的参数是2 × 3 的变换矩阵，而 cv2.warpPerspective 接收的参数是 3 × 3 的变换矩阵。)
# 14.1 扩展缩放
# 扩展缩放只是改变图像的尺寸大小。 OpenCV 提供的函数 cv2.resize();
# 在缩放时我们推荐使用 cv2.INTER_AREA，
# 在扩展时我们推荐使用 v2.INTER_CUBIC（慢) 和 v2.INTER_LINEAR。
# 默认情况下所有改变图像尺寸大小的操作使用的插值方法都是 cv2.INTER_LINEAR
# import numpy as np
# import cv2
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg')
# # 下面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子,因此这里为 None;
# res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# # OR,这里呢，我们直接设置输出图像的尺寸，所以不用设置缩放因子;
# height,width=img.shape[:2]
# res=cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
# while(1):
#     cv2.imshow('res', res)
#     cv2.imshow('img', img)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

# 14.2 平移
# 下面这个例子，它被移动了（ 100,50）个像素。
# -*- coding: utf-8 -*-
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture(0)
# while (1):
#     # 获取每一帧
#     ret, frame = cap.read()
#     # 转换到 HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # 设定蓝色的阈值
#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#     # 根据阈值构建掩模
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     # 对原图像和掩模进行位运算
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#     # 显示图像
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# # 关闭窗口
# cv2.destroyAllWindows()

# 14.3 旋转
# 为了构建这个旋转矩阵， OpenCV 提供了一个函数： cv2.getRotationMatrix2D。
# -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg',0)
# rows,cols=img.shape
# # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
# # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
# M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)
# # 第三个参数是输出图像的尺寸中心
# dst=cv2.warpAffine(img,M,(2*cols,2*rows))
# cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv_Rotation.jpg',dst)
# while(1):
#     cv2.imshow('img',dst)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()


# 14.4 仿射变换
# 为了创建这个矩阵我们需要从原图像中找到三个点以及他们在输出图像中的位置。然后cv2.getAffineTransform 会创建一个 2x3 的矩阵，最后这个矩阵会被传给函数 cv2.warpAffine;

# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg')
# rows, cols = img.shape[:2]
#
# # 变换前的三个点
# pts1 = np.float32([[50, 65], [150, 65], [210, 210]])
# # 变换后的三个点
# pts2 = np.float32([[50, 100], [150, 65], [100, 250]])
#
# # 生成变换矩阵
# M = cv.getAffineTransform(pts1, pts2)
# # 第三个参数为dst的大小
# dst = cv.warpAffine(img, M, (cols, rows))
#
# plt.subplot(121), plt.imshow(img), plt.title('input')
# plt.subplot(122), plt.imshow(dst), plt.title('output')
# # plt.show() # 注释前，保存白底图像，无内容；
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_warpAffine.jpg')


# 14.5 透视变换
# 对于视角变换，我们需要一个 3x3 变换矩阵。在变换前后直线还是直线。
# 要构建这个变换矩阵，你需要在输入图像上找 4 个点，以及他们在输出图
# 像上对应的位置。这四个点中的任意三个都不能共线。这个变换矩阵可以由
# 函数 cv2.getPerspectiveTransform() 构建。然后把这个矩阵传给函数cv2.warpPerspective;

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img = cv.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg') # 'D:\\DL information\\Opencv\\VR.jpg'
#
# # 原图中卡片的四个角点
# pts1 = np.float32([[148, 80], [437, 114], [94, 247], [423, 288]])
# # 变换后分别在左上、右上、左下、右下四个点
# pts2 = np.float32([[0, 0], [320, 0], [0, 178], [320, 178]])
#
# # 生成透视变换矩阵
# M = cv.getPerspectiveTransform(pts1, pts2)
# # 进行透视变换，参数3是目标图像大小
# dst = cv.warpPerspective(img, M, (320, 178))
#
# plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
# plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
# # # plt.show() # 注释前，保存白底图像，无内容；
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_warpPerspective.jpg') # 'D:\\DL information\\Opencv\\VR_warpPerspective.jpg'


# 15 图像阈值
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img=cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg',0)
# ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_warpPerspective.jpg')

# 15.2 自适应阈值
# 当同一幅图像上的不同部分的具有不同亮度时。
# 这种情况下我们需要采用自适应阈值。此时的阈值是根据图像上的
# 每一个小区域计算与其对应的阈值。因此在同一幅图像上的不同区域采用的是
# 不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果。
# Adaptive Method- 指定计算阈值的方法。
# Block Size - 邻域大小（用来计算阈值的区域大小）。
# C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常数。

# -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:\\DL information\\Opencv\\VR.jpg',0)
# # 中值滤波
# img = cv2.medianBlur(img,5)
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# #11 为 Block size, 2 为 C 值
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# titles = ['Original Image', 'Global Thresholding (v = 127)','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('D:\\DL information\\Opencv\\VR_AdaptiveThreshold.jpg')

# 15.3 Otsu’s 二值化
# 这里用到到的函数还是 cv2.threshold()，但是需要多传入一个参数
# （ flag）： cv2.THRESH_OTSU。这时要把阈值设为 0。然后算法会找到最
# 优阈值，这个最优阈值就是返回值 retVal。如果不使用 Otsu 二值化，返回的
# retVal 值与设定的阈值相等。

# -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg',0)
# # global thresholding
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# #（ 5,5）为高斯核的大小， 0 为标准差
# blur = cv2.GaussianBlur(img,(5,5),0)
# # 阈值一定要设为 0！
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # plot all the images and their histograms
# images = [img, 0, th1,
# img, 0, th2,
# blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)','Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# # 这里使用了 pyplot 中画直方图的方法， plt.hist, 要注意的是它的参数是一维数组
# # 所以这里使用了（ numpy） ravel 方法，将多维数组转换成一维，也可以使用 flatten 方法
# # ndarray.flat 1-D iterator over an array.
# # ndarray.flatten 1-D array copy of the elements of an array in row-major order.
# for i in range(3):
#     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
#     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
#     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
#     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_ Otsu-threshold.jpg')

# 15.4 Otsu’s 二值化是如何工作的？
# import cv2
# import numpy as np
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg',0)
# blur = cv2.GaussianBlur(img,(5,5),0)
# # find normalized_histogram, and its cumulative distribution function
# # 计算归一化直方图
# #CalcHist(image, accumulate=0, mask=NULL)
# hist = cv2.calcHist([blur],[0],None,[256],[0,256])
# hist_norm = hist.ravel()/hist.max()
# Q = hist_norm.cumsum()
# bins = np.arange(256)
# fn_min = np.inf
# thresh = -1
# for i in range(1,256):
#     p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
#     q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
#     b1, b2 = np.hsplit(bins, [i])  # weights
#     # finding means and variances
#     m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
#     v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
#     # calculates the minimization function
#     fn = v1 * q1 + v2 * q2
#     if fn < fn_min:
#         fn_min = fn
#         thresh = i
# # find otsu's threshold value with OpenCV function
# ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(thresh,ret)

# 16 图像平滑
# OpenCV 提供的函数 cv.filter2D() 可以让我们对一幅图像进行卷积操作。
# 2D 卷积
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg')
# kernel = np.ones((5,5),np.float32)/25
# #cv.Filter2D(src, dst, kernel, anchor=(-1, -1))
# #ddepth –desired depth of the destination image;
# #if it is negative, it will be the same as src.depth();
# #the following combinations of src.depth() and ddepth are supported:
# #src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
# #src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
# #src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
# #src.depth() = CV_64F, ddepth = -1/CV_64F
# #when ddepth=-1, the output image will have the same depth as the source.
# dst = cv2.filter2D(img,-1,kernel)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_filter2D.jpg')

# 图像模糊（图像平滑）使用低通滤波器可以达到图像模糊的目的。
# 16.1 平均
# -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg')
# blur = cv2.blur(img,(5,5))
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_blur.jpg')

# 16.2 高斯模糊
# 实现的函数是 cv2.GaussianBlur()。
# 高斯滤波可以有效的从图像中去除高斯噪音。
# 0 是指根据窗口大小（ 5,5）来计算高斯函数标准差
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg')
# blur = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow("GaussianBlur",blur)
# cv2.waitKey()
# cv2.imwrite("E:\\Deep Learning\\DeepLearning\\Opencv\\VR_GaussianBlur.jpg",blur) # work;
# plt.show()
# plt.savefig('D:\\DL information\\Opencv\\VR_GaussianBlur.jpg')

# 16.3 中值模糊
# 是用与卷积框对应像素的中值来替代中心像素的值；
# 中值滤波器经常用来去除椒盐噪声；

# import cv2
# from matplotlib import pyplot as plt
# # img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\Opencv.jpg')
# img =cv2.imread('C:\\Users\\eivision\\Desktop\\noise.jpeg')
# median = cv2.medianBlur(img,5)
# # cv2.imshow("medianBlur",median)
# # cv2.waitKey()
# # cv2.imwrite("D:\\DL information\\Opencv\\VR_medianBlur.jpg",median) # work;
# plt.subplot(),plt.imshow(median),plt.title('median-image')
# plt.xticks([]), plt.yticks([])
# # plt.show()
# # plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_median.jpg')
# plt.savefig('D:\\DL information\\Opencv\\noise_median.jpeg')

# 16.4 双边滤波;
# 函数 cv2.bilateralFilter() 能在保持边界清晰的情况下有效的去除噪音。
# 双边滤波会确保边界不会被模糊掉，因为边界处的灰度值变化比较大;

# cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
# d – Diameter of each pixel neighborhood that is used during filtering.
# If it is non-positive, it is computed from sigmaSpace
# 9 邻域直径，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差

# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\VR.jpg')
# blur = cv2.bilateralFilter(img,9,75,75)
# plt.subplot(),plt.imshow(blur),plt.title('bilateralFilter-image')
# plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR_bilateralFilter.jpg')

# 17 形态学转换
# 学习不同的形态学操作，例如腐蚀，膨胀，开运算，闭运算等;
# 我们要学习的函数有： cv2.erode()， cv2.dilate()， cv2.morphologyEx()等;

# 17.1 腐蚀 (去除白噪声很有用)
# # -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# img = cv2.imread('D:\\DL information\\Opencv\\white-noise.jpg',0)
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1) # iterations = 1腐蚀次数；
# cv2.imshow("erosion",erosion)
# cv2.waitKey()
# cv2.imwrite("E:\\Deep Learning\\DeepLearning\\Opencv\\VR_erosion.jpg",erosion) # work;

# 17.2 膨胀
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# kernel = np.ones((3,3),np.uint8)
# dilation = cv2.dilate(img,kernel,iterations = 1)
# plt.subplot(),plt.imshow(dilation),plt.title('dilation-image')
# plt.xticks([]), plt.yticks([])
# plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise_dilation.jpg')

# 17.3 开运算 (先进行腐蚀再进行膨胀就叫做开运算)；
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# plt.subplot(),plt.imshow(opening),plt.title('opening-image')
# plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise_opening.jpg')

# # 17.4 闭运算(先膨胀再腐蚀; 经常被用来填充前景物体中的小洞，或者前景物体上的小黑点);
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# kernel = np.ones((3,3),np.uint8)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# plt.subplot(),plt.imshow(closing),plt.title('closing-image')
# plt.xticks([]), plt.yticks([])
# plt.show() # show和savefig不可同时使用，否则仅能保存为白色图像；
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise_closing.jpg')

# 17.5 形态学梯度(其实就是一幅图像膨胀与腐蚀的差别,结果看上去就像前景物体的轮廓。)
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:\\DL information\\Opencv\\white-noise.jpg',0)
# kernel = np.ones((3,3),np.uint8)
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# plt.subplot(),plt.imshow(gradient),plt.title('gradient-image')
# plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('D:\\DL information\\Opencv\\white-noise_gradient.jpg')

# 17.6 礼帽(原始图像与进行开运算之后得到的图像的差);
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# kernel = np.ones((3,3),np.uint8)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
#
# plt.subplot(),plt.imshow(tophat),plt.title('tophat-image')
# plt.xticks([]), plt.yticks([])
# # plt.show()
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise_tophat.jpg')


# # 17.7 黑帽(进行闭运算之后得到的图像与原始图像的差);
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# kernel = np.ones((3,3),np.uint8)
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# plt.subplot(),plt.imshow(blackhat),plt.title('blackhat-image')
# plt.xticks([]), plt.yticks([])
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise_blackhat.jpg')
# plt.show()

# 18 图像梯度
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
#
# # cv2.CV_64F 输出图像的深度（数据类型），可以使用-1, 与原图像保持一致 np.uint8
# laplacian=cv2.Laplacian(img,cv2.CV_64F)
# # 参数 1,0 为只在 x 方向求一阶导数，最大可以求 2 阶导数。
# sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# # 参数 0,1 为只在 y 方向求一阶导数，最大可以求 2 阶导数。
# sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise-Grad(laplacian+sobel).jpg')
# plt.show()

# # 输出图片的深度不同造成的不同效果
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# # Output dtype = cv2.CV_8U
# sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
# # 也可以将参数设为-1
# # sobelx8u = cv2.Sobel(img,-1,1,0,ksize=5)
# # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
# sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
# plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise-depth.jpg')
# plt.show()


# 19 Canny 边缘检测
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\VR.jpg',0)
# edges = cv2.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR-Canny.jpg')
# plt.show()

# 20 图像金字塔（高斯金字塔用来向下降采样图像；拉普拉斯金字塔: 用来从金字塔低层图像重建上层未采样图像）
# cv2.pyrDown() 和 cv2.pyrUp() 构建图像金字塔;
# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# lower_reso = cv2.pyrDown(img)
# higher_reso = cv2.pyrUp(lower_reso)
#
# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(lower_reso,cmap = 'gray')
# plt.title('down'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(higher_reso,cmap = 'gray')
# plt.title('up'), plt.xticks([]), plt.yticks([])
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise-pyramid.jpg')
# plt.show()


# 使用金字塔（高斯、拉普拉斯）进行图像融合；
# import cv2
# import numpy as np,sys
# A = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
# B = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\VR.jpg',0)
# # generate Gaussian pyramid for A
# G = A.copy()
# gpA = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)
#     gpA.append(G)
# # generate Gaussian pyramid for B
# G = B.copy()
# gpB = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)
#     gpB.append(G)
# # generate Laplacian Pyramid for A
# lpA = [gpA[5]]
# for i in range(5,0,-1):
#     GE = cv2.pyrUp(gpA[i])
#     L = cv2.subtract(gpA[i-1],GE)
# lpA.append(L)
# # generate Laplacian Pyramid for B
# lpB = [gpB[5]]
# for i in range(5,0,-1):
#     GE = cv2.pyrUp(gpB[i])
#     L = cv2.subtract(gpB[i-1],GE)
# lpB.append(L)
# # Now add left and right halves of images in each level
# #numpy.hstack(tup)
# #Take a sequence of arrays and stack them horizontally
# #to make a single array.
# LS = []
# for la,lb in zip(lpA,lpB):
#     rows,cols,dpt = la.shape
#     ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
#     LS.append(ls)
# # now reconstruct
# ls_ = LS[0]
# for i in range(1,6):
#     ls_ = cv2.pyrUp(ls_)
#     ls_ = cv2.add(ls_, LS[i])
# # image with direct connecting each half
# real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
# cv2.imwrite('Pyramid_blending2.jpg',ls_)
# cv2.imwrite('Direct_blending.jpg',real)
#
# cv2.imshow("white-noise",A)
# cv2.imshow("VR",B)
# # cv2.imshow("apple_orange_reconstruct",apple_orange_reconstruct)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# # laod images
#
# # laod images
#
# apple = cv.imread(r"E:\\Deep Learning\\DeepLearning\\Opencv\\apple.jpg")
# orange = cv.imread(r"E:\\Deep Learning\\DeepLearning\\Opencv\\orange.jpg")
#
# print(apple.shape) # (695, 756, 3);
# print(orange.shape) # (695, 756, 3);
#
# # print(apple.dtype, orange.dtype # uint8 uint8
#
# apple_copy = apple.copy()
#
# # Guassian Pyramids for apple
# apple_guassian = [apple_copy]
#
# for i in range(6):
#     apple_copy = cv.pyrDown(apple_copy)
#     apple_guassian.append(apple_copy)
#
# # Guassian Pyramids for apple
# orange_copy = orange.copy()
# orange_gaussian = [orange_copy]
#
# for i in range(6):
#     orange_copy = cv.pyrDown(orange_copy)
#     orange_gaussian.append(orange_copy)
#
# # Laplacian Pyramids for apple
# apple_copy = apple_guassian[5]
# apple_laplacian = [apple_copy]
#
# for i in range(5, 0, -1):
#     gaussian_extended = cv.pyrUp(apple_guassian[i])
#
#     laplacian = cv.subtract(apple_guassian[i - 1], gaussian_extended) # cv2.error: OpenCV(4.5.1);
#     apple_laplacian.append(laplacian)
#
# # Laplacian Pyramids for orange
# orange_copy = orange_gaussian[5]
# orange_laplacian = [orange_copy]
#
# for i in range(5, 0, -1):
#     gaussian_extended = cv.pyrUp(orange_gaussian[i])
#
#     laplacian = cv.subtract(orange_gaussian[i - 1], gaussian_extended)
#     orange_laplacian.append(laplacian)
#
# # join the left half of apple and right half of orange in each levels of Laplacian Pyramids
# apple_orange_pyramid = []
# n = 0
# for apple_lp, orange_lp in zip(apple_laplacian, orange_laplacian):
#     n += 1
#     cols, rows, ch = apple_lp.shape
#     laplacian = np.hstack((apple_lp[:, 0:int(cols / 2)], orange_lp[:, int(cols / 2):]))
#     apple_orange_pyramid.append(laplacian)
#
# # reconstrut image
# apple_orange_reconstruct = apple_orange_pyramid[0]
# for i in range(1, 6):
#     apple_orange_reconstruct = cv.pyrUp(apple_orange_reconstruct)
#     apple_orange_reconstruct = cv.add(apple_orange_pyramid[i], apple_orange_reconstruct)
#
# cv.imshow("apple", apple)
# cv.imshow("orange", orange)
# cv.imshow("apple_orange_reconstruct", apple_orange_reconstruct)
# cv.waitKey(0)
# cv.destroyAllWindows()


############################################################
# work(2022-0127)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def sameSize(img1, img2):
#     rows, cols, dpt = img2.shape
#     dst = img1[:rows, :cols]
#     return dst
#
#
# apple = cv2.imread(r"E:\\Deep Learning\\DeepLearning\\Opencv\\apple.jpg")
# orange = cv2.imread(r"E:\\Deep Learning\\DeepLearning\\Opencv\\orange.jpg")
#
# G = apple.copy()
# gp_apple = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)  # 下采样共6次
#     gp_apple.append(G)
#
# G = orange.copy()
# gp_orange = [G]
# for j in range(6):
#     G = cv2.pyrDown(G)
#     gp_orange.append(G)
#
# lp_apple = [gp_apple[5]]
# for i in range(5, 0, -1):
#     GE = cv2.pyrUp(gp_apple[i])  # 上采样6次
#     L = cv2.subtract(gp_apple[i - 1], sameSize(GE, gp_apple[i - 1]))  # 两个图像相减
#     lp_apple.append(L)
#
# lp_orange = [gp_orange[5]]
# for i in range(5, 0, -1):
#     GE = cv2.pyrUp(gp_orange[i])
#     L = cv2.subtract(gp_orange[i - 1], sameSize(GE, gp_orange[i - 1]))
#     lp_orange.append(L)
#
# LS = []
# for la, lb in zip(lp_apple, lp_orange):  # 一个数组中取一个元素
#     rows, cols, dpt = la.shape
#     ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))  # 水平方向上平铺，各取一半
#     LS.append(ls)
#
# ls_reconstruct = LS[0]
# for i in range(1, 6):
#     ls_reconstruct = cv2.pyrUp(ls_reconstruct)
#     ls_reconstruct = cv2.add(sameSize(ls_reconstruct, LS[i]), LS[i])
#
# r, c, depth = apple.shape
# real = np.hstack((apple[:, 0:c // 2], orange[:, c // 2:]))  # apple和orange各取一半。
#
# plt.subplot(2,2,1), plt.imshow(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB))
# plt.title("apple"), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2), plt.imshow(cv2.cvtColor(orange, cv2.COLOR_BGR2RGB))
# plt.title("orange"), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3), plt.imshow(cv2.cvtColor(real, cv2.COLOR_BGR2RGB))
# plt.title("real"), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4), plt.imshow(cv2.cvtColor(ls_reconstruct, cv2.COLOR_BGR2RGB))
# plt.title("laplace_pyramid"), plt.xticks([]), plt.yticks([])
# plt.savefig("E:\\Deep Learning\\DeepLearning\\Opencv\\fruit-pyramid.jpg")
# plt.show()
############################################################
# work(2022-0127)


# 21 OpenCV 中的轮廓
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR-灰度
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值图像
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img1 = img.copy()  # 对原始图像进行绘制
# contour = cv2.drawContours(img1, contours, -1, (0, 127, 127), 4)  # img1为复制图像，轮廓会修改原始图像
# # cv2.imshow("original", img)
# # cv2.imshow("contours", contour)
# # cv2.imwrite('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise-contour.jpg',contour)
# # cv2.waitKey()
#
# plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(binary, cmap='gray')
# plt.title('binary Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(2, 2, 3), plt.imshow(hierarchy, cmap='gray')
# plt.title('hierarchy Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(contour, cmap='gray')
# plt.title('contour Image'), plt.xticks([]), plt.yticks([])
#
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise-contour2.jpg')
# plt.show()

# 21.2 轮廓特征
# 21.2.1 矩
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\white-noise.jpg',0)
#
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv2.moments(cnt)
# print(M)


# 21.2.2 轮廓面积
# -*- coding: cp936 -*-
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\VR.jpg',0)
#
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv2.moments(cnt)
# area = cv2.contourArea(cnt)
#
# print(area) # 4.0;
# print(M['m00']) # 4.0;2种方法结果一致。


# 21.2.3 轮廓周长
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\VR.jpg',0)
#
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# perimeter = cv2.arcLength(cnt,True)
#
# print(perimeter) # 7.656854152679443;

# 21.2.4 轮廓近似
# -*- coding: utf-8 -*-
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
# image = cv2.imread('E:\\Deep Learning\\DeepLearning\\Opencv\\VR.jpg')
# # cv2.imshow("Image", image)
# # cv2.waitKey()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 200, 255,cv2.THRESH_BINARY_INV)[1]
#
# # cv2.imshow("Thresh", thresh)
# # cv2.waitKey()
#
# plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(1, 3, 2), plt.imshow(gray, cmap='gray')
# plt.title('gray Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(1, 3, 3), plt.imshow(thresh, cmap='gray')
# plt.title('thresh Image'), plt.xticks([]), plt.yticks([])
# plt.savefig('E:\\Deep Learning\\DeepLearning\\Opencv\\VR-contour-approximation.jpg')
# plt.show()

# 21.2.5 凸包
# hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]])
# hull = cv2.convexHull(cnt)

# 21.2.6 凸性检测
# import cv2
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# k = cv2.isContourConvex(cnt)
# print(k)  # return False

# 21.2.7 边界矩形

# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
#
# # 直边界矩形
# x,y,w,h = cv2.boundingRect(cnt)
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#
# # 旋转的边界矩形
# x,y,w,h = cv2.boundingRect(cnt)
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# plt.show()

# 21.2.8 最小外接圆
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
#
# (x,y),radius = cv2.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# img = cv2.circle(img,center,radius,(0,255,0),2)
# # plt.savefig('D:/software/DL information/Opencv/Opencv_wyuan.png')
# plt.show()

# 21.2.9 椭圆拟合
# 函数为 cv2.ellipse()
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]

# ellipse = cv2.fitEllipse(cnt)
# im = cv2.ellipse(img,ellipse,(0,255,0),2)
# print(im)

# 21.3 轮廓的性质
# 21.3.1 长宽比
# x,y,w,h = cv2.boundingRect(cnt)
# aspect_ratio = float(w)/h
# print(aspect_ratio)

# Extent (轮廓面积与边界矩形面积的比)
# area = cv2.contourArea(cnt)
# x,y,w,h = cv2.boundingRect(cnt)
# rect_area = w*h
# extent = float(area)/rect_area
# print(extent)

# 21.3.9 极点
# leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
# rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
# topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
# bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
# plt.savefig('D:/software/DL information/Opencv/Opencv_jdian.png')
# plt.show()

# 22 直方图
# 22.1 直方图的计算，绘制与分析

# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png',0)
# # 别忘了中括号 [img],[0],None,[256],[0,256]，只有 mask 没有中括号
# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# print(hist)
# plt.show()


# 22.1.2 绘制直方图
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# plt.hist(img.ravel(),256,[0,256]);
# plt.savefig('D:/software/DL information/Opencv/VR_hist.jpg')
# plt.show()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg') # 加“0”，则报错；
# color = ('b','g','r')
# # 对一个列表或数组既要遍历索引又要遍历元素时
# # 使用内置 enumerrate 函数会有更加直接，优美的做法
# # enumerate 会将数组或列表组成一个索引序列。
# # 使我们再获取索引和索引内容的时候更加方便
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.savefig('D:/software/DL information/Opencv/VR_hist_1.jpg')
# plt.show()

# 22.1.3 使用掩模

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
#
# # create a mask
# mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300, 100:400] = 255
# masked_img = cv2.bitwise_and(img,img,mask = mask)
# # Calculate histogram with mask and without mask
# # Check third argument for mask
# hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
# hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask,'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0,256])
# plt.savefig('D:/software/DL information/Opencv/VR_mask.jpg')
# plt.show()


# 22.2 直方图均衡化 (p140)
# numpy直方图均衡化；

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# # flatten() 将数组变成一维
# hist,bins = np.histogram(img.flatten(),256,[0,256])
# # 计算累积分布图
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'best')


# 构建 Numpy 掩模数组， cdf 为原数组，当数组元素为 0 时，掩盖（计算时被忽略）。
# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# # 对被掩盖的元素赋值，这里赋值为 0
# cdf = np.ma.filled(cdf_m,0).astype('uint8')
# img2 = cdf[img]
#
# cdf_normalized = cdf * hist.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img2.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf_avg','histogram_avg'), loc = 'best')
#
# plt.savefig('D:/software/DL information/Opencv/VR_avg_1.jpg')
# plt.show()


# 22.2.1 OpenCV 中的直方图均衡化 (142)
# import cv2
# import numpy as np
# img = cv2.imread('D:/software/DL information/Opencv/VR002_1.png',0)
# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ))
# #stacking images side-by-side
# cv2.imwrite('D:/software/DL information/Opencv/VR002_1_avg(cv).png',res)
# cv2.imshow("Image",res)
# cv2.waitKey(0)

# 22.2.2 CLAHE 有限对比适应性直方图均衡化 (144)

# import numpy as np
# import cv2
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# # create a CLAHE object (Arguments are optional).
# # 不知道为什么我没好到 createCLAHE 这个模块
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv2.imwrite('D:/software/DL information/Opencv/VR_avg_(CLAHE).jpg',cl1)
# cv2.imshow("Image",cl1)
# cv2.waitKey(0)     # 延迟显示；

# 22.3 2D 直方图（145）;
# 函数 cv2.calcHist() 来计算直方图;
# 计算一维直方图，要从 BGR 转换到 HSV;

# import cv2
# import numpy as np
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png')
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# cv2.imshow("Image",hist)
# cv2.waitKey(0)

# 22.3.3 Numpy 中 2D 直方图(146);

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/Opencv.png')
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])

# 22.3.4 绘制 2D 直方图
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg')
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
# plt.imshow(hist,interpolation = 'nearest')
# plt.title("2D-HIST")
# plt.savefig('D:/software/DL information/Opencv/VR_2D（hist）.jpg')
# plt.show()

# 练习（149）；
# import numpy as np
# import cv2
# # from time import clock
# from time import perf_counter  #  process_time
# import sys
# import video
# # video 模块也是 opencv 官方文档中自带的
# if __name__ == '__main__':
#     # 构建 HSV 颜色地图
#     hsv_map = np.zeros((180, 256, 3), np.uint8)
#     # np.indices 可以返回由数组索引构建的新数组。
#     # 例如： np.indices（（ 3,2））；其中（ 3,2）为原来数组的维度：行和列。
#     # 返回值首先看输入的参数有几维：（ 3,2）有 2 维，所以从输出的结果应该是
#     # [[a],[b]], 其中包含两个 3 行， 2 列数组。
#     # 第二看每一维的大小，第一维为 3, 所以 a 中的值就 0 到 2（最大索引数），
#     # a 中的每一个值就是它的行索引；同样的方法得到 b（列索引）
#     # 结果就是
#     # array([[[0, 0],
#     # [1, 1],
#     # [2, 2]],
#     #
#     # [[0, 1],
#     # [0, 1],
#     # [0, 1]]])
#     h, s = np.indices(hsv_map.shape[:2])
#     hsv_map[:, :, 0] = h
#     hsv_map[:, :, 1] = s
#     hsv_map[:, :, 2] = 255
#     hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
#     cv2.imshow('hsv_map', hsv_map)
#     cv2.namedWindow('hist', 0)
#     hist_scale = 10
#
#     def set_scale(val):
#         global hist_scale
#         hist_scale = val
#         cv2.createTrackbar('scale', 'hist', hist_scale, 32, set_scale)
#         try:
#             fn = sys.argv[1]
#         except:
#             fn = 0
#         cam = video.create_capture(fn, fallback='synth:bg=../cpp/baboon.jpg:class=chess:noise=0.05')
#         while True:
#             flag, frame = cam.read()
#             cv2.imshow('camera', frame)
#             # 图像金字塔
#             # 通过图像金字塔降低分辨率，但不会对直方图有太大影响。
#             # 但这种低分辨率，可以很好抑制噪声，从而去除孤立的小点对直方图的影响。
#             small = cv2.pyrDown(frame)
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             # 取 v 通道 (亮度) 的值。
#             # 没有用过这种写法，还是改用最常见的用法。
#             # dark = hsv[...,2] < 32
#             # 此步操作得到的是一个布尔矩阵，小于 32 的为真，大于 32 的为假。
#             dark = hsv[:, :, 2] < 32
#             hsv[dark] = 0
#             h = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
#             # numpy.clip(a, a_min, a_max, out=None)[source]
#             # Given an interval, values outside the interval are clipped to the interval edges.
#             # For example, if an interval of [0, 1] is specified, values smaller
#             # than 0 become 0, and values larger than 1 become 1.
#             # >>> a = np.arange(10)
#             # >>> np.clip(a, 1, 8)
#             # array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
#             h = np.clip(h * 0.005 * hist_scale, 0, 1)
#             # In numpy one can use the 'newaxis' object in the slicing syntax to create an
#             # axis of length one. one can also use None instead of newaxis,
#             # the effect is exactly the same
#             # h 从一维变成 3 维
#             vis = hsv_map * h[:, :, np.newaxis] / 255.0
#             cv2.imshow('hist', vis)
#             ch = 0xFF & cv2.waitKey(1)
#             if ch == 27:
#                 break
#             cv2.destroyAllWindows()
# cv2.waitKey(0)
#


# 22.4 直方图反向投影(151)
# 22.4.1 Numpy 中的算法

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# # roi is the object or region of object we need to find
# roi = cv2.imread('D:/software/DL information/Opencv/VR10_3_img.png')
# hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# # target is the image we search in
# target = cv2.imread('D:/software/DL information/Opencv/VR.jpg')
# hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
# # Find the histograms using calcHist. Can be done with np.histogram2d also
# M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
#
# # cv2.imshow("IMG",M)
# # cv2.imshow("IMG",I)
#
# h,s,v = cv2.split(hsvt)
# B = R[h.ravel(),s.ravel()]
# B = np.minimum(B,1)
# B = B.reshape(hsvt.shape[:2])
#
# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# B=cv2.filter2D(B,-1,disc)
# B = np.uint8(B)
# cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
#
# ret,thresh = cv2.threshold(B,50,255,0)
#
# cv2.waitKey(0)


# 22.4.2 OpenCV 中的反向投影(153)

# import cv2
# import numpy as np
# roi = cv2.imread('D:/software/DL information/Opencv/VR.jpg')
# hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# target = cv2.imread('D:/software/DL information/Opencv/Opencv.png')
# hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
# # calculating object histogram
# roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# # normalize histogram and apply backprojection
# # 归一化：原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
# #cv2.NORM_MINMAX 对数组的所有值进行转化，使它们线性映射到最小值和最大值之间
# # 归一化之后的直方图便于显示，归一化之后就成了 0 到 255 之间的数了。
# cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
# dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# # Now convolute with circular disc
# # 此处卷积可以把分散的点连在一起
# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# dst=cv2.filter2D(dst,-1,disc)
# # threshold and binary AND
# ret,thresh = cv2.threshold(dst,50,255,0)
# # 别忘了是三通道图像，因此这里使用 merge 变成 3 通道
# thresh = cv2.merge((thresh,thresh,thresh))
# # 按位操作
# res = cv2.bitwise_and(target,thresh)
# res = np.hstack((target,thresh,res))
# cv2.imwrite('D:/software/DL information/Opencv/Opencv_Backproject.png',res)
# # 显示图像
# cv2.imshow('1',res)
# cv2.waitKey(0)


# 23 图像变换(156)
# 23.1 傅里叶变换
# 23.1.1 Numpy 中的傅里叶变换
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# # 这里构建振幅图的公式没学过
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
#
# # （Magnitude Spectrum）幅度谱
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#
# plt.savefig('D:/software/DL information/Opencv/VR_Magnitude Spectrum.jpg')
# plt.show()

# 频域变换
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# rows, cols = img.shape
# crow,ccol = rows//2 , cols//2
# fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# # 取绝对值
# img_back = np.abs(img_back)
# plt.subplot(131),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.savefig('D:/software/DL information/Opencv/VR_HPF_JET.jpg')
# plt.show()

# 23.1.2 OpenCV 中的傅里叶变换（158）
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.savefig('D:/software/DL information/Opencv/VR_FFT.jpg')
# # cv2.cartToPolar(img)     # 同时返回幅度和相位；
# plt.show()

# LPF（低通滤波）将高频部分去除(160);
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# rows, cols = img.shape
# crow,ccol = rows//2 , cols//2
# # create a mask first, center square is 1, remaining all zeros
# mask = np.zeros((rows,cols,2),np.uint8)
# mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# # apply mask and inverse DFT
# fshift = dft_shift*mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.savefig('D:/software/DL information/Opencv/VR_LPF.jpg')
# plt.show()

# 23.1.3 DFT 的性能优化(160)
# OpenCV 你必须自己手动补 0。但是 Numpy，你只需要指定 FFT 运算的大小，它会自动补 0;
# import cv2
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# rows,cols = img.shape
# print (rows,cols)
#
# nrows = cv2.getOptimalDFTSize(rows)
# ncols = cv2.getOptimalDFTSize(cols)
# print (nrows, ncols)

# (163)
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# # simple averaging filter without scaling parameter
# mean_filter = np.ones((3,3))
# # creating a guassian filter
# x = cv2.getGaussianKernel(5,10)
# #x.T 为矩阵转置
# gaussian = x*x.T
# # different edge detecting filters
# # scharr in x-direction
# scharr = np.array([[-3, 0, 3],
# [-10,0,10],
# [-3, 0, 3]])
# # sobel in x direction
# sobel_x= np.array([[-1, 0, 1],
# [-2, 0, 2],
# [-1, 0, 1]])
# # sobel in y direction
# sobel_y= np.array([[-1,-2,-1],
# [0, 0, 0],
# [1, 2, 1]])
# # laplacian
# laplacian=np.array([[0, 1, 0],
# [1,-4, 1],
# [0, 1, 0]])
# filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
# filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', 'sobel_y', 'scharr_x']
# fft_filters = [np.fft.fft2(x) for x in filters]
# fft_shift = [np.fft.fftshift(y) for y in fft_filters]
# mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
#     plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
# plt.savefig('D:/software/DL information/Opencv/Filter_m.jpg')
# plt.show()


# 24 模板匹配(165) 模板匹配是用来在一副大图中搜寻查找模版图像位置的方法。

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/software/DL information/Opencv/VR.jpg',0)
# img2 = img.copy()
# template = cv2.imread('D:/software/DL information/Opencv/VR_Template.png',0)
# w, h = template.shape[::-1]
# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# for meth in methods:
#     img = img2.copy()
# # exec 语句用来执行储存在字符串或文件中的 Python 语句。
# # 例如，我们可以在运行时生成一个包含 Python 代码的字符串，然后使用 exec 语句执行这些语句。
# # eval 语句用来计算存储在字符串中的有效 Python 表达式
#     method = eval(meth)
# # Apply template Matching
#     res = cv2.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# # 使用不同的比较方法，对结果的解释不同
# # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(img,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

# 24.2 多对象的模板匹配(168);
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img_rgb = cv2.imread('D:/software/DL information/Opencv/mario.png')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('D:/software/DL information/Opencv/mario_coin.png',0)
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# # umpy.where(condition[, x, y])
# # Return elements, either from x or y, depending on condition.
# # If only condition is given, return condition.nonzero().
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv2.imwrite('D:/software/DL information/Opencv/Mario_Template_Result.png',img_rgb)
# cv2.imshow("1",img_rgb)
# cv2.waitKey(0)
