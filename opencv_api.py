#coding:utf-8
import cv2
#读取文件
img = cv2.imread('xxxx')

# 将(0,0)坐标的像素修改为白色
img[0,0] = [255,255,255]

## 保存文件
cv2.imwrite('xxx.png',bgrImage)

# 显示图像
cv2.imshow()