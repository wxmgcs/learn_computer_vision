import cv2
import numpy as np
from scipy import ndimage
from sys import argv

if len(argv) ==  2:
    file_name = argv[1]
else:
    file_name = "../images/statue_small.jpg"

kernel_3x3 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])

# 转换为灰度图像，numpy只接收一维数组
img = cv2.imread(file_name, 0)

# 3x3 卷积 实现高通滤波器
k3 = ndimage.convolve(img, kernel_3x3)
# 5x5 卷积 实现高通滤波器
k5 = ndimage.convolve(img, kernel_5x5)

# 通过对图像应用 低通滤波器 后，与原图像计算差值
blurred = cv2.GaussianBlur(img, (17,17), 0)
g_hpf = img - blurred

cv2.imwrite("../samples/hpf_3x3.jpg",k3)
cv2.imwrite("../samples/hpf_5x5.jpg",k5)
cv2.imwrite("../samples/hpf_g_hpf.jpg",g_hpf)
# cv2.imshow("3x3", k3)
# cv2.imshow("5x5", k5)
# cv2.imshow("g_hpf", g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()
