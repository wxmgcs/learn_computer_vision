from skimage import img_as_float
import matplotlib.pyplot as plt
from skimage import io
import random
import numpy as np

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 392

center_x = IMAGE_WIDTH/2
center_y = IMAGE_HEIGHT/2

R = np.sqrt(center_x**2 + center_y**2)

Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

# 利用 for 循环 实现
for i in range(IMAGE_HEIGHT):
    for j in range(IMAGE_WIDTH):
        dis = np.sqrt((i-center_y)**2+(j-center_x)**2)
        Gauss_map[i, j] = np.exp(-0.5*dis/R)

# 直接利用矩阵运算实现

mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

x1 = np.arange(IMAGE_WIDTH)
x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

y1 = np.arange(IMAGE_HEIGHT)
y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
y_map = np.transpose(y_map)

Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)

Gauss_map = np.exp(-0.5*Gauss_map/R)

# 显示和保存生成的图像
plt.figure()
plt.imshow(Gauss_map, plt.cm.gray)
plt.imsave('out_2.jpg', Gauss_map, cmap=plt.cm.gray)
plt.show()
