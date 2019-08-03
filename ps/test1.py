import numpy as np
from skimage import img_as_float
import matplotlib.pyplot as plt
from skimage import io
import numpy.matlib

file_name2 = './tripod.png'
img=io.imread(file_name2)

img = img_as_float(img)

row, col, channel = img.shape
img_out = img * 1.0
A = 7.0
B = 3.0

center_x = (col-1)/2.0
center_y = (row-1)/2.0

xx = np.arange (col)
yy = np.arange (row)

x_mask = numpy.matlib.repmat (xx, row, 1)
y_mask = numpy.matlib.repmat (yy, col, 1)
y_mask = np.transpose(y_mask)

xx_dif = x_mask - center_x
yy_dif = center_y - y_mask

theta = np.arctan2(yy_dif,  xx_dif)
r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
r1 = r + A*col*0.01*np.sin(B*0.1*r)

x_new = r1 * np.cos(theta) + center_x
y_new = center_y - r1 * np.sin(theta)

int_x = np.floor (x_new)
int_x = int_x.astype(int)
int_y = np.floor (y_new)
int_y = int_y.astype(int)

for ii in range(row):
    for jj in range (col):
        new_xx = int_x [ii, jj]
        new_yy = int_y [ii, jj]

        if x_new [ii, jj] < 0 or x_new [ii, jj] > col -1 :
            continue
        if y_new [ii, jj] < 0 or y_new [ii, jj] > row -1 :
            continue

        img_out[ii, jj, :] = img[new_yy, new_xx, :]


plt.figure (1)
plt.imshow (img)
plt.axis('off')

plt.figure (2)
plt.imshow (img_out)
plt.axis('off')

plt.show()