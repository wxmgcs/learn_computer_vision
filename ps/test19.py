import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import numpy.matlib
from skimage import img_as_float

file_name='tripod.png'
img=io.imread(file_name)

img = img_as_float(img)

row, col, channel = img.shape

rNW = 0.5
rNE = 1.0
rSW = 1.0
rSE = 0.0

gNW = 0.0
gNE = 0.5
gSW = 0.0
gSE = 1.0

bNW = 1.0
bNE = 0.0
bSW = 1.0
bSE = 0.0

xx = np.arange (col)
yy = np.arange (row)

x_mask = numpy.matlib.repmat (xx, row, 1)
y_mask = numpy.matlib.repmat (yy, col, 1)
y_mask = np.transpose(y_mask)

fx = x_mask * 1.0 / col
fy = y_mask * 1.0 / row

p = rNW + (rNE - rNW) * fx
q = rSW + (rSE - rSW) * fx
r = ( p + (q - p) * fy )
r[r<0] = 0
r[r>1] =1

p = gNW + (gNE - gNW) * fx
q = gSW + (gSE - gSW) * fx
g = ( p + (q - p) * fy )
g[g<0] = 0
g[g>1] =1

p = bNW + (bNE - bNW) * fx
q = bSW + (bSE - bSW) * fx
b = ( p + (q - p) * fy )
b[b<0] = 0.0
b[b>1] = 1.0

img[:, :, 0] = r
img[:, :, 1] = g
img[:, :, 2] = b

plt.figure(1)
plt.imshow(img)
plt.axis('off');

plt.show();
