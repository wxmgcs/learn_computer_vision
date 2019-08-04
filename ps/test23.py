# (1)、nRGB = RGB + (RGB - Threshold) * Contrast / 255
# 公式中，nRGB表示图像像素新的R、G、B分量，RGB表示图像像素R、G、B分量，
# Threshold为给定的阈值，Contrast为处理过的对比度增量。
# Photoshop对于对比度增量，是按给定值的正负分别处理的：
# 当增量等于-255时，是图像对比度的下端极限，
# 此时，图像RGB各分量都等于阈值，图像呈全灰色，灰度图上只有1条线，即阈值灰度；
import matplotlib.pyplot as plt
from skimage import io


file_name='tripod.png'
img=io.imread(file_name)

img = img * 1.0

thre = img.mean()

# -100 - 100
contrast = -55.0

img_out = img * 1.0

if contrast <= -255.0:
    img_out = (img_out >= 0) + thre -1
elif contrast > -255.0 and contrast < 0:
    img_out = img + (img - thre) * contrast / 255.0
elif contrast < 255.0 and contrast > 0:
    new_con = 255.0 *255.0 / (256.0-contrast) - 255.0
    img_out = img + (img - thre) * new_con / 255.0
else:
    mask_1 = img > thre
    img_out = mask_1 * 255.0

img_out = img_out / 255.0

# 饱和处理
mask_1 = img_out  < 0
mask_2 = img_out  > 1

img_out = img_out * (1-mask_1)
img_out = img_out * (1-mask_2) + mask_2


plt.figure()
plt.imshow(img/255.0)
plt.axis('off')

plt.figure(2)
plt.imshow(img_out)
plt.axis('off')
plt.show()