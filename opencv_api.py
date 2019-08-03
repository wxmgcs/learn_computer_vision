#coding:utf-8
import cv2
#读取文件
def read_file(filepath):
    img = cv2.imread(filepath)
    return img

def get_fileinfo(img):
    (rows, columns , channels) = img.shape
    print ("图片宽:",rows)
    print ("图片长:",columns)
    print ("图片的属性:",img.shape)
    print ("图片的大小:",img.size)
    print ("图片的数据类型:",img.dtype)

img = read_file('tripod.png')
# 将(0,0)坐标的像素修改为白色
img[0,0] = [255,255,255]

print ("访问R层:",img.item(0,0,2))

# 修改R层的数据
img.itemset((0,0,2),100)
print ("访问R层:",img.item(0,0,2))

print ("ROI")
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

## 保存文件
cv2.imwrite('tripod_out.png',img)

# 显示图像
# cv2.imshow()