import cv2
import numpy
import os

randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)
# 根据随机字节的bytearray转换为灰度图像

grayImage = flatNumpyArray.reshape(300,400)
cv2.imwrite('./RandomGray.png',grayImage)

# 根据随机字节的bytearray转换BGR图像
bgrImage = flatNumpyArray.reshape(100,400,3)
cv2.imwrite('./RomdomColor.png',bgrImage)

