import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sys import argv

if len(argv) ==  2:
    file_name = argv[1]
else:
    file_name = '../images/basil.jpg'

def water_shed(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    # 去除噪声 ,对图像膨胀之后再进行腐蚀操作
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    
    cv.imwrite("../samples/watershed.jpg",img)
    # plt.imshow(img)
    # plt.show()


img = cv.imread(file_name)
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow("input image",img)
start = cv.getTickCount()
water_shed(img)
end = cv.getTickCount()

time = (end-start)/cv.getTickFrequency()

print ("time is :%s ms"%(time*1000))
cv.waitKey(0)
cv.destroyAllWindows()