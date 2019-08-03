import cv2
import numpy as np

filepath = "../images/statue_small.jpg"
filepath = "/Users/wangxiaomin/Downloads/论文及源程序/rice.png"
filepath = "/Users/wangxiaomin/Downloads/论文及源程序/lung_CT.png"
img = cv2.imread(filepath, 0)
cv2.imwrite("canny2.jpg", cv2.Canny(img, 200, 300))
# cv2.imshow("canny", cv2.imread("canny2.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()
