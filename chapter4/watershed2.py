import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sys import argv

if len(argv) ==  2:
    file_name = argv[1]
else:
    file_name = '../images/statue_small.jpg'
    
img = cv.imread(file_name)
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (100,50,421,378)
cv.grabCut


mask