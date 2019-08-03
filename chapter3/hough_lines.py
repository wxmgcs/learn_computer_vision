import cv2
import numpy as np
from sys import argv
if len(argv) ==  2:
    file_name = argv[1]
else:
    file_name = 'lines.jpg'
    
img = cv2.imread(file_name)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,120)
minLineLength = 20
maxLineGap = 5
lines = cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
  cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


cv2.imwrite("../samples/hough_lines_edges.jpg",edges)
cv2.imwrite("../samples/hough_lines_lines.jpg",lines)

# cv2.imshow("edges", edges)
# cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()
