import cv2 as cv
import matplotlib.pyplot as plt

# 轮廓发现
def contous_image(image):
    dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 二值化
    # cv.imshow("binary", binary)
    cv.imwrite("./tripod_binary.png", binary)
    contous, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contou in enumerate(contous):
        cv.drawContours(image, contous, i, (0, 0, 255), 1)
    # 轮廓
    # cv.imshow("contous", image)
    cv.imwrite("./tripod_contous.png",image)
    for i, contou in enumerate(contous):
        print (i," , ",contou)
        cv.drawContours(image, contous, i, (0, 0, 255), -1)
    # 轮廓覆盖
    # cv.imshow("contous_recovey", image)

img1 = cv.imread("/Volumes/DATA/ctu_gitlab/vr_images/src/vr/images/tripod1.png")
img2 = cv.imread("/Volumes/DATA/ctu_gitlab/vr_images/src/vr/images/tripod2.png")
contous_image(img1)
cv.waitKey(0)
cv.destroyAllWindows()