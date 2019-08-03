import cv2 as cv
import numpy as np
from PIL import  Image
import pytesseract
from matplotlib import pyplot as plt

def hsv_handler0(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    cv.imwrite("./test2.png", hsv)
    lower_hsv = np.array([35, 43, 46])  # 设置过滤的颜色的低值
    upper_hsv = np.array([77, 255, 255])  # 设置过滤的颜色的高值
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)  # 调节图像颜色
    cv.imwrite("./test22.png", mask)

def hsv_handler(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    cv.imwrite("./test1.png",hsv)

    lower_hsv = np.array([35, 43, 46])#设置过滤的颜色的低值
    upper_hsv = np.array([77, 255, 255])#设置过滤的颜色的高值
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)#调节图像颜色
    cv.imwrite("./test11.png",mask)

def split_and_merge(img):
    # 通道分离，输出三个单通道图片
    b, g, r = cv.split(img)  # 将彩色图像分割成3个通道
    # cv.imshow("blue", b)
    # cv.imshow("green", g)
    # cv.imshow("red", r)

    # 通道合并
    src = cv.merge([b, g, r])
    cv.imshow("merge", src)

    # 修改某个通道的值
    src[:, :, 2] = 100
    cv.imshow("signle channel", src)


# -*- coding=GBK -*-
import cv2 as cv
import numpy as np


# 模版匹配
def template_image():
    tpl = cv.imread("C://2.jpg")
    target = cv.imread("C://1.jpg")
    cv.imshow("模板", tpl)
    cv.imshow("原图", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("匹配" + np.str(md), target)


# 高斯金字塔
def pyramid_image(image):
    level = 3  # 金字塔的层数
    temp = image.copy()  # 拷贝图像
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("高斯金字塔" + str(i), dst)
        temp = dst.copy()
    return pyramid_images


# 拉普拉斯金字塔
def laplian_image(image):
    pyramid_images = pyramid_image(image)
    level = len(pyramid_images)
    for i in range(level - 1, -1, -1):
        if (i - 1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("拉普拉斯" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            cv.imshow("拉普拉斯" + str(i), lpls)

# 图像梯度：索贝尔算子
def sobel_image(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)  # x方向导数
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)  # y方向导数
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("X方向", gradx)  # 颜色变化在水平分层
    cv.imshow("Y方向", grady)  # 颜色变化在垂直分层
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("合成", gradxy)

#图像梯度：scharr算子：增强边缘
def scharr_image(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)#x方向导数
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)#y方向导数
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("X方向", gradx)#颜色变化在水平分层
    cv.imshow("Y方向", grady)#颜色变化在垂直分层
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("合成", gradxy)

#图像梯度： 拉普拉斯算子
def lapalian_image(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("拉普拉斯", lpls)


# 轮廓发现
def contous_image(image):
    dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 二值化
    cv.imshow("binary", binary)
    contous, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contou in enumerate(contous):
        cv.drawContours(image, contous, i, (0, 0, 255), 1)
    # 轮廓
    cv.imshow("contous", image)
    for i, contou in enumerate(contous):
        cv.drawContours(image, contous, i, (0, 0, 255), -1)
    # 轮廓覆盖
    cv.imshow("contous_recovey", image)


# 分水岭算法
def water_image(img):
    src = img
    print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)  # 去除噪点

    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("二值图像", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    cv.imshow("形态操作", sure_bg)

    # distance transform
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("距离变换", dist_output * 70)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("寻找种子", surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    # watershed transfrom
    markers += 1
    markers[unknown == 255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 0, 255]
    cv.imshow("分水岭结果", src)


# 人脸检测
def face_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 5)  # 第二个参数是移动距离，第三个参数是识别度，越大识别读越高
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 后两个参数，一个是颜色，一个是边框宽度
    # cv.imshow("result", img)
    cv.imwrite("test_face_result.png",img)


# 识别验证码
def recognize_text(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 6))
    binl = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    open_out = cv.morphologyEx(binl, cv.MORPH_OPEN, kernel)
    cv.bitwise_not(open_out, open_out)  # 背景变为白色
    cv.imshow("转换", open_out)
    textImage = Image.fromarray(open_out)
    text = pytesseract.image_to_string(textImage)
    print("This OK:%s" % text)



# 图像的开闭操作
def open_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 二值化
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    # 开操作
    cv.imshow("open image", binary)

def close_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    # 闭操作
    cv.imshow("close image", binary)

# 圆检测
def circles_image(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 255), 2)
    # 圆形
    cv.imshow("circle", image)


# 霍夫直线检测
def line_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 100, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("直线", image)

# 图像二值化 0白色 1黑色
# 全局阈值
def threshold_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("原来", gray)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    print("阈值：%s" % ret)
    cv.imshow("OTSU", binary)

    ret, binary = cv.threshold(gray, 0, 255,
                               cv.THRESH_BINARY | cv.THRESH_TRIANGLE)  # TRIANGLE法,，全局自适应阈值, 参数0可改为任意数字但不起作用，适用于单个波峰
    print("阈值：%s" % ret)
    cv.imshow("TRIANGLE", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)  # 自定义阈值为150,大于150的是白色 小于的是黑色
    print("阈值：%s" % ret)
    # 自定义
    cv.imshow("self define", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)  # 自定义阈值为150,大于150的是黑色 小于的是白色
    print("阈值：%s" % ret)
    # 自定义反色
    cv.imshow("self define reserver", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TRUNC)  # 截断 大于150的是改为150  小于150的保留
    print("阈值：%s" % ret)
    # 截断1
    cv.imshow("cut1", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TOZERO)  # 截断 小于150的是改为150  大于150的保留
    print("阈值：%s" % ret)
    # 截断2
    cv.imshow("cut2", binary)

# 画出图像的直方图
def hist_image(image):
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

# 直方图的应用
# 提升对比度（默认提升），只能是灰度图像
def equalHist_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 原来
    cv.imshow("origin", gray)  # 因为只能处理灰度图像，所以输出原图的灰度图像用于对比
    dst = cv.equalizeHist(gray)
    # 默认处理
    cv.imshow("default handler", dst)
# 对比度限制（自定义提示参数）
def clahe_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # clipLimit是对比度的大小，tileGridSize是每次处理块的大小
    dst = clahe.apply(gray)
    # 自定义处理
    cv.imshow("self define handler", dst)

# 图像模糊处理
def mo_image(src1):
    src2 = cv.blur(src1, (5, 5))
    # 均值模糊
    cv.imshow("1", src2)

    src2 = cv.medianBlur(src1, 5)
    # 中值模糊
    cv.imshow("2", src2)

    src2 = cv.GaussianBlur(src1, (5, 5), 2)
    # 高斯模糊
    cv.imshow("3", src2)

    # 双边滤波
    src2 = cv.bilateralFilter(src1, 5, 5, 2)
    cv.imshow("4", src2)
# 1.均值模糊函数blur()：定义：blur(src,ksize,dst=None, anchor=None, borderType=None)
# 定义是有5个参数，但最后三个均为none,所以也就2个参数
#   src：要处理的原图像
#   ksize: 周围关联的像素的范围：代码中（5，5）就是9*5的大小，就是计算这些范围内的均值来确定中心位置的大小
#
# 2.中值模糊函数medianBlur(): 定义：medianBlur(src, ksize, dst=None)
# ksize与blur()函数不同，不是矩阵，而是一个数字，例如为5，就表示了5*5的方阵
#
# 3.高斯平滑函数GaussianBlur():定义：GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
# sigmaX：标准差
#
#4.双边滤波函数bilateralFilter():定义：bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
# d：邻域直径
# sigmaColor：颜色标准差
# sigmaSpace：空间标准差

#自定义模糊函数
def zi_image(src1):
    kernel1 = np.ones((5, 5), np.float)/25#自定义矩阵，并防止数值溢出
    src2 = cv.filter2D(src1, -1, kernel1)
    # 自定义均值模糊
    cv.imshow("1", src2)
    kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    src2 = cv.filter2D(src1, -1, kernel2)
    # 自定义锐化
    cv.imshow("2", src2)

img1 = cv.imread("/Volumes/DATA/ctu_gitlab/vr_images/src/vr/images/tripod1.png")
img2 = cv.imread("/Volumes/DATA/ctu_gitlab/vr_images/src/vr/images/tripod2.png")
img_face = cv.imread("test_face.jpg")
# split_and_merge(img1)
# laplian_image(img1)
# laplian_image(img2)
# sobel_image(img1)
# scharr_image(img1)
# lapalian_image(img1)

# contous_image(img1)
# contous_image(img2)
# face_image(img_face)

# open_image(img1)
# close_image(img1)

# circles_image(img1)
# line_image(img1)

# threshold_image(img1)

# hist_image(img2)

# equalHist_image(img1)
# clahe_image(img1)

# mo_image(img1)
zi_image(img1)

cv.waitKey(0)
cv.destroyAllWindows()