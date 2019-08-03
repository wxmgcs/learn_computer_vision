# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:56:41 2017

@author: cross
"""
import numpy as np
import cv2
import random

if __name__ == '__main__':

    img = cv2.imread("../tripod.png",4)
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    seed_pt = None
    fixed_range = True
    connectivity = 4

    def update(dummy=None):
        if seed_pt is None:
            cv2.imshow('floodfill', img)
            return
        flooded = img.copy()
        mask[:] = 1
        lo = cv2.getTrackbarPos('lo', 'floodfill')
        hi = cv2.getTrackbarPos('hi', 'floodfill')
        flags = connectivity
        if fixed_range:
            flags |= cv2.FLOODFILL_FIXED_RANGE

        cv2.floodFill(flooded, mask, seed_pt, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), (lo,)*3, (hi,)*3, flags)

        cv2.circle(flooded, seed_pt, 2, (0, 0, 255), -1)#选定基准点用红色圆点标出
        cv2.imshow('floodfill', flooded)

    def onmouse(event, x, y, flags, param):#鼠标响应函数
        global seed_pt
        if flags & cv2.EVENT_FLAG_LBUTTON:#鼠标左键响应，选择漫水填充基准点
            seed_pt = x, y
            update()

    update()
    cv2.setMouseCallback('floodfill', onmouse)
    cv2.createTrackbar('lo', 'floodfill', 20, 255, update)
    cv2.createTrackbar('hi', 'floodfill', 20, 255, update)

    while True:
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
        if ch == ord('f'):
            fixed_range = not fixed_range #选定时flags的高位比特位0，也就是邻域的选定为当前像素与相邻像素的的差，这样的效果就是联通区域会很大
            print ('using %s range' % ('floating', 'fixed')[fixed_range])
            update()
        if ch == ord('c'):
            connectivity = 12-connectivity #选择4方向或则8方向种子扩散
            print ('connectivity =', connectivity)
            update()
    cv2.destroyAllWindows()
