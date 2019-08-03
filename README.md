#### 参考资料
《OpenCV3 计算机视觉 python语言实现第2版》

#### 常见问题
Traceback (most recent call last):
  File "watershed.py", line 12, in <module>
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.error: OpenCV(4.1.0) /Users/travis/build/skvark/opencv-python/opencv/modules/imgproc/src/color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
解决办法: 检查cv2.imread(xxxx) 被读取的图片是否存在