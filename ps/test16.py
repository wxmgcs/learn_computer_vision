# 我们知道，一般的非线性RGB亮度调整只是在原有R、G、B值基础上增加和减少一定量来实现的，而PS的明度调整原理还得从前面那个公式上去找。我们将正向明度调整公式：
# RGB = RGB + (255 - RGB) * value / 255
# 转换为
# RGB = (RGB * (255 - value) + 255 * value) / 255，
# 如果value用１表示最大值255，则为
# RGB = RGB * (1 - value) + 255 * value，
# 可以看出什么呢？凡是知道图像合成的人都知道这个公式，其实PS的明度调整是采用Alpha合成方式，这里的value就是Alpha，公式前面部分RGB * (1 - value)的是图像部分，后面的255 * value部分则是一个白色遮照层，明度越大，遮照层的Alpha越大，图像就越谈，反之亦然。而明度的负调整则是以一个黑色遮照层来完成的。负１００％就全黑了。只有遮照层Alpha=0，也就是明度值为０时，才是完完全全的图片显示。
# 明度调整，利用图层的合成
# 如果alpha大于0，相当于利用一个白色遮罩层合成
# RGB = RGB * (1 - alpha) + 255 * alpha;
# 如果alpha小于0，相当于利用一个黑色遮罩层合成
# RGB=RGB * (1+alpha) + 0 * alpha;
import matplotlib.pyplot as plt
from skimage import io

file_name='tripod.png';
img=io.imread(file_name)

# -255.0 - 255.0  alpha -1.0 - 1.0
Increment = 105.0;
alpha = Increment/255.0;

def Illumi_adjust(alpha, img):
    if alpha > 0 :
        img_out = img * (1 - alpha) + alpha * 255.0
    else:
        img_out = img * (1 + alpha)

    return img_out/255.0

img_out = Illumi_adjust(alpha, img)

plt.figure()
plt.imshow(img)
plt.axis('off')

plt.figure(2)
plt.imshow(img_out)
plt.axis('off')

plt.show()