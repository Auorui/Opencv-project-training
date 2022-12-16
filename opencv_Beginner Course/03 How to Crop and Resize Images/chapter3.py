import cv2
import numpy as np


path="Resources/shapes.png"

def read_resize_image(path, size=1.0):
    """
    读取原始图片文件并从新设置大小
     1) 使用imread读取文件
     2）使用resize从新设置文件大小
     3) 函数返回的是图片的原始数据，可理解为一个三维数组，每个维度分别表示长、宽、通道数，数组中的值表示图片每个像素点不同通道的取值
    """

    original_image = cv2.imread(path)
    if size != 1.0:
        height, width = original_image.shape[:2]
        size = (int(width * size), int(height * size))
        original_image = cv2.resize(original_image, size, interpolation=cv2.INTER_AREA)

    print(original_image)
    print(original_image.shape)

    return original_image

img = cv2.imread(path)
print(img.shape)

imgResize = cv2.resize(img,(1000,500))
print(imgResize.shape)

imgCropped = img[20:245,357:495]

cv2.imshow("Image",img)
# cv2.imshow("Image Resize",imgResize)
cv2.imshow("Image Cropped",imgCropped)

cv2.waitKey(0)
