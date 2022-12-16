import cv2
import numpy as np

img = cv2.imread("Resources/cards.jpg")

width,height = 250,350
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
#大概数值可通过PS的标尺得到
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
#透视变换函数，src：源图像中待测矩形的四点坐标，sdt：目标图像中矩形的四点坐标
imgOutput = cv2.warpPerspective(img,matrix,(width,height))
#参数：输入图像，变换矩阵，目标图像shape

cv2.imshow("Image",img)
cv2.imshow("Output",imgOutput)

cv2.waitKey(0)