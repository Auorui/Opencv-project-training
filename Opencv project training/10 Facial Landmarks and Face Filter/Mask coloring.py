import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def createBox(img, points, scale=3, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
        # cv2.imshow('Mask',mask)

    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y + h, x:x + w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        cv2.imwrite("Mask.jpg", imgCrop)
        return imgCrop
    else:
        return mask

img = cv2.imread('1.png')
img = cv2.resize(img, (0, 0), None, 0.80, 0.80)
imgOriginal = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgOriginal)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    landmarks = predictor(imgGray, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])

    myPoints = np.array(myPoints)
    maskLips = createBox(img, myPoints[48:61], masked=True, cropped=False)

    imgColorLips = np.zeros_like(maskLips)
    imgColorLips[:] = 153, 0, 158
    imgColorLips = cv2.bitwise_and(maskLips, imgColorLips)     #用位运算将蒙版与纯颜色背景板结合起来
    imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)   #添加高斯模糊，不让图像变得生硬
    imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)   #配置权重，使颜色与嘴唇更加融合
    cv2.imshow('Color', imgColorLips)
    cv2.imshow('Lips', maskLips)

cv2.imshow("Originial", imgOriginal)
cv2.waitKey(0)



