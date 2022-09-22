import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def createBox(img, points, scale=3):
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    img = cv2.bitwise_and(img, mask)       #按位与运算，将蒙版与图像共有的区域交叉
    cv2.imshow('Mask',img)

    bbox = cv2.boundingRect(points)
    x, y, w, h = bbox
    imgCrop = img[y:y + h, x:x + w]
    imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
    return imgCrop

img = cv2.imread('1.png')
img = cv2.resize(img, (0, 0), None, 0.80, 0.80)
imgOriginal = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgOriginal)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    imgOriginal=cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmarks = predictor(imgGray, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])
    myPoints = np.array(myPoints)
    imgLips = createBox(img, myPoints[48:61])
    cv2.imshow('Lips', imgLips)

cv2.imshow("Originial", imgOriginal)
cv2.waitKey(0)



