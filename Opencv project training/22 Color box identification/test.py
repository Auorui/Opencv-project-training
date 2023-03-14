import cv2
import numpy as np
from bgr_detector import BGR

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

myPoints =  []  ## [x , y , colorId ]

while True:
    success, img = cap.read()

    bgr = BGR(img)
    imgResult=bgr.findColor(img)
    imgStack = bgr.stackImages(0.8, ([img, imgResult]))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == 27:
        break

