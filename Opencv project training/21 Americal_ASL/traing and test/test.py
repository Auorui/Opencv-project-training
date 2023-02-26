import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
from cvzone.ClassificationModule import Classifier


cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("E:\pythonProject1\Opencv project training//21 Americal_ASL\Model\keras_model.h5",
                      "E:\pythonProject1\Opencv project training//21 Americal_ASL\Model\labels.txt")

offset=20
imgSize=300
# folder=".21 Americal_ASL/Data/D"
count=0
labels = ['A','B','C','D']

while True:
    ret,img=cap.read()
    imgOutput = img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h/w

        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wGap+wCal]=imgResize
            prediction, index =  classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - 50),
                      (x - offset+120, y - offset), (255, 0,  255), cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x+12,y-27),cv2.FONT_HERSHEY_COMPLEX,1.8,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),
                      (x+w+offset,y+h+offset),(255,0,255),4)

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("imageWhite",imgWhite)

    cv2.imshow("Image",imgOutput)
    k=cv2.waitKey(1)
    if k==27:
        break
