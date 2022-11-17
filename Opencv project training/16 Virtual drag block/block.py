import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (0, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class Moveblock():
    def __init__(self,posCenter,size=(200,200)):
        self.posCenter=posCenter
        self.size=size

    def update(self,cursor):
        cx,cy=self.posCenter
        w,h=self.size

        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor[:2]

rectList=[]
for i in range(5):
    rectList.append(Moveblock((i*250+150,150)))

while True:
    succes,img=cap.read()
    img = cv2.flip(img,1)
    lmList ,img = detector.findHands(img)
    # print(lmList)
    if lmList:
        lmList1 = lmList[0]["lmList"]
        length, info, img = detector.findDistance(lmList1[8][:2],lmList1[12][:2],img,draw=False)
        print(length)
        if length<60:
            # lmList1 = lmList[0]["lmList"]
            cursor = lmList1[8]
            # print(cursor)
            for rect in rectList:
                rect.update(cursor)
    #实体框
    # for rect in rectList:
    #     cx, cy = rect.posCenter
    #     w, h = rect.size
    #     cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),colorR,cv2.FILLED)
    #
    #     cvzone.cornerRect(img,(cx-w//2,cy-h//2, w, h),20,rt=0)

    #半透明
    imgNew=np.zeros_like(img,np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),colorR,cv2.FILLED)

        cvzone.cornerRect(imgNew,(cx-w//2,cy-h//2, w, h),20,rt=0)
    out = img.copy()
    alpha=0.6
    mask=imgNew.astype(bool)
    out[mask]=cv2.addWeighted(img,alpha,imgNew,1-alpha,0)[mask]

    cv2.imshow("Image", out)
    k=cv2.waitKey(1)
    if k==27:break



