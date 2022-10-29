import cv2
import time
import HandTrackingModule as htm
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0


detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.GetMute()
volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
##print的结果(-74.0, 0.0, 1.0)
minvol,maxvol = volRange[0],volRange[1]
vol=0
volBar=400
volPer=0
while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        # print(lmList[4])
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2


        cv2.circle(img,(x1,y1),15,(125, 125, 255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (125, 125, 255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(125, 125, 255),3)
        cv2.circle(img, (cx, cy), 15, (125, 125, 255), cv2.FILLED)
        length=math.hypot(x2-x1,y2-y1)
        if length<50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        # print(length)

        #handrange  50——300
        #volume range -74——0

        vol = np.interp(length, [50, 300], [minvol, maxvol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 150), (85, 400), (125, 125, 255), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (125, 125, 255), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (125, 125, 255), 3)

#################打印帧率#####################
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 100, 100), 3)

    cv2.imshow("Img", img)
    k=cv2.waitKey(1)
    if k==27:break