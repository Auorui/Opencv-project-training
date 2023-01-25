import cv2
import numpy as np
import time
import autopy
import HandTrackingModule as htm

class fpsReader():
    def __init__(self):
        self.pTime = time.time()
    def FPS(self,img=None,pos=(20, 50), color=(255, 255, 0), scale=3, thickness=3):
        cTime = time.time()
        try:
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            if img is None:
                return fps
            else:
                cv2.putText(img, f'FPS: {int(fps)}', pos, cv2.FONT_HERSHEY_PLAIN,
                            scale, color, thickness)
                return fps, img
        except:
            return 0
fpsReader = fpsReader()
cap=cv2.VideoCapture(0)

smooth=6
clocx,plocx=0,0
clocy,plocy=0,0
Boundary=100
Wcam, Hcam=640,480
Wscr, Hscr=autopy.screen.size()
# print(Wscr,Hscr)
#1536.0 864.0

cap.set(3,Wcam)
cap.set(4,Hcam)
detector=htm.handDetector(maxHands=1, detectionCon=0.65)


while True:
    _, img=cap.read()

    img=detector.findHands(img, draw=True)
    lmList,bbox=detector.findPosition(img)
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        fingersUp=detector.fingersUp()
        # print(fingersUp)
        cv2.rectangle(img,(Boundary,Boundary),(Wcam-Boundary,Hcam-Boundary),(255,0,0),thickness=3)
        if fingersUp[1]==1 and fingersUp[2]==0:
            x3 = np.interp(x1, (0, Wcam), (0, Wscr))
            y3 = np.interp(y1, (0, Hcam), (0, Hscr))

            clocx=plocx+(x3-plocx)/smooth
            clocy = plocy + (y3 - plocy) / smooth

            autopy.mouse.move(Wscr-clocx,clocy)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocx,plocy=clocx,clocy


        if fingersUp[1] == 1 and fingersUp[2] == 1:
            length, info, img=detector.findDistance(8,12,img)
            # print(length) 30可能是一个不错的范围
            if length<30:
                cv2.circle(img,(info[-2],info[-1]),15,(0,0,255),cv2.FILLED)
                autopy.mouse.click()
    new_window = cv2.flip(img, 1)
    fps, img = fpsReader.FPS(new_window)
    cv2.imshow("img",img)
    k=cv2.waitKey(1)
    if k==27:
        break


