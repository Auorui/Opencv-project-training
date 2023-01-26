import os
import cv2
import mediapipe as mp
import time
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

Wcam, Hcam = 980, 980
cap.set(3, Wcam)
cap.set(4, Hcam)
cap.set(10,150)


img_path="image_figures"
mulu=os.listdir(img_path)
print(mulu)
Laylist=[]
for path in mulu:
    image=cv2.imread(f"{img_path}/{path}")
    Laylist.append(image)

detector = htm.handDetector(detectionCon=0.75)

while 1:
    _, img = cap.read()

    detector.findHands(img)
    lmList,_= detector.findPosition(img, draw=False)


    if len(lmList) != 0:
        fingerup=detector.fingersUp()
        print(fingerup)
        all_figures=fingerup.count(1)
        print(all_figures)
        h, w, _ = Laylist[all_figures].shape
        img[0:h, 0:w] = Laylist[all_figures]
        # img[0:300,0:220]=Laylist[0]

        cv2.rectangle(img,(0,350),(220,550),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(all_figures),(45,510),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)


    #################打印帧率#####################
    fps, img = fpsReader.FPS(img,pos=(880,50))
    cv2.imshow("image",img)
    k=cv2.waitKey(1)
    if k==27:
        break
