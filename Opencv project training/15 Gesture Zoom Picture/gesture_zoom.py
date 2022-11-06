import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

startDist = None
scale = 0
cx, cy = 500,200
wCam, hCam = 1280,720
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(10,150)

detector = htm.handDetector(detectionCon=0.75)

while 1:
    success, img = cap.read()
    handsimformation,img=detector.findHands(img)

    img1 = cv2.imread("1.png")
    # img[0:360, 0:260] = img1
    if len(handsimformation)==2:

        # print(detector.fingersUp(handsimformation[0]),detector.fingersUp(handsimformation[1]))
        #detector.fingersUp(handimformation[0]右手
        if detector.fingersUp(handsimformation[0]) == [1, 1, 1, 0, 0] and \
                detector.fingersUp(handsimformation[1]) == [1, 1, 1 ,0, 0]:
            lmList1 = handsimformation[0]['lmList']
            lmList2 = handsimformation[1]['lmList']
            if startDist is None:
                #lmList1[8],lmList2[8]右、左手指尖

                # length,info,img=detector.findDistance(lmList1[8],lmList2[8], img)
                length, info, img = detector.findDistance(handsimformation[0]["center"], handsimformation[1]["center"], img)
                startDist=length
            length, info, img = detector.findDistance(handsimformation[0]["center"], handsimformation[1]["center"], img)
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
            scale=int((length-startDist)//2)
            cx, cy=info[4:]
            print(scale)
    else:
        startDist=None
    try:
        h1, w1, _ = img1.shape
        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
        img1 = cv2.resize(img1, (newW, newH))

        img[cy-newH//2:cy+ newH//2, cx-newW//2:cx+newW//2] = img1
    except:
        pass
    #################打印帧率#####################
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (100, 0, 255), 3)

    cv2.imshow("image",img)
    k=cv2.waitKey(1)
    if k==27:
        break
















