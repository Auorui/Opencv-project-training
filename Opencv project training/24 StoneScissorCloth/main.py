import pyzjr as pz
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import random
import cvzone

Vap=pz.VideoCap()
Vap.CapInit()
imgList,all = pz.getPhotopath(r"Resources",debug=False)
detector = HandDetector(maxHands=1,detectionCon=0.8)
timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
imgAI = None
initialTime = 0
while True:
    img = Vap.read(flip=1)
    imgbackground = cv2.imread(imgList[0])

    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)[:, 80:480]
    hands = detector.findHands(imgScaled,draw=False)

    if startGame:

        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgbackground, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                if hands:
                    playerMove = None
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)
                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1
                    if fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2
                    if fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3

                    randomNumber = random.randint(1, 3)
                    imgAI = cv2.imread(imgList[randomNumber], cv2.IMREAD_UNCHANGED)
                    imgbackground = cvzone.overlayPNG(imgbackground, imgAI, (149, 310))

                    winning_rules = {
                        # 1-石头,2-布,3-剪刀
                        (1, 3): "player",  # 石头 vs 剪刀
                        (2, 1): "player",  # 布 vs 石头
                        (3, 2): "player",  # 剪刀 vs 布
                        (3, 1): "AI",      # 剪刀 vs 石头
                        (1, 2): "AI",      # 石头 vs 布
                        (2, 3): "AI",      # 布 vs 剪刀
                    }
                    result = winning_rules.get((playerMove, randomNumber))
                    if result == "player":
                        scores[1] += 1
                    elif result == "AI":
                        scores[0] += 1

    imgbackground[234:654, 795:1195] = imgScaled

    if stateResult:
        imgbackground = cvzone.overlayPNG(imgbackground, imgAI, (149, 310))

    cv2.putText(imgbackground, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, pz.white, 6)
    cv2.putText(imgbackground, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, pz.white, 6)

    cv2.imshow("background", imgbackground)

    k = cv2.waitKey(1)
    if k == 32:
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif k == 27:
        break