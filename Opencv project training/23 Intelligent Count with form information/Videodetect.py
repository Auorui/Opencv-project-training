"""
author : Auorui(夏天是冰红茶)
time : 2023-6-18
"""
import cv2
import tally as ta
import os
from datetime import datetime

Vcap = ta.VideoCap()
Vcap.CapInit(mode=0, w=845, h=480)

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshold1", "Settings", 82, 255, ta.empty)
cv2.createTrackbar("Threshold2", "Settings", 101, 255, ta.empty)

success_text = "Successfully saved"
success_text_color = (0, 0, 255)
text_size = 20

save_dir = "./Video/img"
excel_path = "./Video/data.xlsx"

image_count = 1
data = []

while True:
    img = Vcap.read()

    imgPre = ta.preProcessing(img)

    imgStacked,total,imgContours=ta.drawContour(img, imgPre, minArea=15)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_name = f"Videoimg_{str(image_count).zfill(3)}.png"
        image_path = os.path.join(save_dir, image_name)
        cv2.imwrite(image_path, imgContours)

        data.append([current_time, image_name, total])

        image_count += 1

        cv2.rectangle(imgStacked, (700, 140),
                      (1005, 190),
                      (255, 0, 0), -1)
        cv2.putText(imgStacked,
                    success_text,
                    (705, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size / 20, success_text_color, 2)

        cv2.imshow("Settings", imgStacked)
        while True:
            k = cv2.waitKey(1)
            if k == ord(' '):
                break

        ta.excelmation(excel_path,data)

    elif k == 27:
        break

cv2.destroyAllWindows()


