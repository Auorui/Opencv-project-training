import cv2
import tally as ta
import os
from datetime import datetime

# 通用
success_text = "Successfully saved"
success_text_color = (0, 0, 255)
text_size = 20
image_count = 1
data = []

####################----------轨迹栏初始化----------#####################
cv2.namedWindow("Settings")                                           #
cv2.resizeWindow("Settings", 640, 240)                                #
cv2.createTrackbar("Threshold1", "Settings", 82, 255, ta.empty)       #
cv2.createTrackbar("Threshold2", "Settings", 101, 255, ta.empty)      #
########################################################################

##########-----模式检测-----##########
print("Real-time | Multi-image")    #
print("请选择检测识别模式：",end=' ')  #
recongnitionMode = input()          #
#####################################

save_dir, excel_path=ta.ImagePath(recongnitionMode)

if recongnitionMode=='Real-time':
    Vcap = ta.VideoCap()
    Vcap.CapInit(mode=0, w=845, h=480)
    while True:
        img = Vcap.read()
        imgPre = ta.preProcessing(img)
        imgStacked, total, imgContours = ta.drawContour(img, imgPre, minArea=15)

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

            ta.excelmation(excel_path, data)
        elif k == 27:
            break

if recongnitionMode == 'Multi-image':
    while True:
        print("是否经过Single-image检测(Y or N)（或按下Esc键退出）:", end=' ')
        Single_mode = input()
        if Single_mode.lower() == 'esc':
            break
        if Single_mode == 'Y':
            while True:
                print("请输入要检测图片的路径（或按下Esc键返回）:", end=' ')
                path = input().strip()  # ./photodata/image005.png
                if path.lower() == 'esc':
                    break
                if path.lower() == 'exit()':
                    break
                img = cv2.imread(path)
                while True:
                    imgPre = ta.preProcessing(img)
                    imgStacked, total, imgContours = ta.drawContour(img, imgPre, minArea=15)
                    k = cv2.waitKey(1)
                    if k == 27:
                        break
        elif Single_mode == 'N':
            print("请输入图片文件夹路径:", end=' ')
            folder_path = input().strip()
            if folder_path.lower() == 'esc':
                break
            if folder_path.lower() == 'exit()':
                break
            image_files = ta.readPath(folder_path)
            # print(image_files)
            for image in image_files:
                print(image)
                img = cv2.imread(image)
                imgPre = ta.preProcessing(img)
                imgStacked, total, imgContours = ta.drawContour(img, imgPre, minArea=15)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                image_name = f"Picimg_{str(image_count).zfill(3)}.png"
                image_path = os.path.join(save_dir, image_name)
                cv2.imwrite(image_path, imgContours)
                data.append([current_time, image_name, total])
                image_count += 1
                ta.excelmation(excel_path, data)
            print("多文件检测结束！")
            break

        elif Single_mode.lower() == 'exit()':
            break