import cv2

pic_path="source/AI.png"
video_path="source/C25.mp4"
frame_Width=640
frame_Heigh=480
brightness=150
# 读取图像

img = cv2.imread(pic_path)
cv2.imshow("PIC-Image",img)
cv2.waitKey(0)

# 读取视频
"""
cap = cv2.VideoCapture(video_path)
while True:
    success, img = cap.read()
    img = cv2.resize(img,(frame_Width,frame_Width))
    cv2.imshow("Video_Image",img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
"""
# # 调用镜头
# cap = cv2.VideoCapture(0)
# cap.set(3,frame_Width)
# cap.set(4,frame_Heigh)
# cap.set(10,brightness)
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video_Image",img)
#     k = cv2.waitKey(1)
#     if k == ord("q"):
#         break