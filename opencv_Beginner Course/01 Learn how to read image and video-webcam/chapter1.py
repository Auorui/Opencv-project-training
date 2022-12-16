######################## READ IMAGE ############################
import cv2
# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("Resources/lena.png")
# DISPLAY
cv2.imshow("Lena Soderberg",img)
cv2.waitKey(0)

######################### READ VIDEO #############################
# import cv2
# frame_Width = 640
# frame_Height = 480
# cap = cv2.VideoCapture("E:/pycharmlujin/Opencv/Opencv_learning/Resources/test_video.mp4")
# while True:
#     success, img = cap.read()
#     img = cv2.resize(img, (frame_Width, frame_Height))
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
######################### READ WEBCAM  ############################

# import cv2
# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)    #设置宽度q
# cap.set(4, frameHeight)   #设置高度
# cap.set(10,150)   #设置亮度
# while True:
#     success, img = cap.read()
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


