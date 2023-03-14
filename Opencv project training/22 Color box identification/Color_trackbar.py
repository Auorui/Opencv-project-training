import cv2
import numpy as np
from bgr_detector import BGR,empty


path = 'test.png'

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,250)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)


while True:
    img = cv2.imread(path)
    bgr = BGR(img)
    #图像转化为HSV格式，H:色调S:饱和度V:明度
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #获取轨迹栏位
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)

    #创建一个蒙版，提取需要的颜色为白色，不需要的颜色为白色
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    imgStack = bgr.stackImages(0.5,([img,imgHSV],[mask,imgResult]))
    #定义比例尺
    cv2.imshow("Stacked Images", imgStack)

    if cv2.waitKey(1) & 0xFF == 27:
        break

