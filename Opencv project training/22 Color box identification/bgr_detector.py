import cv2
import numpy as np


class BGR():
    def __init__(self,img,scale=0.7):
        self.img=img
        self.scale=scale
        self.imgResult=img.copy()
        self.myColors = [[5, 107, 0, 19, 255, 255],
                        [133, 56, 0, 159, 156, 255],
                        [57, 76, 0, 100, 255, 255],
                        [90, 48, 0, 118, 255, 255]]
        self.myColorValues = [[51, 153, 255],  ## BGR
                              [255, 0, 255],   # https://www.rapidtables.org/zh-CN/web/color/RGB_Color.html
                              [0, 255, 0],
                              [255, 0, 0]]
        self.objectColor=["Orange","Purple","Green","Blue"]
    def stackImages(self,scale,imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, self.scale, self.scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, self.scale, self.scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor
        return ver

    def getContours(self,img, minArea=500):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = 0, 0, 0, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > minArea:
                cv2.drawContours(self.imgResult, cnt, -1, (255, 0, 0), 3)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(self.imgResult, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return x + w // 2, y,w,h

    def findColor(self,img):
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        count = 0
        for color in self.myColors:
            lower = np.array(color[0:3])
            upper = np.array(color[3:6])
            mask = cv2.inRange(imgHSV, lower, upper)
            x,y,w,h=self.getContours(mask)
            cv2.putText(self.imgResult, self.objectColor[count],
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)
            count += 1
        return self.imgResult


def empty(ways):
    pass
