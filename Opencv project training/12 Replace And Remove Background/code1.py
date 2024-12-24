import cv2
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time
import numpy as np
import copy

class FPS:
    """
    Detect video frame rate and refresh the video display
    Examples:
    ```
        fpsReader = FPS()
        Vcap = VideoCap(mode=0)
        while True:
            img = Vcap.read()
            fps, img = fpsReader.update(img)
            Vcap.show("ss", img)
    ```
    """
    def __init__(self):
        self.pTime = time.time()

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        """
        Update frame rate
        :param img: The displayed image can be left blank if only the fps value is needed
        :param pos: Position on FPS on image
        :param color: The color of the displayed FPS value
        :param scale: The proportion of displayed FPS values
        :param thickness: The thickness of the displayed FPS value
        """
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

def stackImages(_imgList, cols, scale):
    """
    Stack Images together to display in a single window
    :param _imgList: list of images to stack
    :param cols: the num of img in a row
    :param scale: bigger~1+ ans smaller~1-
    :return: Stacked Image
    """
    imgList = copy.deepcopy(_imgList)

    # Get dimensions of the first image
    width1, height1 = imgList[0].shape[1], imgList[0].shape[0]

    # make the array full by adding blank img, otherwise the openCV can't work
    totalImages = len(imgList)
    rows = totalImages // cols if totalImages // cols * cols == totalImages else totalImages // cols + 1
    blankImages = cols * rows - totalImages

    # Create a blank image with dimensions of the first image
    imgBlank = np.zeros((height1, width1, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)

    # resize the images to be the same as the first image and apply scaling
    for i in range(cols * rows):
        imgList[i] = cv2.resize(imgList[i], (width1, height1), interpolation=cv2.INTER_AREA)
        imgList[i] = cv2.resize(imgList[i], (0, 0), None, scale, scale)

        if len(imgList[i].shape) == 2:  # Convert grayscale to color if necessary
            imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_GRAY2BGR)

    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    return ver

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60)  #帧速率
segmentor=SelfiSegmentation()
fpsReader = FPS()


listimg=os.listdir("Images")
print(listimg)
imglist = []
for imgpath in listimg:
    img=cv2.imread(f'Images/{imgpath}')
    imglist.append(img)
print(len(imglist))

indeximg = 0

while 1:
    reg, img=cap.read()
    imgout = segmentor.removeBG(img,imglist[indeximg], 0.8)

    imgstacked = stackImages([img,imgout],2,1)
    fps,imgstacked=fpsReader.update(imgstacked,color=(0,0,255))
    print(indeximg)
    cv2.imshow("Imagestacked",imgstacked)
    key=cv2.waitKey(1)
    if key & 0xFF == 27:
        break
    elif key == ord('q'):
        if indeximg>0:
            indeximg -=1
    elif key == ord('w'):
        if indeximg<len(imglist)-1:
            indeximg +=1

