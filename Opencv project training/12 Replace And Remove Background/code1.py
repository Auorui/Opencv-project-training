import cv2
import cvzone
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap=cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60)  #帧速率
segmentor=SelfiSegmentation()
fpsReader =cvzone.FPS()


listimg=os.listdir("Images")
print(listimg)
imglist = []
for imgpath in listimg:
    img=cv2.imread(f'Images/{imgpath}')
    imglist.append(img)
print(len(imglist))

indeximg = 0

while 1:
    reg,img=cap.read()
    imgout = segmentor.removeBG(img,imglist[indeximg],threshold=0.8)

    imgstacked=cvzone.stackImages([img,imgout],2,1)
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

