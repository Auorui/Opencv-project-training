import cv2
import os

mainFolder = 'Images'
myFolders = os.listdir(mainFolder)  #读取主文件
print(myFolders)

for folder in myFolders:
    path = mainFolder +'/'+folder    #path=Image/1, Image/2
    print(path)
    images =[]
    myList = os.listdir(path)
    print(f'Total number of images detected {len(myList)}')
    for imgN in myList:
        curImg = cv2.imread(f'{path}/{imgN}')
        curImg = cv2.resize(curImg,(0,0),None,0.2,0.2)   #比例减小
        images.append(curImg)

    stitcher = cv2.Stitcher.create()
    (status,result) = stitcher.stitch(images)
    if (status == cv2.STITCHER_OK):
        print('Panorama Generated')
        cv2.imshow(folder,result)
        cv2.waitKey(1)
    else:
        print('Panorama Generation Unsuccessful')

cv2.waitKey(0)