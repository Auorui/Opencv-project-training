"""
author : Auorui(夏天是冰红茶)
time : 2023-6-3
function:It is used for data acquisition. Place the round sticks
        in the red box, and perform Gaussian filter processing
        outside the box. Adjust the brightness and size of the
        box adaptively through the track bar. Press the keyboard
        "s" to take photos of the round sticks. The number is unlimited.
"""
import cv2
import tally as ta

Vcap = ta.VideoCap()
Vcap.CapInit(mode=0, w=640, h=480)
count = 1

cv2.namedWindow("photodata")
cv2.createTrackbar("bbox_scale", "photodata", 100, 900, ta.onTrackbarChange)
cv2.createTrackbar("brightness_factor", "photodata", 50, 100, ta.onTrackbarChange)

while True:
    img = Vcap.read()
    img = cv2.flip(img,1)
    bbox_scale = cv2.getTrackbarPos("bbox_scale", "photodata") / 1000.0 + 0.1
    brightness_factor = cv2.getTrackbarPos("brightness_factor", "photodata") / 100.0 + 0.5

    position, img_with_box = ta.Bbox_img(img, bbox_scale=bbox_scale)
    img_with_blur = ta.maskBbox(img, position)

    img_with_blur_and_light = ta.Adjusted_image(img_with_blur, brightness_factor=brightness_factor)

    imgStacked = ta.stackImages(1, [img_with_box,img_with_blur_and_light])
    cv2.imshow("photodata", imgStacked)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        filename = "./photodata/image{:03d}.png".format(count)
        cv2.imwrite(filename, img_with_blur_and_light)
        print("image{:03d}.png保存成功！".format(count))
        count += 1
    elif k == 27:
        break







