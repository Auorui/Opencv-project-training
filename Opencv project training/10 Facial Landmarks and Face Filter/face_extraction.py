import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def createBox(img, points, scale=3):
    bbox = cv2.boundingRect(points)
    x, y, w, h = bbox
    imgCrop = img[y:y + h, x:x + w]
    imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
    return imgCrop

img = cv2.imread('1.png')
img = cv2.resize(img, (0, 0), None, 0.80, 0.80)
imgOriginal = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgOriginal)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    imgOriginal=cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmarks = predictor(imgGray, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])
        # cv2.circle(imgOriginal, (x, y), 5, (50,50,255),cv2.FILLED)
        # cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)
    myPoints = np.array(myPoints)
    imgEyeBrowLeft = createBox(img, myPoints[17:22])
    imgEyeBrowRight = createBox(img, myPoints[22:27])
    imgNose = createBox(img, myPoints[27:36])
    imgLeftEye = createBox(img, myPoints[36:42])
    imgRightEye = createBox(img, myPoints[42:48])
    imgLips = createBox(img, myPoints[48:61])
    cv2.imshow('Left Eyebrow', imgEyeBrowLeft)
    cv2.imshow('Right Eyebrow', imgEyeBrowRight)
    cv2.imshow('Nose', imgNose)
    cv2.imshow('Left Eye', imgLeftEye)
    cv2.imshow('Right Eye', imgRightEye)
    cv2.imshow('Lips', imgLips)

cv2.imshow("Originial", imgOriginal)
cv2.waitKey(0)

