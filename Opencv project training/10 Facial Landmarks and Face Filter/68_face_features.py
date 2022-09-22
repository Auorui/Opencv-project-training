import cv2
import numpy as np
import dlib

img = cv2.imread('1.png')
img = cv2.resize(img, (0, 0), None, 0.80, 0.80)
imgOriginal = img.copy()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
        cv2.circle(imgOriginal, (x, y), 5, (50,50,255),cv2.FILLED)
        cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)
    print(myPoints)

cv2.imshow("Originial", imgOriginal)
cv2.waitKey(0)





























