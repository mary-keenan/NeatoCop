#!/usr/bin/env python
# import rospy
import numpy
import cv2 as cv

# img = cv.imread('standing.jpeg')
# img = cv.imread('walkers.jpg')
# img = cv.imread('walker.jpg')
img = cv.imread('runner.jpg')

#-------------------------------------------------
# both work poorly when people are sideways, but the second one works better


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')
bodies = body_cascade.detectMultiScale(gray)

# hog = cv.HOGDescriptor()
# hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
# (bodies, weights) = hog.detectMultiScale(img, winStride=(4, 4),
# 		padding=(8, 8), scale=1.05)
#-------------------------------------------------

print (bodies)

for (x,y,w,h) in bodies:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv.imshow('img',img)
cv.waitKey(0)
cv.imwrite('haar_runner.png',img)
cv.destroyAllWindows()