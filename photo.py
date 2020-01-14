import numpy as np
import cv2
from matplotlib import pyplot as plt

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w,
                                                y+h-pad_h), (0, 255, 0), thickness)


img = cv2.imread(
    'Janik_Bild.jpeg', cv2.IMREAD_COLOR)


face_cascade = cv2.CascadeClassifier(
    'faces.xml')
low_cascade = cv2.CascadeClassifier(
    'body.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

found, w = hog.detectMultiScale(
    img, winStride=(8, 8), padding=(32, 32), scale=1.05)
draw_detections(img, found)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

img = rescale_frame(img, percent=20)
cv2.imshow('image', img)
# If you don'tput this line,thenthe image windowis just a flash. If you put any number other than 0, the same happens.
cv2.waitKey(0)
cv2.destroyAllWindows()
