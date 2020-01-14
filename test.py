import cv2
import time

face_cascade = cv2.CascadeClassifier('faces.xml')
upperbody_cascade = cv2.CascadeClassifier('body.xml')

vc = cv2.VideoCapture(0)


def make_480p():
    vc.set(3, 1080)
    vc.set(4, 720)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    bodys = upperbody_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in bodys:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    frame = rescale_frame(frame, percent=40)
    cv2.imshow("preview", frame)

    rval, frame = vc.read()
    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break
vc.release()
cv2.destroyWindow("preview")
