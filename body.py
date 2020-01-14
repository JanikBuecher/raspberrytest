#!/usr/bin/env python

'''
example to detect upright people in images using HOG features
Usage:
    peopledetect.py <image_names>
Press any key to continue, ESC to stop.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('faces.xml')


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w,
                                               y+h-pad_h), (0, 255, 0), thickness)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def main():
    import sys
    from glob import glob
    import itertools as it

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

   # default = [cv.samples.findFile('basketball2.png')] if len(
    #    sys.argv[1:]) == 0 else []

    vc = cv.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        frame = rescale_frame(frame, percent=30)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        found, _w = hog.detectMultiScale(
            frame, winStride=(16, 16), padding=(32, 32), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        draw_detections(frame, found)
        draw_detections(frame, found_filtered, 3)
        print('%d (%d) found' % (len(found_filtered), len(found)))
        cv.imshow('img', frame)
        rval, frame = vc.read()
        key = cv.waitKey(1)
        if key == 27:  # exit on ESC
            break
    vc.release()
    cv.destroyWindow("img")
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
