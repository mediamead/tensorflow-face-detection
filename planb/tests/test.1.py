#!/usr/bin/python

# Standard imports
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    #cap.open(1 + cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    s, im = cap.read() # captures image
    cv2.imshow("Test Picture", im) # displays captured image
    cv2.imwrite("test.bmp",im) # writes image test.bmp to disk
