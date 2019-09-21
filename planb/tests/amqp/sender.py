#!/usr/bin/env python
import pika
import base64
import pickle

# https://www.rabbitmq.com/tutorials/tutorial-one-python.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('test.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/3)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/3)
 
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret != True:
        break
    frame = cv2.resize(frame, (w, h))  # resize the frame        
    #cv2.imshow('Frame', frame)

    #_, buffer = cv2.imencode('.jpg', frame)
    #jpg_as_text = base64.b64encode(buffer)
    data = pickle.dumps(frame)
    text = base64.b64encode(data)

    channel.basic_publish(exchange='', routing_key='hello', body=text)
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

connection.close()
