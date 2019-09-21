#!/usr/bin/env python

import pika
import cv2
import base64
import pickle

# https://www.rabbitmq.com/tutorials/tutorial-one-python.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_delete(queue='hello')
channel.queue_declare(queue='hello')

def callback(ch, method, properties, text):
    #print(frame_enc)
    #buffer = base64.b64decode(jpg_as_text)
    #npimg = np.fromstring(buffer, dtype=np.uint8)
    #frame = cv2.imdecode(npimg, 1)

    frame_data = base64.b64decode(text)
    frame = pickle.loads(frame_data)
    
    cv2.imshow('Frame', frame)

channel.basic_consume(
    queue='hello', on_message_callback=callback, auto_ack=True)

channel.start_consuming()

# Closes all the frames
cv2.destroyAllWindows()