import socket
import struct
import json
import base64
import cv2
import time
import logging

logger = logging.getLogger('upstream')
logger.setLevel(logging.INFO)

RETRY_PERIOD = 3


class Upstream:
    last_retry_time = None

    def __init__(self, host, port, file, stream_frames):
        logger.info('INIT: host=%s, port=%d, file=%s' % (host, port, file))

        self.stream_frames = stream_frames
        self.host = host
        self.port = port
        self.connect_upstream()

        self.open_upstream_logfile(file)

    def connect_upstream(self):
        if self.port <= 0:
            return

        now = time.time()
        if self.last_retry_time is None or (now - self.last_retry_time >= RETRY_PERIOD):
            try:
                self.clientsocket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.clientsocket.connect((self.host, self.port))
                self.last_retry_time = None
                logger.info('Connected to upstream server %s:%d' %
                            (self.host, self.port))
            except Exception as ex:
                logger.error('Failed to connect upstream: %s' % ex)
                self.clientsocket = None
                self.last_retry_time = now

    def open_upstream_logfile(self, file):
        if file is not None:
            self.fileh = open(file, 'wb')
            logger.info('Opened upstream file %s' % file)
        else:
            self.fileh = None

    def send_effect_start(self, image, meta, box, T):
        self.send_event(
            image, meta, {'mode': 'effect_start', 'box': box, 'time': T})

    def send_effect_run(self, image, meta, box):
        self.send_event(image, meta, {'mode': 'effect_run', 'box': box})

    def send_effect_abort(self, image, meta, T):
        self.send_event(image, meta, {'mode': 'effect_abort', 'time': T})

    def send_face(self, face_image):
        _, face_imdata = cv2.imencode('.png', face_image)
        encoded_face_imdata = base64.b64encode(face_imdata).decode('ascii')
        self.send_event(None, None, {'face': encoded_face_imdata})

    def send_event(self, image, meta, event):
        logger.debug('send_event(%s)' % event)

        if self.clientsocket is None:
            self.connect_upstream()

        if self.clientsocket is not None or self.fileh is not None:
            if self.stream_frames:
                # add encoded frame to the event structure
                image = cv2.resize(image, (640, 480))  # FIXME
                _, imdata = cv2.imencode('.png', image)
                encoded_imdata = base64.b64encode(imdata).decode('ascii')
                event['frame'] = encoded_imdata

            # if self.stream_timestamps:
            #   events['t'] = ...

            msg = (json.dumps(event) + '\r\n').encode('ascii')

            # send it upstream
            if self.clientsocket is not None:
                try:
                    self.clientsocket.sendall(msg)
                except Exception as ex:
                    logger.error("Sending message upstream failed: %s" % ex)
                    self.clientsocket = None

            if self.fileh is not None:
                self.fileh.write(msg)
                self.fileh.flush()

    def close(self):
        if self.clientsocket is not None:
            self.clientsocket.close()
            self.clientsocket = None
        if self.fileh is not None:
            self.fileh.close()
            self.fileh = None
