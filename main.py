#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import profiler
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import logging

import planb

from utils import label_map_util
from utils import visualization_utils_color as vis_util

logging.basicConfig()
logging.basicConfig(level=logging.ERROR, format='%(message)s')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.25

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')
        # Actual detection.
        #start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        #print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rotate", action='store_true',
                        help="rotate clockwise for NN processing")
    parser.add_argument("-f", "--stream_frames",
                        action='store_true', help="send encoded frames upstream")
    parser.add_argument("-n", "--no_visual", action='store_true',
                        default=False, help="do not visualize video stream")
    parser.add_argument("-p", "--upstream-port", type=int,
                        default=8089, help="upstream port")
    parser.add_argument("-u", "--upstream-log", help="upstream logfile")
    parser.add_argument("-P", "--profiling",
                        action='store_true', help="create profiler.csv")
    parser.add_argument("-m", "--mirror",
                        action='store_true', help="flip the image horizontally to mimic a mirror")
    parser.add_argument("-c", "--camera", type=int,
                        default=0, help="camera id")
    parser.add_argument("-s", "--skip", type=int,
                        default=2, help="skip so many frames")
    parser.add_argument("-F", "--no-face-contour", action='store_true',
                        default=False, help="do not send face contour upstream")
    parser.add_argument("--debug-face", action='store_true',
                        default=False, help="show face as sent upstream (halts execution)")

    args = parser.parse_args()

    pb = planb.PlanB(args)

    if args.profiling:
        profiler = profiler.Profiler()
    else:
        profiler = None

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(args.camera)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off

    windowName = None
    visualize = not args.no_visual

    nframe = 0
    while True:
        t = [time.time(), 0, 0, 0, 0, 0]

        ret, image = cap.read()
        if ret == 0:
            break
        t[1] = time.time()

        # skip
        if nframe > 0:
            nframe = nframe - 1
            continue
        else:
            nframe = args.skip

        meta = {}
        if args.mirror:
            image = cv2.flip(image, 1)

        if args.rotate:
            image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)

        t[2] = time.time()

        (boxes, scores, classes, num_detections) = tDetector.run(image)
        t[3] = time.time()

        pb.run(image, meta, boxes[0], scores[0])
        t[4] = time.time()

        pb.show_info(image)

        if visualize:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)

            if windowName is None:
                [h, w] = image.shape[:2]
                windowName = "tensorflow based (%d, %d)" % (w, h)
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

            cv2.imshow(windowName, image)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            print("Exiting...")
            break
        elif k == ord('r'):
            print("Resetting move timing (%d/%d)" %
                  (pb.move_duration, pb.move_period))
            pb.reset_move()
        elif k == ord('v'):
            visualize = not visualize
            print("Visualize = %s" % visualize)

        t[5] = time.time()
        if profiler:
            profiler.report(t)

    if profiler:
        profiler.close()
    pb.upstream.close()
    cap.release()
