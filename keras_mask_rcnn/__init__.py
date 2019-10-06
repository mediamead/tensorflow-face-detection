#!/usr/bin/env python

# https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

import cv2
import imutils
import logging

logger = logging.getLogger('keras_mask_rcnn')
logger.setLevel(logging.DEBUG)

LABELS_FILE="keras_mask_rcnn/mask_rcnn_coco.labels"
WEIGHTS_FILE="keras_mask_rcnn/mask_rcnn_coco.h5"
PERSON_LABEL="person"

class PersonDetector(object):
    model = None
    
    def __init__(self):
        self.CLASS_NAMES = open(LABELS_FILE).read().strip().split("\n")

        class SimpleConfig(Config):
                # give the configuration a recognizable name
                NAME = "coco_inference"

                # set the number of GPUs to use along with the number of images
                # per GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1

                # number of classes (we would normally add +1 for the background
                # but the background class is *already* included in the class
                # names)
                NUM_CLASSES = len(self.CLASS_NAMES)

        import os
        config = SimpleConfig()
        self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
        self.model.load_weights(WEIGHTS_FILE, by_name=True)

    def get_person_image(self, image):
        """
            Detects person in the given image,
            makes background transparent,
            returns RGBA image
        """
        # prepare and process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = imutils.resize(image, width=512)
        r = self.model.detect([image])[0]

        # select main character among all detected objects
        selected_i = None
        for i in range(0, r["rois"].shape[0]):
            label = self.CLASS_NAMES[r["class_ids"][i]]
            if label != PERSON_LABEL:
                continue
            
            #if r["scores"][i] > ... FIXME
            selected_i = i

        if selected_i is None:
            return None

        import numpy as np
        # https://stackoverflow.com/questions/32290096/python-opencv-add-alpha-channel-to-rgb-image
        # make non-masked area transparent
        b_channel, g_channel, r_channel = cv2.split(image)
        mask = r["masks"][:, :, selected_i]
        alpha_channel = mask.astype(np.uint8) * 255 # alpha channel: 0=transparent background, 255=non-transparent person
        image = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))

        #(startY, startX, endY, endX) = r["rois"][selected_i]
        #print("%d:%d %d:%d" % (startY, startX, endY, endX))
        #image = image[startY:endY, startX:endX]
        #cv2.rectangle(image, (startX, startY), (endX, endY), (255,0,0), 2)

        #from matplotlib import pyplot as plt
        #plt.imshow(image), plt.colorbar(), plt.show()

        return image

if __name__ == "__main__":
    pd = PersonDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if ret == 0:
            print("VideoCapture failed, exiting")
            break

        person_image = pd.get_person_image(image)
        cv2.imshow("Output", person_image)
        cv2.waitKey()

