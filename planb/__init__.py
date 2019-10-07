from enum import Enum
import cv2
import math
import time

from planb.upstream import Upstream
from keras_mask_rcnn import PersonDetector

import logging
logger = logging.getLogger('planb')
logger.setLevel(logging.DEBUG)


def _get_box_distance(box1, box2):
    [startY1, startX1, endY1, endX1] = box1
    [startY2, startX2, endY2, endX2] = box2

    # return distance between the centers of the boxes
    box1_cx = (startX1 + endX1)/2
    box1_cy = (startY1 + endY1)/2
    box2_cx = (startX2 + endX2)/2
    box2_cy = (startY2 + endY2)/2
    dx = box1_cx - box2_cx
    dy = box1_cy - box2_cy
    return math.sqrt(dx*dx + dy*dy)


def _get_box_sq(box):
    # return distance between the centers of the boxes
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    return math.fabs(box_w * box_h)


def _get_extended_face_image(image, box):
    """
        returns the region of image containing face
        taking area somewhat bigger than the bounding box
    """
    h, w = image.shape[0:2]
    [startY, startX, endY, endX] = box
    logger.debug('_get_face_image: image h/w=%d/%d,  startX=%.2f, startY=%.2f, endX=%.2f, endY=%.2f' %
                 (h, w, startX, startY, endX, endY))

    dy1 = (endY - startY)/3  # +33% on top
    dy2 = (endY - startY)/3  # +33% on the botton
    dx = (endX - startX)/4   # +25% on the sides

    startX1 = int((startX - dx)*w)
    startY1 = int((startY - dy1)*h)
    endX1 = int((endX + dx)*w)
    endY1 = int((endY + dy2)*h)

    if startX1 < 0:
        startX1 = 0
    if startY1 < 0:
        startY1 = 0
    if endX1 > w-1:
        endX1 = w-1
    if endY1 > h-1:
        endY1 = h-1
    logger.debug('_get_face_image: startX1=%d, startY1=%d, endX1=%d, endY1=%d' %
                 (startX1, startY1, endX1, endY1))

    # cut image by that enlarged box
    return image[startY1:endY1, startX1:endX1]


def _get_box_width(box):
    [_, startX, _, endX] = box
    return endX - startX

# ========================================================================================================


class Mode(Enum):
    IDLE = 1
    EFFECT_START = 2
    EFFECT_RUN = 3
    EFFECT_ABORT = 4


class PlanB:
    mode = Mode.IDLE
    mode_endtime = None

    # min acceptable score during initial detection
    box_score_threshold_detection = 0.8
    box_score_threshold_tracking = 0.5  # min acceptable score during tracking
    max_distance = 1  # max distance between face on consecutive frames (* face_width)
    max_ndrops = 3  # max number of dropped frames before tracking failure

    T1 = 3
    T2 = 10
    T3 = 3

    def __init__(self, args):
        self.upstream = Upstream(
            'localhost', args.upstream_port, args.upstream_log, args.stream_frames)
        self.target = None
        self.start_time = time.time()
        self.args = args

        if not args.no_face_contour:
            self.pd = PersonDetector()
        else:
            self.pd = None

    # ============================================================================

    import time

    def unrotate(self, box):
        [startY, startX, endY, endX] = box
        #[x1, y1, x2, y2] = box
        #return [1-y2, x2, 1-y1, x1]
        return [1-endX, startY, 1-startX, endY]

    def run(self, image, meta, boxes, scores):
        # process single frame
        # send messages upstream
        if self.mode == Mode.IDLE:
            [i, _] = self.find_best_target(image, meta, boxes, scores)
            if i is None:
                return  # end of processing

            # new face found - extract face contour
            box = boxes[i].tolist()
            face_image = None
            if self.pd is not None:
                # take area slightly larger than the face bounding box and find person there
                face_image = _get_extended_face_image(image, box)
                face_image = self.pd.get_person_image(face_image)
                
                if self.args.debug_face:
                    from matplotlib import pyplot as plt
                    plt.imshow(face_image), plt.show()

            # send upstream events, in the original system of coordinates
            if self.args.rotate:
                box = self.unrotate(box)
            self.upstream.send_effect_start(
                image, meta, box, [self.T1, self.T2, self.T3])

            if self.pd is not None:
                if face_image is None:
                    print("person not detected")
                else:
                    self.upstream.send_face(face_image)

            self.mode = Mode.EFFECT_START
            self.mode_endtime = time.time() + self.T1
            return  # end of processing

        if self.mode == Mode.EFFECT_START:
            if time.time() < self.mode_endtime:
                [i, _] = self.find_locked_target(image, meta, boxes, scores)
                if i is None:
                    self.mode = Mode.EFFECT_ABORT
                    self.mode_endtime = time.time() + self.T3
                    self.upstream.send_effect_abort(image, meta, [self.T3])
                    return  # end of processing
                if i < 0:
                    return  # temporary face loss, do nothing

                if self.args.rotate:
                    box = self.unrotate(boxes[i].tolist())
                else:
                    box = boxes[i].tolist()
                self.upstream.send_effect_run(image, meta, box)
                return  # end of processing

            self.mode = Mode.EFFECT_RUN
            self.mode_endtime = time.time() + self.T2
            # fall through

        if self.mode == Mode.EFFECT_RUN:
            if time.time() < self.mode_endtime:
                [i, _] = self.find_locked_target(image, meta, boxes, scores)
                if i is None:
                    self.mode = Mode.EFFECT_ABORT
                    self.mode_endtime = time.time() + self.T3
                    self.upstream.send_effect_abort(image, meta, [self.T3])
                    return  # end of processing
                if i < 0:
                    return  # temporary face loss, do nothing

                if self.args.rotate:
                    box = self.unrotate(boxes[i].tolist())
                else:
                    box = boxes[i].tolist()
                self.upstream.send_effect_run(image, meta, box)
                return  # end of processing

            self.mode = Mode.EFFECT_ABORT
            self.mode_endtime = time.time() + self.T3
            return  # end of processing

        if self.mode == Mode.EFFECT_ABORT:
            if time.time() < self.mode_endtime:
                return  # end of processing

            self.mode = Mode.IDLE
            self.mode_endtime = None
            return  # end of processing

    # ============================================================================

    def find_best_target(self, image, meta, boxes, scores):
        # find the biggest face
        best_box_i = None
        best_box_sq = 0

        for i in range(len(boxes)):
            if scores[i] < self.box_score_threshold_detection:
                # ignore low-certainty faces
                continue

            box = boxes[i]
            # FIXME: only consider targets which are large enough for person detection
            sq = _get_box_sq(box)
            if sq > best_box_sq:
                best_box_i = i
                best_box_sq = sq

        if best_box_i is not None:
            self.ndrops = 0
            self.target = {'box': boxes[best_box_i]}

        return [best_box_i, best_box_sq]

    def find_locked_target(self, image, meta, boxes, scores):
        # find box closest to the target
        closest_box_i = None
        closest_box_distance = float('inf')
        maxd = _get_box_width(self.target['box']) * self.max_distance

        for i in range(len(boxes)):
            if scores[i] < self.box_score_threshold_tracking:
                # ignore low-certainty faces
                continue

            box = boxes[i]
            distance = _get_box_distance(self.target['box'], box)

            if distance > maxd:
                print("%d ignored %f > %f" % (i, distance, maxd))
                continue # ignore faces which are too far from previous position

            #if distance > self.max_distance:
            #    continue  # ignore faces too far from the expected location

            if distance < closest_box_distance:
                print("%d learned %f < %f" % (i, distance, closest_box_distance))
                closest_box_i = i
                closest_box_distance = distance

        if closest_box_i is not None:
            self.ndrops = 0
            self.target = {'box': boxes[closest_box_i]}
        else:
            # face not found
            self.ndrops = self.ndrops + 1
            if self.ndrops < self.max_ndrops:
                # face not found, but only few drops so far
                closest_box_i = -1

        return [closest_box_i, closest_box_distance]

    # ====================================================================================================

    NFRAMES = 100
    nframes = 0
    nframes_time = None
    fps = -1

    def show_info(self):
        self.nframes = self.nframes - 1
        if self.nframes <= 0:
            now = time.time()
            if self.nframes_time is not None:
                self.fps = self.NFRAMES / (now - self.nframes_time)
            self.nframes_time = now
            self.nframes = self.NFRAMES

        print('fps=%5.2f mode=%12s' %
              (self.fps, self.mode), end=' ')
        if self.mode_endtime is not None:
            print(' mode_remains=%.2f' %
                  (self.mode_endtime - time.time()), end='')
        if self.mode == Mode.EFFECT_START or self.mode == Mode.EFFECT_RUN:
            print(' ndrops=%d' % (self.ndrops), end='')
        print()
