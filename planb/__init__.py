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
    # return distance between the centers of the boxes
    box1_cx = (box1[0] + box1[2])/2
    box1_cy = (box1[1] + box1[3])/2
    box2_cx = (box2[0] + box2[2])/2
    box2_cy = (box2[1] + box2[3])/2
    dx = box1_cx - box2_cx
    dy = box1_cy - box2_cy
    return math.sqrt(dx*dx + dy*dy)


def _get_box_sq(box):
    # return distance between the centers of the boxes
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    return math.fabs(box_w * box_h)


def _get_face_image(image, box):
    h, w = image.shape[0:2]
    [startY, startX, endY, endX] = box
    logger.debug('_get_face_image: image h/w=%d/%d,  startX=%s, startY=%d, endX=%d, endY=%d' %
                 (h, w, startX, startY, endX, endY))
    # take a box larger by 50% each way, while keeping within ranges

    dy1 = (endY - startY)*3/4  # 75% on top
    dy2 = 0 # (endY - startY)/4    # 25% on the botton
    dx = (endX - startX)/2     # 50% on the sides

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
    logger.debug('_get_face_image: startX1=%s, startY1=%d, endX1=%d, endY1=%d' %
                 (startX1, startY1, endX1, endY1))

    # cut image by that enlarged box
    return image[startY1:endY1, startX1:endX1]


# ========================================================================================================


class Mode(Enum):
    MOVING = 1
    IDLE = 2
    EFFECT_START = 3
    EFFECT_RUN = 4
    EFFECT_ABORT = 5


class PlanB:
    mode = Mode(Mode.IDLE)
    mode_endtime = None

    # min acceptable score during initial detection
    box_score_threshold_detection = 0.8
    box_score_threshold_tracking = 0.5  # min acceptable score during tracking
    max_distance = 100  # max distance between face on consecutive frames
    max_ndrops = 3  # max number of dropped frames before tracking failure

    move_period = 10
    move_duration = 3

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

    def reset_move(self):
        self.start_time = time.time()

    def unrotate(self, box):
        [x1, y1, x2, y2] = box
        return [1-y2, x2, 1-y1, x1]

    def run(self, image, meta, boxes, scores):
        # how many seconds passed since the start of the cycle
        elapsed = (time.time() - self.start_time) % self.move_period
        moving = (elapsed < self.move_duration)

        # process single frame
        # send messages upstream

        if self.mode == Mode.MOVING:
            if moving:
                # still moving
                return  # end of processing
            else:
                self.mode = Mode.IDLE
                # fall through

        if self.mode == Mode.IDLE:
            if moving:
                # started to move
                self.mode = Mode.MOVING
                return  # end of processing

            [i, _] = self.find_best_target(image, meta, boxes, scores)
            if i is None:
                return  # end of processing

            # new face found
            if self.args.rotate:
                box = self.unrotate(boxes[i].tolist())
            else:
                box = boxes[i].tolist()
            self.upstream.send_effect_start(
                image, meta, box, [self.T1, self.T2, self.T3])

            if self.pd is not None:
                # take area slightly larger than the face bounding box and find person there
                face_image = _get_face_image(image, box)
                face_image = self.pd.get_person_image(face_image)
                # cv2.imshow("face", face_image)
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

            if moving:
                self.mode = Mode.MOVING
            else:
                self.mode = Mode.IDLE
            self.mode_endtime = None
            return  # end of processing

    # ============================================================================

    def find_best_target(self, image, meta, boxes, scores):
        # FIXME-6: find the biggest face
        best_box_i = None
        best_box_sq = 0

        for i in range(len(boxes)):
            if scores[i] < self.box_score_threshold_detection:
                # ignore low-certainty faces
                continue

            box = boxes[i]
            sq = _get_box_sq(box)
            if sq > best_box_sq:
                best_box_i = i
                best_box_sq = sq

        if best_box_i is not None:
            self.ndrops = 0
            self.target = {'box': boxes[best_box_i]}

        return [best_box_i, best_box_sq]

    def find_locked_target(self, image, meta, boxes, scores):
        # FIXME-12: find box closest to the target
        closest_box_i = None
        closest_box_distance = float('inf')

        for i in range(len(boxes)):
            if scores[i] < self.box_score_threshold_tracking:
                # ignore low-certainty faces
                continue

            box = boxes[i]
            distance = _get_box_distance(self.target['box'], box)
            if distance > self.max_distance:
                continue  # ignore faces too far from the expected location

            if distance < closest_box_distance:
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

    def show_info(self, frame):
        self.nframes = self.nframes - 1
        if self.nframes <= 0:
            now = time.time()
            if self.nframes_time is not None:
                self.fps = self.NFRAMES / (now - self.nframes_time)
            self.nframes_time = now
            self.nframes = self.NFRAMES

        elapsed = (time.time() - self.start_time) % self.move_period
        print('fps=%5.2f elapsed=%5.2f mode=%12s' %
              (self.fps, elapsed, self.mode), end=' ')
        if self.mode_endtime is not None:
            print(' mode_remains=%.2f' %
                  (self.mode_endtime - time.time()), end='')
        if self.mode == Mode.EFFECT_START or self.mode == Mode.EFFECT_RUN:
            print(' ndrops=%d' % (self.ndrops), end='')
        print()
