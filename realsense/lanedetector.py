import cv2
import numpy as np

class LaneDetector(object):
    r"""Lane Detector"""
    def __init__(self):
        pass

    def detect(self, image):
        r"""
        Detect White and Yellow Lane
        """
        # white color mask
        lower = np.uint8([200, 200, 200])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(image, lower, upper)
        # yellow color mask
        lower = np.uint8([190, 190,   0])
        upper = np.uint8([255, 255, 255])
        yellow_mask = cv2.inRange(image, lower, upper)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked = cv2.bitwise_and(image, image, mask = mask)
        return masked