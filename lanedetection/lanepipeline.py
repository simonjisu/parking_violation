import pickle
import cv2
import numpy as np
from pathlib import Path
from .curves import Curves
from .birdseye import BirdsEye
from .lanefilter import LaneFilter
from .helpers import roi, get_points, resize_image

class PipeLine:
    def __init__(self, params):
        r"""
        width: width of image
        height: height of image
        cali_path: calibration path (should be preprocessed by chessboard.py)
        src_ratio: points to detect line (with ratio from 0~1) 
        dest_ratio: points to transform (with ratio from 0~1)
        lanefilter: params dictionary for LaneFilter
        curves: params dictionary for Curves
        """
        self.w, self.h = params["width"], params["height"]
        self.calibration_data = pickle.load(open(params["cali_path"], "rb"))
        self.src_ratio = params["src_ratio"]
        self.dest_ratio = params["dest_ratio"]

        matrix = self.calibration_data['camera_matrix']
        distortion_coef = self.calibration_data['distortion_coefficient']
        source_points = get_points(self.src_ratio, self.w, self.h)
        dest_points = get_points(self.dest_ratio, self.w, self.h)
        
        self.birdseye = BirdsEye(source_points, dest_points, matrix, distortion_coef)
        self.lanefilter = LaneFilter(params['lanefilter'])
        self.curves = Curves(params['curves'])
        
    def process(self, frame):
        frame = resize_image(frame, self.w, self.h)
        ground_img = self.birdseye.undistort(frame)
        binary = self.lanefilter.apply(ground_img)
        bird_mask = np.logical_and(self.birdseye.sky_view(binary), roi(binary)).astype(np.uint8)
        bird_ground_img = self.birdseye.sky_view(ground_img)
        bird_img = cv2.bitwise_and(bird_ground_img, bird_ground_img, 
            mask=bird_mask)

        result = self.curves.fit(bird_mask, bird_img)

        ground_img_with_projection = self.birdseye.project(
            ground_img, binary, 
            result['pixel_left_best_fit_curve'], 
            result['pixel_right_best_fit_curve'], 
            result['left_color'][1], 
            result['right_color'][1]
        )
        text_pos = f"vehicle position: {result['vehicle_position_words']}"
        text_l = f"left radius: {np.round(result['left_radius'], 2)} | color: {result['left_color'][0]}"
        text_r = f"right radius: {np.round(result['right_radius'], 2)} | color {result['right_color'][0]}"  
        text_color = (255, 255, 255)
        cv2.putText(ground_img_with_projection, text_l, (20, 40), 
                    cv2.FONT_HERSHEY_PLAIN, 1, text_color, 2)
        cv2.putText(ground_img_with_projection, text_r, (20, 60), 
                    cv2.FONT_HERSHEY_PLAIN, 1, text_color, 2)
        cv2.putText(ground_img_with_projection, text_pos, (20, 80), 
                    cv2.FONT_HERSHEY_PLAIN, 1, text_color, 2)
        
        return ground_img_with_projection