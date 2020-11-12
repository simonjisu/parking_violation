from .utils import AppState

from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


class D455(object):
    def __init__(self, width, height, framerate, max_dist, sv_path):
        r"""
        Realsense camera pipeline
        """
        self.width = width
        self.height = height 
        self.framerate = framerate
        self.max_dist = max_dist  # filter
        self.sv_path = sv_path
        self.create_filter()

    def create_filter(self):
        self.th_filter = rs.threshold_filter(max_dist=self.max_dist)
        self.sp_filter = rs.spatial_filter()
        self.sp_filter.set_option(rs.option.filter_magnitude, 3.0)
        self.sp_filter.set_option(rs.option.holes_fill, 2.0)
        self.tmp_filter = rs.temporal_filter()

    def run_record(self, state, pipeline):
        r"""
        run the program
        """
        while state.record_btn:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            # filter
            depth_frame = self.th_filter.process(depth_frame)
            depth_frame = self.sp_filter.process(depth_frame)
            depth_frame = self.tmp_filter.process(depth_frame)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # save images

            images = np.hstack((color_image, depth_colormap))
            # Show images
            # cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(state.WIN_NAME, images)
            cv2.setMouseCallback(state.WIN_NAME, state.mouse_btn)
            cv2.waitKey(1)

    def run_app(self):
        state = AppState()
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.depth, 
            self.width, self.height, rs.format.z16, self.framerate)
        config.enable_stream(rs.stream.color, 
            self.width, self.height, rs.format.bgr8, self.framerate)
        # start stream
        pipeline.start(config)
        cv2.namedWindow(state.WIN_NAME, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        # cv2.resizeWindow(state.WIN_NAME, self.width*2, self.height)
        while state.app_btn:
            print()
            # frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame()
            # if not depth_frame or not color_frame:
            #     continue

            # depth_frame = self.th_filter.process(depth_frame)
            # depth_frame = self.sp_filter.process(depth_frame)
            # depth_frame = self.tmp_filter.process(depth_frame)

            # # Convert images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())

            # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # # save images

            # images = np.hstack((color_image, depth_colormap))
            # # Show images
            # cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
            # cv2.imshow(state.WIN_NAME, images)
            # cv2.setMouseCallback(state.WIN_NAME, state.mouse_btn)
            # cv2.waitKey(1)
            if state.record_btn:
                self.run_record(state, pipeline)
            