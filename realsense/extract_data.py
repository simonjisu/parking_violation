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
        
        self.__state = None
        self.__pipeline = None
        self.__config = None
        self.create_filter()
#        self.create_stream()

    @property
    def state(self):
        return self.__state
    
    @state.setter
    def state(self, state):
        self.__state = state

    @property
    def pipeline(self):
        return self.__pipeline
    
    @pipeline.setter
    def pipeline(self, pipeline):
        self.__pipeline = pipeline

    @property
    def config(self):
        return self.__config
    
    @config.setter
    def config(self, config):
        self.__config = config
    
    def create_filter(self):
        self.th_filter = rs.threshold_filter(max_dist=self.max_dist)
        self.sp_filter = rs.spatial_filter()
        self.sp_filter.set_option(rs.option.filter_magnitude, 3.0)
        self.sp_filter.set_option(rs.option.holes_fill, 2.0)
        self.tmp_filter = rs.temporal_filter()

    def create_stream(self):
        r"""
        `self.config.enable_stream` function args
        - stream type: depth, color
        - width
        - height
        - format
        - framerate
        """
        self.config.enable_stream(rs.stream.depth, 
            self.width, self.height, rs.format.z16, self.framerate)
        self.config.enable_stream(rs.stream.color, 
            self.width, self.height, rs.format.bgr8, self.framerate)

    def run(self):
        r"""
        run the program
        """
        while self.state.start_app_btn:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # save images

def main():
    width, height, framerate =(640, 480, 30) 
    max_dist = 2.5
    sv_path = Path().absolute().parent.parent / "saved"
    if not sv_path.exists():
        sv_path.mkdir()

    cm = D455(width, height, framerate, max_dist, sv_path)
    cm.state = AppState()
    cm.pipeline = rs.pipeline()
    cm.config = rs.config()
    
    cv2.namedWindow(cm.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(cm.state.WIN_NAME, width, height)
    cv2.setMouseCallback(cm.state.WIN_NAME, cm.state.mouse_btn)

    cm.create_stream()
    # start stream
    cm.pipeline.start(cm.config)
    while cm.state.start_app_btn:
        cm.run()

if __name__ == "__main__":
    main()