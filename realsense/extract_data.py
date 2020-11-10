from datetime import datetime as dt
from datetime import timedelta
import time
import cv2
import numpy as np
import pyrealsense2 as rs

class D455(object):
    def __init__(self, width, height, framerate, runtime, sv_path):
        r"""
        Realsense camera pipeline
        """
        self.width = width
        self.height = height 
        self.framerate = framerate
        self.runtime = runtime
        self.sv_path = sv_path

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.create_stream()
        
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
        self.pipeline.start(self.config)

        start_time = dt.now()
        end_time = start_time + timedelta(seconds=self.runtime)
        state = AppState()
        
        while (dt.now() - start_time).seconds <= self.runtime:

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

        
            # Get stream profile and camera intrinsics
            profile = self.pipeline.get_active_profile()
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            depth_intrinsics = depth_profile.get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            # Processing blocks
            pc = rs.pointcloud()
            decimate = rs.decimation_filter()
            decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
            colorizer = rs.colorizer()



class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = np.radians(-10), np.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


