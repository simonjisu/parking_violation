import cv2
import screeninfo
import numpy as np
import pyrealsense2 as rs
from .utils import AppState
from .lanedetector import LaneDetector
from datetime import datetime as dt


class D455(object):
    def __init__(self, width, height, framerate, max_dist, sv_path, 
        record_time, saveimg, savepc, savebag):
        r"""
        Realsense camera pipeline
        """
        self.width = width
        self.height = height
        self.framerate = framerate
        self.max_dist = max_dist  # filter
        self.sv_path = sv_path
        self.record_time = record_time
        self.saveimg = saveimg
        self.savepc = savepc
        self.savebag = savebag

        self.lane_detector = LaneDetector()
        self.pc = rs.pointcloud()
        self.colorizer = rs.colorizer()
        self.create_filter()
        

    def create_filter(self):
        self.th_filter = rs.threshold_filter(max_dist=self.max_dist)
        self.sp_filter = rs.spatial_filter()
        self.sp_filter.set_option(rs.option.filter_magnitude, 3.0)
        self.sp_filter.set_option(rs.option.holes_fill, 2.0)
        self.tmp_filter = rs.temporal_filter()

    def run_record(self, state, pipeline, record_path):
        r"""
        run the program
        """
        e1 = cv2.getTickCount()
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

            depth_colormap = self.colorizer.colorize(depth_frame)
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_colormap.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            lane_masked = self.lane_detector.detect(color_image)
            # Show images
            stacked_imgs = (color_image, depth_image, lane_masked)
            images = np.hstack(stacked_imgs)
            cv2.resizeWindow(state.WIN_NAME, 
                self.width*len(stacked_imgs), 
                self.height)
            cv2.imshow(state.WIN_NAME, images)
            cv2.setMouseCallback(state.WIN_NAME, state.mouse_controll)
            key = cv2.waitKey(1)
            if key == 27:
                state.app_btn = False
                state.record_btn = False
                break
            # Calculate Runtime Tick to quit
            e2 = cv2.getTickCount()
            tick = int((e2 - e1) / cv2.getTickFrequency())
            # Save images per tick
            if self.saveimg:
                color_file = record_path / f"color-{tick}.npy"
                depth_file = record_path / f"depth-{tick}.npy"
                ps_file = record_path / f"ps-{tick}.ply"
                if not ps_file.exists():
                    np.save(color_file, color_image)
                    np.save(depth_file, depth_image)
                    
                # Create point cloud
                if self.savepc and (not ps_file.exists()):
                    points = self.pc.calculate(depth_frame)
                    self.pc.map_to(depth_frame)
                    points.export_to_ply(str(ps_file), color_frame)

            if tick > self.record_time:
                print("Finish Record")
                state.app_btn = False
                state.record_btn = False
                cv2.destroyAllWindows()
                break

            if not state.app_btn:
                break

    def run_app(self):
        state = AppState()
        pipeline = rs.pipeline()
        config = rs.config()
        screen = screeninfo.get_monitors()[0]

        while state.app_btn:
            # Make window full screen to make sure start with mouse click
            cv2.namedWindow(state.WIN_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(state.WIN_NAME, screen.x - 1, screen.y - 1)
            cv2.setWindowProperty(state.WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(state.WIN_NAME, state.mouse_controll)
            key = cv2.waitKey(1)
            if key == 27:
                state.app_btn = False
                break

            if state.record_btn:
                
                folder = dt.now().strftime("record_%Y-%m-%d-%H-%M-%S")
                record_path = self.sv_path / folder
                if not record_path.exists():
                    record_path.mkdir()
                # Config
                config.enable_stream(rs.stream.depth, 
                    self.width, self.height, rs.format.z16, self.framerate)
                config.enable_stream(rs.stream.color, 
                    self.width, self.height, rs.format.bgr8, self.framerate)
                if self.savebag:
                    config.enable_record_to_file(str(record_path / "bagrecord.bag"))
                pipeline.start(config)
                self.run_record(state, pipeline, record_path)

        if not state.app_btn:
            print("Finish App")
            