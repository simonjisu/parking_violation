import numpy as np
import cv2 

class AppState(object):
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = "RealSense"
        self.pitch, self.yaw = np.radians(-10), np.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.decimate = 1
        self.scale = True
        self.color = True
        self.app_btn = True
        self.record_btn = False

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    # def mouse_controll(self):
    #     r"""
    #     """
    #     with mouse.Events() as events:
    #         for event in events:
    #             try: 
    #                 if (event.button == mouse.Button.right) and (event.pressed == True):
    #                     self.app_btn = not self.app_btn
    #                     print(f"Running App: {self.app_btn}")
    #                     break
    #                 elif event.button == mouse.Button.left and (event.pressed == True):
    #                     self.record_btn = not self.record_btn
    #                     print(f"Recording: {self.record_btn}")
    #                     break
    #                 else:
    #                     pass
    #             except:
    #                 continue

    def mouse_controll(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.app_btn = not self.app_btn
            print(f"Running App: {self.app_btn}")

        if event == cv2.EVENT_RBUTTONDOWN:
            self.record_btn = not self.record_btn
            print(f"Running App: {self.record_btn}")        

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

    def view(self, v):
        """apply view transformation on vector array"""
        return np.dot(v - self.pivot, self.rotation) + self.pivot - self.translation

    def project(self, v, h, w):
        """project 3d vector array to 2d"""
        # h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def pointcloud(self, out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        h, w = out.shape[:2]
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(v[s], h, w)
        else:
            proj = self.project(self.view(verts), h, w)

        if self.scale:
            proj *= 0.5**self.decimate

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]
        return out
