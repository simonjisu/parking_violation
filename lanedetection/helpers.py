import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle

imreader = lambda path: np.array(Image.open(path))
imsaver = lambda x, path: Image.fromarray(x).save(path)

def scale_abs(x, m = 255):
    x = np.absolute(x)
    x = np.uint8(m * x / np.max(x))
    return x 

def roi(gray, mn = 125, mx = 1200):
    m = np.copy(gray) + 1
    m[:, :mn] = 0 
    m[:, mx:] = 0 
    return m 

def show_images(imgs, per_row = 3, per_col = 2, W = 10, H = 5, tdpi = 80):
    fig, ax = plt.subplots(per_col, per_row, figsize = (W, H), dpi = tdpi)
    ax = ax.ravel()

    for i in range(len(imgs)):
        img = imgs[i]
        ax[i].imshow(img)

    for i in range(per_row * per_col):
        ax[i].axis('off')

def show_dotted_image(this_image, points, thickness = 5, color = [255, 0, 255 ], d = 15):
    image = this_image.copy()
    cv2.line(image, points[0], points[1], color, thickness)
    cv2.line(image, points[2], points[3], color, thickness)

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(image)

    for (x, y) in points:
        dot = Circle((x, y), d)
        ax.add_patch(dot)
    plt.show()

def get_points(ratios, width, height):
    r"""
    apply point ratios to specific width & height
    """
    apply_ratio = lambda x, w, h: (int(w*x[0]), int(h*x[1]))
    return [apply_ratio(ratio, w=width, h=height) for ratio in ratios]

def resize_image(img, width, height):
    r"""
    resize image
    """
    return cv2.resize(img, (width, height))

def correct_rgb(frame):
    r"""
    convert BGR format to RGB format
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def check_rotation(video_path):
    r"""
    check if the video is rotated and return the right degree of cv2 object
    """
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(video_path)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotate_code = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotate_code = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotate_code

def correct_rotation(frame, rotate_code):
    r"""
    correct rotation using cv2 object(degree)
    """
    return cv2.rotate(frame, rotate_code)