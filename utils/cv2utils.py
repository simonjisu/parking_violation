import cv2
from PIL import Image

imreader = lambda path: np.array(Image.open(path))
imsaver = lambda x, path: Image.fromarray(x).save(path)

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
    rotate_code = None
    if meta_dict['streams'][0]['tags'].get('rotate'):
        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
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