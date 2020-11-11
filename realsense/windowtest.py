import cv2
import numpy as np

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("L BUTTON DOWN")

    if event == cv2.EVENT_LBUTTONUP:
        print("L BUTTON UP")

    if event == cv2.EVENT_RBUTTONDOWN:
        print("R BUTTON DOWN")

    if event == cv2.EVENT_RBUTTONUP:
        print("R BUTTON UP")

    if event == cv2.EVENT_MBUTTONDOWN:
        print("M BUTTON DOWN")

    if event == cv2.EVENT_MBUTTONUP:
        print("M BUTTON UP")


h, w = (480, 640)
out = np.zeros((h, w, 3), dtype=np.uint8)
while True:
    cv2.namedWindow("a", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("a", w, h)
    cv2.setMouseCallback("a", click)
    cv2.imshow("a", out)
    key = cv2.waitKey(1)