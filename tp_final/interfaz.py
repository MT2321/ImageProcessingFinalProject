import numpy as np
import cv2
import cvui
from callbacks.callbacks import get_external_filepath
import matplotlib.pyplot as plt

WINDOW_NAME = 'CVUI Hello World!'

# Create a frame where components will be rendered to.
frame = np.zeros((600, 600, 3), np.uint8)

# Init cvui and tell it to create a OpenCV window, i.e. cv2.namedWindow(WINDOW_NAME).
cvui.init(WINDOW_NAME)
DEBUG = True
while True:
    # Fill the frame with a nice color
    frame[:] = (49, 52, 49)

    if(cvui.button(frame, 0, 0, 100, 20, "Load File")):
        if not DEBUG:
            image_path = get_external_filepath()
        else:
            image_path = "/assets/img/betsson.jpeg"
    # Update cvui stuff and show everything on the screen
    cvui.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(20) == 27:
        break
