import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

import segment_image

image = segment_image.open_and_process_image()

ret,thresh3 = cv.threshold(image[:,:,:], 150,255, cv.THRESH_TOZERO)

# to_gray = cv.cvtColor(image_value_channel,cv.COLOR_HSV2BGR)
edges = cv.Canny(thresh3,100,100)

fig, ax = plt.subplots()
im = ax.imshow(edges)
plt.show()
k = cv.waitKey(0)
