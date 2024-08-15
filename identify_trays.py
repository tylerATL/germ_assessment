
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys


def identify_full_trays():
    alpha = 1.5
    beta = 60
    pic = cv.imread(r".\IMG_0883.JPG")
    pic_brighter = cv.convertScaleAbs(pic,alpha=alpha,beta=beta)
    hls = cv.cvtColor(pic_brighter, cv.COLOR_BGR2HSV)

    # # define range of green color in HSV
    # lower_white = np.array([20,0,225])
    # upper_white = np.array([255,10,255])
    # lower_gray = np.array([55, 255, 2])
    # upper_gray = np.array([70, 255, 2])
    lower_green = np.array([30,80,100])
    upper_green = np.array([60,255,255])


    # blur_pic = cv.blur(hsv,(15,15))

    # Threshold the HSV image to get only green colors
    mask = cv.inRange(pic_brighter, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = ~cv.bitwise_and(hls, hls, mask=~mask)

    something, binary = cv.threshold(res, 30,255, cv.THRESH_TOZERO_INV)
    res2 = cv.blur(binary, (30,30))
    contours, rank = cv.findContours(res2[:,:,2],cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(pic, contours, -1, (0, 255, 0), 3)
    print(contours)
    # green = np.uint8([[[255,255,255 ]]])
    # hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
    # print( hsv_green )
    #
    fig, ax = plt.subplots()
    ax.imshow(res)
    plt.show()
    k = cv.waitKey(0)

    return res2

identify_full_trays()