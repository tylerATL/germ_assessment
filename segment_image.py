
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys


def open_and_process_image():


    alpha = .9
    beta = 60
    pic = cv.imread(r".\IMG_0883.JPG")
    denoised = cv.GaussianBlur(pic,ksize=(21,21),sigmaX=0,sigmaY=0)
    brightened = cv.convertScaleAbs(denoised,alpha=alpha,beta=beta)



    hsv = cv.cvtColor(brightened, cv.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([30,80,100])
    upper_green = np.array([60,255,255])


    blur_pic = cv.blur(hsv,(1,1))

    # Threshold the HSV image to get only green colors
    mask = cv.inRange(blur_pic, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(blur_pic, blur_pic, mask=mask)
    res2 = cv.GaussianBlur(res, (15,15), 0)

    # green = np.uint8([[[0,1,0 ]]])
    # hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
    # print( hsv_green )
    #
    fig, ax = plt.subplots()
    im = ax.imshow(res)
    plt.show()
    k = cv.waitKey(0)

    return res2

open_and_process_image()