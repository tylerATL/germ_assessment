
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys


def identify_full_trays():
    alpha = .9
    beta = 60
    pic = cv.imread(r"C:\Users\TylerThompson\OneDrive - Gotham Greens Holdings, LLC\Desktop\IMG_0883.JPG")
    pic_brighter = cv.convertScaleAbs(pic,alpha=alpha,beta=beta)
    pic_brighter = cv.blur(pic_brighter,(30,30))
    hsv = cv.cvtColor(pic, cv.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_white = np.array([20,0,225])
    upper_white = np.array([255,10,255])
    lower_gray = np.array([55, 255, 2])
    upper_gray = np.array([70, 255, 2])


    blur_pic = cv.blur(hsv,(15,15))

    # Threshold the HSV image to get only green colors
    mask = cv.inRange(blur_pic, lower_gray, upper_gray)

    # Bitwise-AND mask and original image
    res = ~cv.bitwise_and(blur_pic, blur_pic, mask=~mask)

    something, binary = cv.threshold(res, 30,255, cv.THRESH_TOZERO_INV)
    res2 = cv.blur(binary, (30,30))
    contours, rank = cv.findContours(res2[:,:,2],cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(hsv, contours, -1, (0, 255, 0), 3)
    print(contours)
    # green = np.uint8([[[255,255,255 ]]])
    # hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
    # print( hsv_green )
    #
    fig, ax = plt.subplots()
    ax.imshow(pic_brighter)
    plt.show()
    k = cv.waitKey(0)

    return res2

identify_full_trays()