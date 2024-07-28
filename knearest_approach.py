


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.cluster import KMeans

def color_quanitization():
    # https: // github.com / Abdesol / Cartoonify - Image / blob / master / notebook.ipynb
    # https: // scikit - learn.org / 0.19 / modules / generated / sklearn.cluster.KMeans.html
    # img = cv.imread(r"C:\Users\TylerThompson\OneDrive - Gotham Greens Holdings, LLC\Desktop\Image (22).jpg")
    img = cv.imread(r"C:\Users\TylerThompson\Downloads\Image (70).jfif")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # edge mask generation
    line_size = 33
    blur_value = 15

    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray_blur = cv.medianBlur(gray_img, blur_value)
    edges = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, line_size, blur_value)

    k = 2
    
    data = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    img_reduced = kmeans.cluster_centers_[kmeans.labels_]
    img_reduced = img_reduced.reshape(img.shape)


    img_reduced = img_reduced.astype(np.uint8)

    # something, binary = cv.threshold(img_reduced, 160,255, cv.THRESH_BINARY)
    # something_else, binary_two = cv.threshold(binary, 250,255, cv.THRESH_BINARY)
    #
    # res2 = cv.blur(binary_two, (3,3))
    # contours, rank = cv.findContours(res2[:,:,1],cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img_reduced, contours, -1, (0, 255, 0), 2)
    # print(img_reduced)

    fig, ax = plt.subplots()
    im = ax.imshow(img_reduced)
    plt.show()
    k = cv.waitKey(0)

    return

color_quanitization()