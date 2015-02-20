# Student:     Kasper van Veen & Wessel Huising
# Stdnr:       6139752 & 10011277
# Course:      Beeldbewerken
# Date:        17/02/15
# Assignment:  2.1: Image Warping

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt


def histogramEqualization(f, m, bins=100):
    his, be = np.histogram(f, bins=bins, range=(0, m))
    his = his.astype(float)/sum(his)
    return np.interp(f, be[1:], np.cumsum(his))

if __name__ == "__main__":
    image1 = misc.imread('image1.jpeg')
    image2 = misc.imread('image2.jpeg')
    image3 = misc.imread('image3.jpeg')

    img1 = plt.subplot(2, 3, 1)
    img1.imshow(image1)

    img2 = plt.subplot(2, 3, 2)
    img2.imshow(image2)

    img3 = plt.subplot(2, 3, 3)
    img3.imshow(image3)

    new_img1 = plt.subplot(2, 3, 4)
    new_img1_hist = histogramEqualization(image1, 255, 100)
    new_img1.imshow(new_img1_hist)

    new_img2 = plt.subplot(2, 3, 5)
    new_img2_hist = histogramEqualization(image2, 255, 100)
    new_img2.imshow(new_img2_hist)

    new_img3 = plt.subplot(2, 3, 6)
    new_img3_hist = histogramEqualization(image3, 255, 100)
    new_img3.imshow(new_img3_hist)

    plt.show()
