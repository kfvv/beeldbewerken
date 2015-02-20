# Student:     Kasper van Veen & Wessel Huising
# Stdnr:       6139752 & 10011277
# Course:      Beeldbewerken
# Date:        17/02/15
# Assignment:  2.2: Histogram Equalization

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt


def histogramEqualization(f, m, bins=100):
    his, be = np.histogram(f, bins=bins, range=(0, m))
    his = his.astype(float)/sum(his)
    return np.interp(f, be[1:], np.cumsum(his))

if __name__ == "__main__":
    image = misc.imread('cameraman.jpe')

    orig = plt.subplot(2, 2, 1)
    orig.imshow(image)

    orig_histo = plt.subplot(2, 2, 3)
    orig_histo.hist(image.flatten(), range=(0, 255))

    new_image = plt.subplot(2, 2, 2)
    new_image_hist = histogramEqualization(image, 255, 100)
    new_image.imshow(new_image_hist)

    histo = plt.subplot(2, 2, 4)
    histo.hist(new_image_hist.reshape(-1))

    plt.show()
