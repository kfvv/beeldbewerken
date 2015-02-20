# Student:     Kasper van Veen & Wessel Huising
# Stdnr:       6139752 & 10011277
# Course:      Beeldbewerken
# Date:        17/02/15
# Assignment:  2.1: Image Warping

from pylab import svd
import numpy as np
import cv2


def affineTransform(image, x1, y1, x2, y2, x3, y3, M, N):
    matM = np.array([[x1, y1, 1, 0, 0, 0],
                     [0, 0, 0, x1, y1, 1],
                     [x2, y2, 1, 0, 0, 0],
                     [0, 0, 0, x2, y2, 1],
                     [x3, y3, 1, 0, 0, 0],
                     [0, 0, 0, x3, y3, 1]])

    # x1, y1 at origin (0,0), x2, y2 at N-1,0 and x3, y3 at 0, M-1
    q = np.array([[0], [0], [(N - 1)], [0], [0], [(M - 1)]])

    # leastsquares gives a tuple, only need first.
    p = np.linalg.lstsq(matM, q)[0]
    p = p.reshape(2, 3)
    print(p)

    new_image = cv2.warpAffine(image, p, (M, N))

    return new_image


def perspectiveTransform(image, x1, y1, x2, y2, x3, y3, x4, y4, M, N):
    # from tl -> tr -> bl -> br
    nx1 = 0
    ny1 = 0
    nx2 = M
    ny2 = 0
    nx3 = 0
    ny3 = N
    nx4 = M
    ny4 = N

    # given matrix from exercise
    matM = np.array([[x1, y1, 1, 0,  0,  0, -nx1 * x1, -nx1 * y1, -nx1],
                     [0,  0,  0, x1, y1, 1, -ny1 * x1, -ny1 * y1, -ny1],
                     [x2, y2, 1, 0,  0,  0, -nx2 * x2, -nx2 * y2, -nx2],
                     [0,  0,  0, x2, y2, 1, -ny2 * x2, -ny2 * y2, -ny2],
                     [x3, y3, 1, 0,  0,  0, -nx3 * x3, -nx3 * y3, -nx3],
                     [0,  0,  0, x3, y3, 1, -ny3 * x3, -ny3 * y3, -ny3],
                     [x4, y4, 1, 0,  0,  0, -nx4 * x4, -nx4 * y4, -nx4],
                     [0,  0,  0, x4, y4, 1, -ny4 * x4, -ny4 * y4, -ny4]])

    # take the singular value decomposition of matrix M
    U, D, V = svd(matM)

    p = V[-1]

    # reshape p to a 3 x 3 matrix
    p = p.reshape(3, 3)
    print(p)

    new_image = cv2.warpPerspective(image, p, (M, N))

    return new_image

if __name__ == "__main__":

    image = cv2.imread('cameraman.jpg')

    M, N = image.shape[:2]

    new_image = affineTransform(image, 135, 20, 275, 140, 15, 140, M, N)

    cv2.imshow("original", image)
    cv2.imshow("transformed", new_image)
    """
    image = cv2.imread('flyeronground.png')

    M, N = image.shape[:2]

    new_image = perspectiveTransform(image, 569, 187, 825, 178, 350, 555,
                                     598, 588, 400, 600)
    cv2.imshow("original", image)
    cv2.imshow("perspectiveTransformed", new_image)
    """
    cv2.waitKey(0)
