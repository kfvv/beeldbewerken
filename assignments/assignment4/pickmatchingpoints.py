# -*- coding: utf-8 -*-
"""
Pick matching points in two images
"""
from pylab import *
import cv2
from ginput import ginput


def marker(im, x, y, r, s):
    cv2.circle(im, (x, y), r, (0, 255, 255))
    cv2.line(im, (x, y - r), (x, y + r), (0, 255, 255))
    cv2.line(im, (x - r, y), (x + r, y), (0, 255, 255))
    cv2.putText(im, s, (x, y + r), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255))


def pickMatchingPoints(im1, im2, n):
    dispim1 = im1.copy()
    dispim2 = im2.copy()

    cv2.imshow('Image 1', dispim1)
    cv2.imshow('Image 2', dispim2)

    xy = zeros((4, 2))
    for i in range(n):
        print('Click at point %s in image 1' % (i + 1))
        x, y = ginput('Image 1', 1)
        marker(dispim1, x, y, 3, str(i + 1))
        cv2.imshow('Image 1', dispim1)
        xy[i, 0] = x
        xy[i, 1] = y

    xaya = zeros((4, 2))
    for i in range(n):
        print('Click at point %s in image 2' % (i + 1))
        x, y = ginput('Image 2', 1)
        marker(dispim2, x, y, 3, str(i + 1))
        cv2.imshow('Image 2', dispim2)
        xaya[i, 0] = x
        xaya[i, 1] = y

    # the are points i have clicked that lead to reasonable result
#    xy = array([[157,  32],
#                [211,  37],
#                [222, 107],
#                [147, 124]])
#    xaya = array([[6,  38],
#                  [56, 31],
#                  [82, 87],
#                  [22, 118]])
#
#    return xy, xaya
#

if __name__ == "__main__":
    im1 = cv2.imread('images/nachtwacht1.jpg')
    im2 = cv2.imread('images/nachtwacht2.jpg')
    xy, xaya = pickMatchingPoints(im1, im2, 4)

    P = cv2.getPerspectiveTransform(xy.astype(float32), xaya.astype(float32))

    # I warp from image2 to image1 because image 1 is nicely upright
    # The sizez of images are set by trial and error...
    # The final code for the exercise should not contain these magic numbers
    # the size of the final image should be calculated to exactly contain
    # both (warped) images.
    # In the warped version of image2 i simply overwrite with data
    # from image 1.
    tim = cv2.warpPerspective(im2, linalg.inv(P), (450, 300))
    M, N = im1.shape[:2]
    tim[0:M, 0:N, :] = im1
    cv2.waitKey(1)
    cv2.imshow('result', tim)

    cv2.waitKey()
