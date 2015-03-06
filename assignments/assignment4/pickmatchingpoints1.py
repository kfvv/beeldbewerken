# -*- coding: utf-8 -*-
"""
Pick matching points in two images
"""
from pylab import *
import cv2
from ginput import ginput
import SIFT
import keypoints


def marker(im, x, y, r, s):
    cv2.circle(im, (x, y), r, (0, 255, 255))
    cv2.line(im, (x, y - r), (x, y + r), (0, 255, 255))
    cv2.line(im, (x - r, y), (x + r, y), (0, 255, 255))
    cv2.putText(im, s, (x, y + r), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255))


#def pickMatchingPoints(im1, im2, n):
#    dispim1 = im1.copy()
#    dispim2 = im2.copy()
#
#    cv2.imshow('Image 1', dispim1)
#    cv2.imshow('Image 2', dispim2)
#
#    xy = zeros((4, 2))
#    for i in range(n):
#        print('Click at point %s in image 1' % (i + 1))
#        x, y = ginput('Image 1', 1)
#        marker(dispim1, x, y, 3, str(i + 1))
#        cv2.imshow('Image 1', dispim1)
#        xy[i, 0] = x
#        xy[i, 1] = y
#
#    xaya = zeros((4, 2))
#    for i in range(n):
#        print('Click at point %s in image 2' % (i + 1))
#        x, y = ginput('Image 2', 1)
#        marker(dispim2, x, y, 3, str(i + 1))
#        cv2.imshow('Image 2', dispim2)
#        xaya[i, 0] = x
#        xaya[i, 1] = y
#
## the are points i have clicked that lead to reasonable result
##    xy = array([[157,  32],
##                [211,  37],
##                [222, 107],
##                [147, 124]])
##    xaya = array([[6,  38],
##                  [56, 31],
##                  [82, 87],
##                  [22, 118]])
##
#    return xy, xaya

def pickMatchingPoints(im1, im2, n):
    pass


def getPerspectiveTransform(xy, xaya):
    matM = zeros((len(xy) * 2, 9))
    for i in xrange(len(xy)):
        matM[i * 2]     = [xy[i, 0], xy[i, 1], 1, 0, 0, 0, -xaya[i,0] * xy[i,0], -xaya[i,0] * xy[i,1], -xaya[i,0]]
        matM[i * 2 + 1] = [0, 0, 0, xy[i, 0], xy[i, 1], 1, -xaya[i,1] * xy[i,0], -xaya[i,1] * xy[i,1], -xaya[i,1]]

    U, D, V = svd(matM)

    p = V[-1]

    # reshape p to a 3 x 3 matrix
    p = p.reshape(3, 3)

    return p


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits
     well to data
    """ 

    # iterations = 0
    # bestfit = None
    # besterr = numpy.inf
    # best_inlier_idxs = None
    # while iterations < k:
    #     maybe_idxs, test_idxs = random_partition(n, data.shape[0])
    #     maybeinliers = data[maybe_idxs, :]
    #     test_points = data[test_idxs]
    #     maybemodel = model.fit(maybeinliers)
    #     test_err = model.get_error(test_points, maybemodel)
    #     # select indices of rows with accepted points
    #     also_idxs = test_idxs[test_err < t]
    #     alsoinliers = data[also_idxs, :]
    #     if len(alsoinliers) > d:
    #         betterdata = numpy.concatenate((maybeinliers, alsoinliers))
    #         bettermodel = model.fit(betterdata)
    #         better_errs = model.get_error(betterdata, bettermodel)
    #         thiserr = numpy.mean(better_errs)
    #         if thiserr < besterr:
    #             bestfit = bettermodel
    #             besterr = thiserr
    #             best_inlier_idxs = numpy.concatenate((maybe_idxs, also_idxs))
    #     iterations += 1
    # if bestfit is None:
    #     raise ValueError("did not meet fit acceptance criteria")
    # if return_all:
    #     return bestfit, {'inliers': best_inlier_idxs}
    # else:
    #     return bestfit


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange(n_data)
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


if __name__ == "__main__":
    im1 = cv2.imread('images/nachtwacht1.jpg')
    im2 = cv2.imread('images/nachtwacht2.jpg')
    # xy, xaya = pickMatchingPoints(im1, im2, 4)

    # P = cv2.getPerspectiveTransform(xy.astype(float32), xaya.astype(float32))
    #P = getPerspectiveTransform(xy, xaya)

    sift = SIFT.SIFT()

    kpts1, descrs1 = sift.detectAndCompute(im1)
    kpts2, descrs2 = sift.detectAndCompute(im2)
    """
    xy is array with the points in first image, xaya the corresponding points
    in the second image, kp_matches is an array of tuples, each tuple contains
    matching keypoints (see documentation for the attributes of the class)
    """
    xy, xaya, kp_matches = SIFT.matchDescriptors(kpts1, descrs1, kpts2, descrs2)
    result = keypoints.drawMatches(im1, xy, im2, xaya, kp_matches)
    cv2.imshow('matches', result)
    # I warp from image2 to image1 because image 1 is nicely upright
    # The sizez of images are set by trial and error...
    # The final code for the exercise should not contain these magic numbers
    # the size of the final image should be calculated to exactly contain
    # both (warped) images.
    # In the warped version of image2 i simply overwrite with data
    # from image 1.
    #tim = cv2.warpPerspective(im2, linalg.inv(P), (450, 300))
    #M, N = im1.shape[:2]
    #tim[0:M, 0:N, :] = im1
    #cv2.waitKey(1)
    #cv2.imshow('result', tim)

    cv2.waitKey(0)
