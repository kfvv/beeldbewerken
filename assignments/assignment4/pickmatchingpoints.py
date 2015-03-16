# -*- coding: utf-8 -*-
# Student:     Kasper van Veen & Wessel Huising
# Stdnr:       6139752 & 10011277
# Course:      Beeldbewerken
# Date:        10/03/15
# Assignment:  4: SIFT and RANSAC
"""
Pick matching points in two images
"""
from pylab import *
import cv2
from ginput import ginput
import SIFT
import keypoints
import numpy


def marker(im, x, y, r, s):
    cv2.circle(im, (x, y), r, (0, 255, 255))
    cv2.line(im, (x, y - r), (x, y + r), (0, 255, 255))
    cv2.line(im, (x - r, y), (x + r, y), (0, 255, 255))
    cv2.putText(im, s, (x, y + r), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255))


class Model:
    def fit(self, data):
        """
        Given the data fit the data with your model and return the model
        (a vector)
        """
        xy = data[:, 0:2]
        xaya = data[:, 2:4]

        P = getPerspectiveTransform(xy, xaya)

        return P

    def get_error(self, data, model):
        """
        Given a set of data and a model, what is the error of using this model
        to estimate the data
        """
        xy = data[:, 0:2]
        xy = np.hstack((xy, np.ones((xy.shape[0], 1))))
        xaya = data[:, 2:4]
        xaya = np.hstack((xaya, np.ones((xaya.shape[0], 1))))

  #      xaya_fit = scipy.dot(xy, model)

        err_per_point = np.zeros((len(xy)))
        for i in range(len(xy)):
            err_per_point[i] = ((xaya[i, 0]-((model[0, 0] * xy[i, 0] + model[0, 1] * xy[i, 1] + model[0, 2]) /
                (model[2, 0] * xy[i, 0] + model[2, 1] * xy[i, 1] + model[2, 2]))) ** 2 +
                (xaya[i, 1] - ((model[1, 0] * xy[i, 0] + model[1, 1] * xy[i, 1] + model[1, 2]) /
                    (model[2, 0] * xy[i, 0] + model[2, 1] * xy[i, 1] + model[2, 2]))) ** 2)

        return err_per_point


def getPerspectiveTransform(xy, xaya):
    matM = zeros((len(xy) * 2, 9))

    for i in xrange(len(xy)):
        matM[i * 2]     = [xy[i, 0], xy[i, 1], 1, 0, 0, 0, -xaya[i, 0] * xy[i, 0], -xaya[i, 0] * xy[i,1], -xaya[i,0]]
        matM[i * 2 + 1] = [0, 0, 0, xy[i, 0], xy[i, 1], 1, -xaya[i, 1] * xy[i, 0], -xaya[i, 1] * xy[i, 1], -xaya[i, 1]]

    U, D, V = svd(matM)

    return V[-1].reshape(3,3)


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits
     well to data
    """

    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < t]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = numpy.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


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
    # result = keypoints.drawMatches(im1, xy, im2, xaya, kp_matches)
    # cv2.imshow('matches', result)

    data = np.hstack((xy, xaya))
    ransac_match = ransac(data, Model(), 4, 10, 3, 10)

    # I warp from image2 to image1 because image 1 is nicely upright
    # The sizez of images are set by trial and error...
    # The final code for the exercise should not contain these magic numbers
    # the size of the final image should be calculated to exactly contain
    # both (warped) images.
    # In the warped version of image2 i simply overwrite with data
    # from image 1.
    tim = cv2.warpPerspective(im2, linalg.inv(ransac_match), (450, 300))
    M, N = im1.shape[:2]
    tim[0:M, 0:N, :] = im1
    cv2.waitKey(1)
    cv2.imshow('result', tim)

    cv2.waitKey(0)
