# Student:     Kasper van Veen & Wessel Huising
# Stdnr:       6139752 & 10011277
# Course:      Beeldbewerken
# Date:        18/03/15
# Assignment:  5: Camera Calibration

from pylab import zeros, svd, imread, imshow, show, axis
import numpy as np
import matplotlib.pyplot as plt
import time


def calibration(XYZ, xy):
    """
    Find null vector of matrix A, next use SVD and reshape into matrix with the
    correst shape.
    """
    matA = zeros((len(XYZ) * 2, 12))

    for i in range(len(xy)):
        matA[i * 2] = [XYZ[i, 0], XYZ[i, 1], XYZ[i, 2], 1, 0, 0, 0, 0,
                       -xy[i, 0] * XYZ[i, 0], -xy[i, 0] * XYZ[i, 1],
                       -xy[i, 0] * XYZ[i, 2], -xy[i, 0]]
        matA[i * 2 + 1] = [0, 0, 0, 0, XYZ[i, 0], XYZ[i, 1], XYZ[i, 2], 1,
                           -xy[i, 1] * XYZ[i, 0], -xy[i, 1] * XYZ[i, 1],
                           -xy[i, 1] * XYZ[i, 2], -xy[i, 1]]

    U, D, V = svd(matA)

    return V[-1].reshape(3, 4)


def calc_3d(P, XYZ):
    """Calculate 2D points from 3D"""

    # calc first point
    result = np.append(XYZ[0], 1)
    result = np.dot(P, result)
    result /= result[-1]
    result = result[:2]

    # calc for the rest of the points
    for point in XYZ[1:]:
        h_point = np.append(point, 1)
        tmp_result = np.dot(P, h_point)
        tmp_result /= tmp_result[-1]
        result = np.vstack((result, tmp_result[:2]))

    return result


def average_error(P, XYZ, xy):
    """
    Reprojection distance: sqrt((xi - ai)^2 + (yi - bi)^2), used to calculate
    the reprojection error as the avarage of the Euclidean distance.
    """
    # calc first point
    result = calc_3d(P, XYZ)

    return np.linalg.norm(result - xy)


def drawLines(vertices, color='b'):
    """Draw lines for the top- and bottom plain."""
    x_values = vertices[:, 0]
    y_values = vertices[:, 1]

    for i in range(len(vertices)):
        # wrap around
        end_vertex = (i + 1) % len(vertices - 1)

        plt.plot([x_values[i], x_values[end_vertex]],
                 [y_values[i], y_values[end_vertex]], color=color)


def connectPlaines(vertices, color='b'):
    """
    Connect the top- and bottom plains with each other by drawing lines between
    them.
    """
    for vertex in vertices:
        top = vertex[1]
        bottom = vertex[0]
        plt.plot([top[0], bottom[0]], [top[1], bottom[1]], color=color)


def drawCube(P, X, Y, Z):
    """
    Use the dot product between P and the homogeneous representation of the 3D
    calibration to get the euclidean norm.
    """
    vertices_bottom = np.array([[X,     Y,     Z],
                                [X,     Y + 1, Z],
                                [X + 1, Y + 1, Z],
                                [X + 1, Y,     Z]])

    vertices_top = np.array([[X,     Y,     Z + 1],
                            [X,     Y + 1, Z + 1],
                            [X + 1, Y + 1, Z + 1],
                            [X + 1, Y,     Z + 1]])

    vertices_x = calc_3d(P, vertices_bottom)
    vertices_y = calc_3d(P, vertices_top)

    drawLines(vertices_x)
    drawLines(vertices_y)

    connected_vertices = zip(vertices_x, vertices_y)

    return connectPlaines(connected_vertices)


def movingCube(P, image):
    """Draw a 3D cube that moves along the checkerboard"""
    for i in range(0, 7):
        imshow(image)
        cube = drawCube(P, -i, 0, 0)
        plt.pause(0.001)
    plt.show(block=True)


if __name__ == "__main__":
    # matrix containing image coordinates in 2D on the checkerboard.
    xy = np.array([[213.1027,  170.0499], [258.1908,  181.3219],
                   [306.41,    193.8464], [351.498,   183.8268],
                   [382.8092,  155.6468], [411.6155,  130.5978],
                   [223.7485,  218.2691], [267.5841,  230.7935],
                   [314.5509,  244.5705], [357.7603,  235.1771],
                   [387.819,   205.1184], [415.3728,  178.1908],
                   [234.3943,  263.9834], [276.9775,  277.1341],
                   [323.318,   291.5372], [363.3963,  282.1438],
                   [392.8288,  251.4589], [419.1301,  223.9051]])

    # corresponding points in 3D
    XYZ = np.array([[0, -5, 5], [0, -3, 5], [0, -1, 5], [-1, 0, 5],
                    [-3, 0, 5], [-5, 0, 5], [0, -5, 3], [0, -3, 3],
                    [0, -1, 3], [-1, 0, 3], [-3, 0, 3], [-5, 0, 3],
                    [0, -5, 1], [0, -3, 1], [0, -1, 1], [-1, 0, 1],
                    [-3, 0, 1], [-5, 0, 1]])

    # print matrix P
    P = calibration(XYZ, xy)
    drawCube(P, 0, 0, 0)

    print('Average error is: ', average_error(P, XYZ, xy))
    print(P)
    image = imread('images/calibrationpoints.jpg')

    # movingCube(P, image)
    imshow(image)

    plt.plot(xy[:, 0], xy[:, 1], 'd')
    axis('off')
    axis('equal')
    show()
