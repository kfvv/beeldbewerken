from pylab import *
import numpy
import numpy.linalg
import matplotlib.pyplot as plt


def calibration(XYZ, xy):
    matM = zeros((len(XYZ) * 2, 12))

    for i in xrange(len(xy)):
        matM[i * 2] = [XYZ[i, 0], XYZ[i, 1], XYZ[i, 2], 1, 0, 0, 0, 0,
                -xy[i, 0] * XYZ[i, 0], -xy[i, 0] * XYZ[i,1],
                -xy[i, 0] * XYZ[i, 2], -xy[i, 0]]
        matM[i * 2 + 1] = [0, 0, 0, 0, XYZ[i, 0], XYZ[i, 1], XYZ[i, 2], 1,
                -xy[i, 1] * XYZ[i, 0], -xy[i, 1] * XYZ[i, 1], -
                xy[i, 1] * XYZ[i, 2], -xy[i, 1]]

    U, D, V = svd(matM)

    p = V[-1]

    P = p.reshape(3, 4)
    print(p)

    return P

    # return V[-1].reshape(3, 4)


def avarage_error(ai, xy):
    """
    reprojection distance: sqrt((xi - ai)^2 + (yi - bi)^2), used to calculate
    the reprojection error as the avarage of the Euclidean distance
    """
    d = 0

    for i in range(len(ai)):
        d += sqrt((xy[i, 0] - ai[i, 0]) ** 2 + (xy[i, 1] - ai[i, 1]) ** 2)

        result = d / len(ai)

    return result


def drawCube(P, X, Y, Z):
    vertices = np.array([[X,     Y,     Z],
                        [X + 1, Y,     Z],
                        [X,     Y + 1, Z],
                        [X + 1, Y + 1, Z],
                        [X,     Y,     Z - 1],
                        [X + 1, Y,     Z - 1],
                        [X,     Y + 1, Z - 1],
                        [X + 1, Y + 1, Z - 1]])
    print(vertices.shape)

# matrix containing image coordinates in 2D on the checkerboard.
xy = array([[213.1027,  170.0499], [258.1908,  181.3219],
            [306.41,    193.8464], [351.498,   183.8268],
            [382.8092,  155.6468], [411.6155,  130.5978],
            [223.7485,  218.2691], [267.5841,  230.7935],
            [314.5509,  244.5705], [357.7603,  235.1771],
            [387.819,   205.1184], [415.3728,  178.1908],
            [234.3943,  263.9834], [276.9775,  277.1341],
            [323.318,   291.5372], [363.3963,  282.1438],
            [392.8288,  251.4589], [419.1301,  223.9051]])

# corresponding points in 3D
XYZ = array([[0, -5, 5], [0, -3, 5], [0, -1, 5], [-1, 0, 5],
            [-3, 0, 5], [-5, 0, 5], [0, -5, 3], [0, -3, 3],
            [0, -1, 3], [-1, 0, 3], [-3, 0, 3], [-5, 0, 3],
            [0, -5, 1], [0, -3, 1], [0, -1, 1], [-1, 0, 1],
            [-3, 0, 1], [-5, 0, 1]])

# print matrix P
P = calibration(XYZ, xy)
image = imread('images/calibrationpoints.jpg')
imshow(image)

plt.plot(xy[:, 0], xy[:, 1], 'd')
axis('off')
axis('equal')
show()
