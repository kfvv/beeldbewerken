# Student:     Kasper van Veen & Wessel Huising
# Stdnr:       6139752 & 10011277
# Course:      Beeldbewerken
# Date:        25/02/15
# Assignment:  3: Local Structure I

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.ndimage import convolve, convolve1d
from scipy.misc import imread
from pylab import subplot, imshow
import time
from mpl_toolkits.mplot3d import axes3d


def analytical_local_structure():
    """
    plots the quiver after calculating the derivatives of F
    """
    A = 1
    B = 2
    V = (6 * np.pi) / 201
    W = (4 * np.pi) / 201

    x = np.arange(-100, 101)
    y = np.arange(-100, 101)

    Y, X = np.meshgrid(x, y)

    xx = np.arange(-100, 101, 10)
    yy = np.arange(-100, 101, 10)

    YY, XX = np.meshgrid(yy, xx)

    F = (A * np.sin(V * X)) + (B * np.cos(W * Y))

    Fx = V * A * np.cos(V * XX)
    Fy = - W * B * np.sin(W * YY)

    plt.imshow(F, cmap=cm.gray, extent=(-100, 100, -100, 100))
    plt.quiver(yy, xx, Fy, -Fx, color='red')
    plt.show()


def gauss(s):
    """
    gauss returns the derivative of the gaussian function with scale s
    """
    size = s * 3.0
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)

    X, Y = np.meshgrid(x, y)

    G = np.exp(-(X ** 2 + Y ** 2) / (2.0 * s ** 2))
    new_G = G / G.sum()

    return new_G


def gauss_convolution():
    """
    gauss returns the derivative of the gaussian function with scale s
    """
    F = imread('cameraman.jpg', flatten=True)
    W = gauss(3)

    G = convolve(F, W, mode='nearest')

    subplot(1, 2, 1)
    imshow(F, cmap=cm.gray)
    subplot(1, 2, 2)
    imshow(G, cmap=cm.gray)
    plt.show()

    for s in [1, 2, 3, 5, 7, 9, 11, 15, 19]:
        now = time.time()
        W1 = gauss1(s)
        H = convolve1d(F, W1, axis=0, mode='nearest')
        G = convolve1d(H, W1, axis=1, mode='nearest')
        print('s={:2d}: {:.3f} ms'.format(s, (time.time() - now) * 1000))

    plot_gauss(3)


def gauss1(s):
    """
    gauss1 returns the derivative of the gaussian function with scale s
    """
    size = s * 3.0
    x = np.arange(-size, size + 1)

    G = np.exp(-(x ** 2) / (2.0 * s ** 2))
    new_G = G / G.sum()

    return new_G


def gauss1_deriv(s):
    """
    gauss_deriv returns the derivative of the gauss1 function with scale s
    """
    size = s * 3.0
    x = np.arange(-size, size + 1)
    G = (-x * np.exp(-x ** 2 / (2 * s ** 2)) / (s ** 2)) * (1 / (np.sqrt(2 * np.pi) * s))

    return G


def gauss1_second_deriv(s):
    """
    gauss1_second_deriv returns the second derivative of the gauss1 function
    with scale s
    """
    size = s * 3.0
    x = np.arange(-size, size + 1)

    G = (((x ** 2) - (s ** 2)) * np.exp(-((x ** 2)/(2*(s ** 2))))/(s ** 4)) * (1 / (np.sqrt(2 * np.pi) * s))

    return G


def plot_gauss(s):
    """
    plt_gauss return a 3D plot of the gaussian function
    """
    size = s * 3.0
    x = np.arange(-size, size + 1)
    X, Y = np.meshgrid(x, x)

    Z = gauss(s)

    F = plt.figure()
    ax = F.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, Z)

    plt.show()


def separable_gauss_convolution():
    """
    times and plots the derivatives of the gaussian function
    """
    F = imread('cameraman.jpg', flatten=True).astype(np.float)
    W1 = gauss1(3)

    H = convolve1d(F, W1, axis=0, mode='nearest')
    G = convolve1d(H, W1, axis=1, mode='nearest')

    subplot(1, 2, 1)
    imshow(F, cmap=cm.gray)
    subplot(1, 2, 2)
    imshow(G, cmap=cm.gray)
    plt.show()

    for s in [1, 2, 3, 5, 7, 9, 11, 15, 19]:
        now = time.time()
        W1 = gauss1(s)
        H = convolve1d(F, W1, axis=0, mode='nearest')
        G = convolve1d(H, W1, axis=1, mode='nearest')
        print('s={:2d}: {:.3f} ms'.format(s, (time.time() - now) * 1000))

    plot_gauss(3)


def gD(F, s, iorder, jorder):
    """
    convolve the gauss1, first- and second order derivative
    """
    if iorder == 0:
        F = convolve1d(F, gauss1(s), axis=0, mode='nearest')
    if iorder == 1:
        F = convolve1d(F, gauss1_deriv(s), axis=0, mode='nearest')
    if iorder == 2:
        F = convolve1d(F, gauss1_second_deriv(s), axis=0, mode='nearest')
    if jorder == 0:
        F = convolve1d(F, gauss1(s), axis=0, mode='nearest')
    if jorder == 1:
        F = convolve1d(F, gauss1_deriv(s), axis=1, mode='nearest')
    if jorder == 2:
        F = convolve1d(F, gauss1_second_deriv(s), axis=1, mode='nearest')
    return F


def gaussian_derivatives():
    """
    plot F, Fx, Fy, Fxx, Fyy and Fxy
    """
    F = imread('cameraman.jpg', flatten=True)

    # F
    G = gD(F, 3, 0, 0)
    plt.subplot(2, 3, 1)
    imshow(F, cmap=cm.gray)
    plt.title('F')

    # Fx
    G = gD(F, 3, 1, 0)
    plt.subplot(2, 3, 2)
    imshow(G, cmap=cm.gray)
    plt.title('Fx')

    # Fy
    G = gD(F, 3, 0, 1)
    plt.subplot(2, 3, 3)
    imshow(G, cmap=cm.gray)
    plt.title('Fy')

    # Fxx
    G = gD(F, 3, 2, 0)
    plt.subplot(2, 3, 4)
    imshow(G, cmap=cm.gray)
    plt.title('Fxx')

    # Fyy
    G = gD(F, 3, 0, 2)
    plt.subplot(2, 3, 5)
    imshow(G, cmap=cm.gray)
    plt.title('Fyy')

    # Fxy
    G = gD(F, 3, 1, 1)
    plt.subplot(2, 3, 6)
    imshow(G, cmap=cm.gray)
    plt.title('Fxy')

    plt.show()


def comparison():
    pass


def canny(F, s):
    """
    derivatives over x and y, it applies the canny edge detection on image F
    with scale s
    """
    fx = gD(F, s, 1, 0)
    fy = gD(F, s, 0, 1)
    fxx = gD(F, s, 2, 0)
    fyy = gD(F, s, 0, 2)
    fxy = gD(F, s, 1, 1)

    # conditions expressed in cartesian coordinates; fw >> 0, fww = 0
    fw = np.sqrt(fx ** 2 + fy ** 2)
    fww = fx ** 2 * fxx + 2 * fx * fy * fxy + fy ** 2 * fyy

    pict_edges = np.array(fw)

    height, width = F.shape

    for y in xrange(width - 2):
        for x in xrange(height - 2):
            if ((fww[y][x - 1] > 0 and fww[y][x + 1] < 0) or
                (fww[y][x - 1] < 0 and fww[y][x + 1] > 0) or
                (fww[y - 1][x] > 0 and fww[y + 1][x] < 0) or
                (fww[y - 1][x] < 0 and fww[y + 1][x] > 0) or
                (fww[y - 1][x + 1] > 0 and fww[y + 1][x + 1] < 0) or
                (fww[y - 1][x + 1] < 0 and fww[y + 1][x + 1] > 0) or
                (fww[y - 1][x - 1] > 0 and fww[y - 1][x + 1] < 0) or
                (fww[y - 1][x - 1] < 0 and fww[y - 1][x + 1] > 0) or
                (fww[y + 1][x - 1] > 0 and fww[y + 1][x + 1] < 0) or
                (fww[y - 1][x - 1] < 0 and fww[y + 1][x - 1] > 0) or
                (fww[y - 1][x - 1] < 0 and fww[y + 1][x + 1] > 0)):
                    pict_edges[y][x] = 0

    return pict_edges


def canny_edge_detection():
    """
    plots the canny edge detection figure. fw is using the canny edge, and
    fw_grad is using the gradient to detect the edges
    """
    F = imread('cameraman.jpg', flatten=True)
    s = 2.0

    fx = gD(F, s, 1, 0)
    fy = gD(F, s, 0, 1)

    fw = np.sqrt(fx ** 2 + fy ** 2)
    fw_grad = canny(F, 3)

    plt.subplot(1, 2, 0)
    imshow(fw, cmap=cm.gray)
    plt.subplot(1, 2, 1)
    imshow(fw_grad, cmap=cm.gray)
    plt.show()


if __name__ == "__main__":
    # analytical_local_structure()
    # gauss_convolution()
    # separable_gauss_convolution()
    # gaussian_derivatives()
    # comparison()
    canny_edge_detection()
