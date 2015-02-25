import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.ndimage import convolve, convolve1d
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imread
from pylab import subplot, imshow
import time
from mpl_toolkits.mplot3d import axes3d


def analytical_local_structure():

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
    size = s * 3.0
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)

    X, Y = np.meshgrid(x, y)

    G = np.exp(-(X**2 + Y**2) / (2.0 * s**2))
    new_G = G / G.sum()

    return new_G


def gauss_convolution():
    F = imread('cameraman.jpg', flatten=True)
    W = gauss(3)

    G = convolve(F, W, mode='nearest')

    subplot(1, 2, 1)
    imshow(F, cmap=cm.gray)
    subplot(1, 2, 2)
    imshow(G, cmap=cm.gray)
    plt.show()


def gauss1(s):
    size = s * 3.0
    x = np.arange(-size, size + 1)

    G = np.exp(-(x ** 2) / (2.0 * s ** 2))
    # G = (1 / np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / (2.0 * s ** 2))
    new_G = G / G.sum()

    return new_G


def gauss1_deriv(s):
    size = s * 3.0
    x = np.arange(-size, size + 1)
    G = (-x * np.exp(-x ** 2 / (2 * s ** 2)) / (s ** 2)) * (1 / (np.sqrt(2 * np.pi) * s))

    return G


def gauss1_second_deriv(s):
    size = s * 3.0
    x = np.arange(-size, size + 1)

    G = (((x ** 2) - (s ** 2)) * np.exp(-((x ** 2)/(2*(s ** 2))))/(s ** 4)) * (1 / (np.sqrt(2 * np.pi) * s))

    return G


def plot_gauss(s):
    size = s * 3.0
    x = np.arange(-size, size + 1)
    X, Y = np.meshgrid(x, x)

    Z = gauss(s)

    F = plt.figure()
    ax = F.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, Z)

    plt.show()


def separable_gauss_convolution():
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
    F = imread('cameraman.jpg', flatten=True)

    G = gD(F, 3, 0, 0)
    plt.subplot(1, 9, 1)
    imshow(F, cmap=cm.gray)

    G = gD(F, 3, 0, 1)
    plt.subplot(1, 9, 2)
    imshow(G, cmap=cm.gray)

    G = gD(F, 3, 0, 2)
    plt.subplot(1, 9, 3)
    imshow(G, cmap=cm.gray)

    G = gD(F, 3, 1, 0)
    plt.subplot(1, 9, 4)
    imshow(G, cmap=cm.gray)

    G = gD(F, 3, 2, 0)
    plt.subplot(1, 9, 5)
    imshow(G, cmap=cm.gray)

    G = gD(F, 3, 1, 1)
    plt.subplot(1, 9, 6)
    imshow(G, cmap=cm.gray)

    G = gD(F, 3, 1, 2)
    plt.subplot(1, 9, 7)
    imshow(G, cmap=cm.gray)

    G = gD(F, 3, 2, 1)
    plt.subplot(1, 9, 8)
    imshow(G, cmap=cm.gray)

    G = gD(F, 3, 2, 2)
    plt.subplot(1, 9, 9)
    imshow(G, cmap=cm.gray)

    plt.show()


def comparison():
    pass


def canny(F, s):
    # derivatives over x and y
    fx = gD(F, s, 1, 0)
    fy = gD(F, s, 0, 1)
    fxx = gD(F, s, 2, 0)
    fyy = gD(F, s, 0, 2)
    fxy = gD(F, s, 1, 1)

    # conditions expressed in cartesian coordinates; fw >> 0, fww = 0
    fw = np.sqrt(fx ** 2 + fy ** 2)
    fww = fx ** 2 * fxx + 2 * fx * fy * fxy + fy ** 2 * fyy

    edges = np.copy(fw)

    def check_edges(x, y):
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
                edges[y][x] = 0

    height, width = F.shape

    for i in range(width - 2):
        for j in range(height - 2):
            check_edges(i + 1, j + 1)

    return edges

    plt.imshow(edges)
    plt.gray()
    plt.show()


def canny_edge_detection():
    F = imread('cameraman.jpg', flatten=True)
    s = 2.0

    fx = gD(F, s, 1, 0)
    fy = gD(F, s, 0, 1)

    fw = np.sqrt(fx ** 2 + fy ** 2)
    fw_canny = canny(F, 3)

    plt.subplot(1, 2, 0)
    imshow(fw, cmap=cm.gray)
    plt.subplot(1, 2, 1)
    imshow(fw_canny, cmap=cm.gray)
    plt.show()

if __name__ == "__main__":
    # analytical_local_structure()
    # gauss_convolution()
    # separable_gauss_convolution()
    # gaussian_derivatives()
    # comparison()
    canny_edge_detection()
