import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.ndimage import convolve, convolve1d
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

    G = np.exp(-(x**2) / (2.0 * s**2))
    new_G = G / G.sum()

    return new_G


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


def gauss_derivatives():
    pass


def comparison():
    pass


def canny_edge_detector():
    pass


if __name__ == "__main__":
    # analytical_local_structure()
    # gauss_convolution()
    separable_gauss_convolution()
