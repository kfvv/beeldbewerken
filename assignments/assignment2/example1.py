import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def show_exercise1():
    A = 1
    B = 2
    V = 6 * np.pi / 201
    W = 4 * np.pi / 201
    x = np.arange(-100, 101)
    y = np.arange(-100, 101)
    Y, X = np.meshgrid(x, y)
    F = A * np.sin(V * X) + B * np.cos(W * Y)
    xx = np.arange(-100, 101, 10)
    yy = np.arange(-100, 101, 10)
    YY, XX = np.meshgrid(yy, xx)
    FFx = A * V * np.cos(V * XX)
    FFy = -B * W * np.sin(W * YY)
    plt.clf()
    plt.imshow(F, cmap=cm.gray, extent=(-100, 100, -100, 100))
    plt.quiver(yy, xx, FFy, -FFx, color='red')
    plt.show()


if __name__ == "__main__":
    show_exercise1()
