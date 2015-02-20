import numpy as np


A = 1
B = 2
V = (6 * np.pi) / 201
W = (4 * np.pi) / 201

x = np.arange(-100, 101)
y = np.arange(-100, 101)

xx = np.arange(-100, 101, 10)
yy = np.arange(-100, 101, 10)

Y, X = np.meshgrid(x, y)
YY, XX = np.meshgrid(yy, xx)


def F(X, Y, A, B, V, W):
    F = (A * np.sin(V * X)) + (B * np.cos(W * Y))
    return F


def fx(X, Y, A, B, V, W):
    Fx = V * A * np.cos(V * X)
    return Fx


def fy(X, Y, A, B, V, W):
    Fy = - W * B * np.sin(W * Y)
    return Fy


def FFx(X, Y, A, B, V, W):
    FFx = - A * V * V * np.sin(V * XX)
    return FFx


def FFy(X, Y, A, B, V, W):
    FFy = - B * W * W * np.cos(W * YY)
    return FFy


if __name__ == "__main__":
    print(F(X, Y, A, B, V, W))
