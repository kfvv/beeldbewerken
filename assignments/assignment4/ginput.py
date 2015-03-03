# -*- coding: utf-8 -*-
"""
ginput(win,n) returns n points clicked in the image displayed in win
A very clumsy implementation using globals (please mail me improvements)
"""
import cv2

ginput_xs = []
ginput_ys = []


def ginput(win, n):
    global ginput_xs, ginput_ys
    ginput_xs = []
    ginput_ys = []

    def onMouse(event, x, y, flags, param):
        global ginput_xs, ginput_ys
        if event == cv2.EVENT_LBUTTONDOWN:
            ginput_xs += [x]
            ginput_ys += [y]
            print(ginput_xs)

    cv2.setMouseCallback(win, onMouse)

    while len(ginput_xs) != n:
        cv2.waitKey(1)

    if n == 1:
        return ginput_xs[0], ginput_ys[0]
    return ginput_xs, ginput_ys


if __name__ == '__main__':
    a = cv2.imread('images/trui.png')
    cv2.imshow('test', a)
    print(ginput('test', 4))
