import numpy as np
import cv2


def drawMatches(img1, kp_img1, img2, kp_img2, matches):
    """Draw matches between img1 and img2, given there keypoints and matches

    Args:
        img1(np.array) - image 1
        kp_img1(list) - output of this:
                        kp_img1 des1 = sift.detectAndCompute(img1, None)
        img2(np.array) - image 2
        kp_img2(list) - same as kp_img1 but than for img2
        matches(list)- list of DMatches type (see opencv doc)
    """
    img1_height, img1_width, depth = img1.shape
    img2_height, img2_width, depth = img2.shape

    match_img = np.zeros((max(img1_height, img2_height),
                          img1_width + img2_width, depth)).astype('uint8')

    match_img[:img1_height, :img1_width, :] = img1
    match_img[:img2_height, img1_width:img1_width + img2_width, :] = img2

    # get the indexes of the matches
    #matches_im1 = [m.queryIdx for m in matches]
    #matches_im2 = [m.trainIdx for m in matches]

    # get source and dest keppoints
    #source = np.array([list(kp_img1[i].pt)
    #                  for i in matches_im1]).astype('uint8')
    #dest = np.array([list(kp_img2[i].pt)
    #                for i in matches_im2]).astype('uint8')

    # draw circles and lines between these points
    for i in range(len(matches)):
        src_pt, dest_pt = matches[i][0].pt, matches[i][1].pt

        src_pt = tuple(np.array(src_pt).astype('uint8'))
        dest_pt = tuple(np.array(dest_pt).astype('uint8'))
        dest_pt = dest_pt[0] + img1_width, dest_pt[1]

        cv2.circle(match_img, src_pt, 4, (0, 255, 0))
        cv2.circle(match_img, dest_pt, 4, (255, 0, 0))

        cv2.line(match_img, src_pt, dest_pt, (0, 0, 255))

    return match_img
