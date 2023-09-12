import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    map = np.zeros((img_l.shape[0], img_l.shape[1]))
    dispSize = disp_range[1] - disp_range[0] + 1
    for i in range(0, img_l.shape[0]):
        for j in range(0, img_l.shape[1]):
            minSSD = np.inf
            removeI = i + disp_range[1] + k_size
            removeJ = j + disp_range[1] + k_size
            if checkPicLimits(img_l, (removeI, removeJ), k_size, 0):
                continue
            for o in range(dispSize):
                if checkPicLimits(img_r, (removeI, removeJ), k_size, o):
                    continue
                leftImg = img_l[removeI - k_size:removeI + k_size + 1, removeJ - k_size:removeJ + k_size + 1]
                rightImg = img_r[removeI - k_size: removeI + k_size + 1, removeJ - o - k_size: removeJ + k_size - o + 1]
                SSD = np.sum(np.square(leftImg - rightImg))
                if SSD < minSSD:
                    map[i][j] = o
                    minSSD = SSD
    return map


def checkPicLimits(img, point: (float, float), k_size, mov) -> bool:
    if point[0] + mov - k_size < 0 or point[0] + mov + k_size >= img.shape[0] or point[1] + mov - k_size < 0 or point[1] + mov + k_size >= img.shape[1]:
        return True
    return False


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    map = np.zeros((img_l.shape[0], img_l.shape[1]))
    dispSize = disp_range[1] - disp_range[0] + 1
    for i in range(0, img_l.shape[0]):
        for j in range(0, img_l.shape[1]):
            maxNC = 0
            removeI = i + disp_range[1] + k_size
            removeJ = j + disp_range[1] + k_size
            if checkPicLimits(img_l, (removeI, removeJ), k_size, 0):
                continue
            for o in range(dispSize):
                if checkPicLimits(img_r, (removeI, removeJ), k_size, o):
                    continue
                numerator = calculatingRFunction(img_l, img_r, (removeI, removeJ), k_size, o, 0)
                secondPart = calculatingRFunction(img_r, img_r, (removeI, removeJ), k_size, o, 2)
                thirdPart = calculatingRFunction(img_l, img_l, (removeI, removeJ), k_size, o, 1)
                denominator = np.sqrt(secondPart * thirdPart)
                if denominator != 0:
                    NC = numerator / denominator
                else:
                    NC = 0
                if NC > maxNC:
                    maxNC = NC
                    map[i][j] = o
    return map


def calculatingRFunction(imgLeft, imgRight, point: (float, float), k_size, mov, mod):
    if mod == 0:  # right and left
        leftImg = imgLeft[point[0] - k_size:point[0] + k_size + 1, point[1] - k_size:point[1] + k_size + 1]
        rightImg = imgRight[point[0] - k_size: point[0] + k_size + 1, point[1] - mov - k_size: point[1] + k_size - mov + 1]
        NC = np.sum(leftImg * rightImg)
        return NC
    elif mod == 1:  # Both left
        leftImg = imgLeft[point[0] - k_size:point[0] + k_size + 1, point[1] - k_size:point[1] + k_size + 1]
        NC = np.sum(leftImg * leftImg)
        return NC
    else:  # Both right
        rightImg = imgRight[point[0] - k_size: point[0] + k_size + 1, point[1] - mov - k_size: point[1] + k_size - mov + 1]
        NC = np.sum(rightImg * rightImg)
        return NC


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    A = []  # creating mat A
    src = []
    dst = []
    for i in range(src_pnt.shape[0]):
        A.append(
            [src_pnt[i][0], src_pnt[i][1], 1, 0, 0, 0, -dst_pnt[i][0] * src_pnt[i][0], -dst_pnt[i][0] * src_pnt[i][1],
             -dst_pnt[i][0]])
        A.append(
            [0, 0, 0, src_pnt[i][0], src_pnt[i][1], 1, -dst_pnt[i][1] * src_pnt[i][0], -dst_pnt[i][1] * src_pnt[i][1],
             -dst_pnt[i][1]])

    u, s, vh = np.linalg.svd(np.asarray(A))
    M = vh[-1].reshape(3, 3)
    M = M / M[-1, -1]  # normalize the by last place.
    for i in range(src_pnt.shape[0]):
        src.append(np.append(src_pnt[i], 1))
        dst.append(np.append(dst_pnt[i], 1))

    srcPoint = M.dot(np.array(src).T) / M.dot(np.array(src).T)[-1]  # normalize by last place
    err = np.sqrt(np.sum(np.square(srcPoint.T - dst)))

    return M, err


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """


    dst_p = []
    src_p = []
    fig1 = plt.figure()
    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        print(dst_p)
        plt.show()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        print(src_p)
        plt.show()

    fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)


    fig2 = plt.figure()
    fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)
    H, err = computeHomography(src_p, dst_p)
    proj_src = cv2.warpPerspective(src_img, H, (dst_img.shape[1], dst_img.shape[0]))
    mask = (proj_src == 0)
    canvas = dst_img * mask + (1 - mask) * proj_src
    plt.imshow(canvas)
    plt.show()

