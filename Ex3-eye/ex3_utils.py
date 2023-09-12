import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from skimage.registration import phase_cross_correlation
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import math
from matplotlib.pyplot import imshow, plot, figure


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208407379


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    vecForX = np.array([[1, 0, -1]])
    vecForY = vecForX.T
    originalPoints = []
    distForOriginalPoints = []

    Ix = ndimage.convolve(im1, vecForX)
    Iy = ndimage.convolve(im1, vecForY)
    It = im2 - im1

    for i in range(0, im1.shape[0], step_size):
        for j in range(0, im1.shape[1], step_size):
            IxForWind = Ix[i:i + win_size, j:j + win_size].flatten()  ## for the matrix
            IYForWind = Iy[i:i + win_size, j:j + win_size].flatten()  ## for the matrix
            ItForWind = It[i:i + win_size, j:j + win_size].flatten()  ## for the matrix
            A = [[(IxForWind * IxForWind).sum(), (IxForWind * IYForWind).sum()],
                 [(IxForWind * IYForWind).sum(), (IYForWind * IYForWind).sum()]]
            if not checkEigenvaluesCorrectness(A):
                originalPoints.append([j, i])
                distForOriginalPoints.append(np.array([0., 0.]))
            else:
                At = [[-(IxForWind * ItForWind).sum()], [-(IYForWind * ItForWind).sum()]]
                originalPoints.append([j, i])
                # ans = (np.linalg.inv(A) @ At).reshape(2)
                distForOriginalPoints.append((np.linalg.inv(A) @ At).reshape(2))
    return np.array(originalPoints), np.array(distForOriginalPoints)


def checkEigenvaluesCorrectness(mat: np.ndarray) -> bool:
    eigenvalues = np.linalg.eigvals(
        mat)  # func that i found in numpy https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html
    lamda1 = np.max(eigenvalues)
    lamda2 = np.min(eigenvalues)
    if lamda2 == 0.0:
        return False
    proLambdas = lamda1 / lamda2
    if lamda2 <= 1 or proLambdas >= 100:
        return False
    return True



def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    # In this question I was consulting with a classmate
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    gaussianPyrImg1 = gaussianPyr(img1, k)  # Gaussian pyramid of size k
    gaussianPyrImg2 = gaussianPyr(img2, k)  # Gaussian pyramid of size k

    currRows, currColums = gaussianPyrImg1[-1].shape
    orgPoints, distForOriginal = opticalFlow(gaussianPyrImg1[-1], gaussianPyrImg2[-1], stepSize, winSize)
    finalPic = distForOriginal.reshape(currRows // stepSize, currColums // stepSize, 2)
    for i in range(1, k):
        finalPic = expendingThePic(finalPic)
        orgPoints, distForOriginal = opticalFlow(gaussianPyrImg1[-i - 1], gaussianPyrImg2[-i - 1], stepSize, winSize)
        currRows, currColums = gaussianPyrImg1[-i - 1].shape
        upRows = currRows // stepSize
        upColumns = currColums // stepSize
        finalPic += distForOriginal.reshape(upRows, upColumns, 2)
    return finalPic

def expendingThePic(finalPic:np.ndarray) -> np.ndarray:
    finalPic *= 2
    rows, columns = (finalPic.shape[1] * 2, finalPic.shape[0] * 2)
    imgAfterExp = np.zeros((columns, rows, 2), dtype=finalPic.dtype)
    imgAfterExp[::2, ::2] = finalPic
    finalPic = imgAfterExp
    return finalPic


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
# In this question I was consulting with a classmate


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []

    for mat in matches[:10]:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])

    A = np.tile([[1, 0], [0, 1]], (10, 1))
    B = (np.array(list_kp2) - np.array(list_kp1)).reshape(-1, 1)

    ans = (np.linalg.inv(A.T @ A) @ A.T @ B)
    Dx, Dy = ans[0, 0], ans[1, 0]

    transelationMat = np.array([[1, 0, Dx],
                                [0, 1, Dy],
                                [0, 0, 1]])
    return transelationMat


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """

    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []
    for mat in matches[:10]:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])

    ang = -25
    rotatedImg = rotation(im2, ang)
    matrix = np.array([[np.cos(np.radians(ang)), -np.sin(np.radians(ang)), 0],
                       [np.sin(np.radians(ang)), np.cos(np.radians(ang)), 0],
                       [0, 0, 1]], dtype=np.float64)

    trans = findTranslationLK(im1, rotatedImg)
    ans = matrix @ trans

    return ans


def rotation(image: np.ndarray, theta) -> np.ndarray:
    h, w = image.shape
    X, Y = np.meshgrid(range(w), range(h))

    img_padded = np.ones((h * 2, w * 2)) * 255
    img_padded[0: h, 0: w] = image
    mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    mat_inv = np.linalg.inv(mat)
    XY1 = np.ones((h, w, 3))
    XY1[:, :, 0] = X
    XY1[:, :, 1] = Y
    XY1 = XY1.reshape((h * w, 3))

    XY2 = mat_inv.dot(XY1.T)
    new_img = img_padded[XY2.T[:, 1].astype(int), XY2.T[:, 0].astype(int)]
    new_img = new_img.reshape((h, w))

    return new_img



def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []

    for mat in matches[:10]:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt


        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])

    A = np.tile([[1, 0], [0, 1]], (10, 1))
    B = (np.array(list_kp2) - np.array(list_kp1)).reshape(-1, 1)

    ans = (np.linalg.inv(A.T @ A) @ A.T @ B)
    Dx, Dy = ans[0, 0], ans[1, 0]

    transelationMat = np.array([[1, 0, Dx],
                                [0, 1, Dy],
                                [0, 0, 1]])
    return transelationMat

def angle(s1, s2):
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []
    for mat in matches[:10]:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])

    theta = -25
    rotatedImg = rotation(im2, theta)
    mat = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
                       [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
                       [0, 0, 1]], dtype=np.float64)

    trans = findTranslationLK(im1, rotatedImg)
    res = mat @ trans

    return res


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    fig = plt.figure()
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(im1, cmap='gray')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(im2, cmap='gray')
    plt.show()
    return im1



# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyrLst = []
    pyrLst.append(img)
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    for i in range(1, levels):
        tmpPyr = cv2.GaussianBlur(pyrLst[i - 1], (5, 5), sigma, borderType=cv2.BORDER_REPLICATE)
        tmpPyr = tmpPyr[::2, ::2]
        pyrLst.append(tmpPyr)
    return pyrLst


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    # """
    LalplacianPy = []
    GaussianLst = gaussianPyr(img, levels)
    LalplacianPy.append(GaussianLst[-1])  #Last picture is identical
    for i in range(1, levels):
        tmp = GaussianLst[-i]
        rows, columns = (tmp.shape[1] * 2, tmp.shape[0] * 2)
        pictureAfterExp = cv2.resize(tmp, (rows, columns))
        if pictureAfterExp.shape != GaussianLst[-i-1].shape:  # because of RGB or a small arithmetic issues.
            pictureAfterExp = cv2.resize(pictureAfterExp, (GaussianLst[-i-1].shape[1], GaussianLst[-i-1].shape[0]))
        LalplacianPy.insert(0, np.asarray(GaussianLst[-i-1] - pictureAfterExp))
    return LalplacianPy


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    originalPicture = lap_pyr[-1]
    # sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    for i in range(1, len(lap_pyr)):
        rows, columns = (originalPicture.shape[1] * 2, originalPicture.shape[0] * 2)
        pictureAfterExp = cv2.resize(originalPicture, (rows, columns))
        # gaussianKerForExp = cv2.getGaussianKernel(5, sigma) * 4
        # pictureAfterExp = cv2.sepFilter2D(pictureAfterExp, -1, gaussianKerForExp, gaussianKerForExp)
        originalPicture = lap_pyr[-i-1] + pictureAfterExp

    return originalPicture


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    laplacianPyramidImg1 = laplaceianReduce(img_1, levels)
    laplacianPyramidImg2 = laplaceianReduce(img_2, levels)
    gaussianPyramidMask = gaussianPyr(mask, levels)
    mergedImg = laplacianPyramidImg1[-1] * gaussianPyramidMask[-1] + (1 - gaussianPyramidMask[-1]) * laplacianPyramidImg2[-1]
    naiveBlend = img_1 * mask + (1-mask) * img_2
    for i in range(1, levels):
        rows, columns = laplacianPyramidImg1[-i-1].shape[1], laplacianPyramidImg1[-i-1].shape[0]
        mergedImg = cv2.resize(mergedImg, (rows, columns)) + laplacianPyramidImg1[-i-1] * gaussianPyramidMask[-i-1] + (1 - gaussianPyramidMask[-i-1]) * laplacianPyramidImg2[-i-1]
    return naiveBlend, mergedImg


