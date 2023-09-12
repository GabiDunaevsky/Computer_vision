import math


import numpy as np
import cv2
import time
from collections import defaultdict



def myID()->np.int:
    return 208407379

def norm2dArray(mat: np.ndarray)->np.ndarray:
    if mat.max() > 1:
        mat = cv2.normalize(mat, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return mat

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    lst = []
    kernel = k_size[::-1]
    for i in range(1 - len(kernel), len(in_signal)):
        lst.append(np.dot(
            in_signal[max(0, i):min(i + len(kernel), len(in_signal))],
            kernel[max(-i, 0):len(in_signal) - i * (len(in_signal) - len(kernel) < i)],
        ))
    return np.array(lst)




def conv2D(in_image:np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    KernelTranspose = kernel.T   # Fliping the kernel
    # normalizedImg = norm2dArray(in_image)  # Normalizing The img
    FramedImg = cv2.copyMakeBorder(in_image, int((KernelTranspose.shape[0] / 2)), int((KernelTranspose.shape[0] / 2)), int((KernelTranspose.shape[1] / 2)), int((KernelTranspose.shape[1] / 2)),
                                   cv2.BORDER_REPLICATE)
    imgConRes = np.ndarray(in_image.shape)
    for y in range(imgConRes.shape[1]): #  number rows
        for x in range(imgConRes.shape[0]): # number columns
            imgConRes[x, y] = np.multiply(FramedImg[x: x + KernelTranspose.shape[0], y: y + KernelTranspose.shape[1]], KernelTranspose).sum()
    if imgConRes.max() > 1: # after normlize all the values need to be in int so i round them.
        imgConRes = np.round(imgConRes)
    return imgConRes



def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    kernel = np.array([[1, 0, -1]])
    Ix = conv2D(in_image, kernel.T)
    Iy = conv2D(in_image, kernel)

    imageMag = np.sqrt((np.power(Ix, 2) + np.power(Iy, 2)))
    imageDir = np.arctan2(Iy, Ix).astype(np.float64)

    return imageDir, imageMag


def getPascalLine(line):
    ans = []
    for i in range(0, line):
        ans.append(binomialCoeff(line - 1, i))
    return np.array([ans])


def binomialCoeff(n, k):
    res = 1
    if k > n - k:
        k = n - k
    for i in range(0, k):
        res = res * (n - i)
        res = res // (i + 1)

    return res


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    pascalLine = getPascalLine(k_size)
    kernel = np.multiply(np.array(pascalLine), np.array(pascalLine).T)
    kernel = kernel / np.sum(kernel)
    blurIm = conv2D(in_image, kernel)

    return blurIm


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    pascalLine = cv2.getGaussianKernel(k_size, 0)
    kernel = np.multiply(np.array(pascalLine), np.array(pascalLine).T)
    # return cv2.GaussianBlur(in_image, (k_size, k_size), 0, borderType=cv2.BORDER_REPLICATE)
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    # return conv2D(in_image, kernel)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    print("I implemented The ZeroCrossingLog")
    return img



def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    imgAfterBlur = cv2.GaussianBlur(img, (9, 9), 2, borderType=cv2.BORDER_REPLICATE)
    imgAfterLap = cv2.Laplacian(imgAfterBlur, -1, borderType=cv2.BORDER_REPLICATE, ksize=5)
    time.sleep(2)
    # Making the zero crossing.
    edges = np.empty((len(imgAfterLap), len(imgAfterLap[0]))).astype(np.bool8)
    for i in range(len(imgAfterLap) - 1):
        for j in range(len(imgAfterLap[0]) - 1):
            if (imgAfterLap[i][j] > 0) == (imgAfterLap[i][j + 1] < 0) or \
                    (imgAfterLap[i][j] > 0) == (imgAfterLap[i + 1][j] < 0) or \
                    (imgAfterLap[i][j] == 0 and imgAfterLap[i][j + 1] != 0 and imgAfterLap[i + 1][i] != 0):
                edges[i][j] = 1
    return edges

def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use Open CV function: cv.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    pixels = []
    steps = 60
    thresh = 0.55
    for r in range(min_radius, max_radius + 1):
        for t in range(steps):
            pixels.append((r, int(r * np.cos(2 * np.pi * t / steps)), int(r * np.sin(2 * np.pi * t / steps))))

    edges = np.where(cv2.Canny((img * 255).astype(np.uint8), 45, 150) == 255)
    dDict = defaultdict(int)

    for i in range(edges[0].shape[0]):
        y = edges[0][i]
        x = edges[1][i]
        for r, distX, distY in pixels:
            a = x - distX
            b = y - distY
            dDict[(a, b, r)] += 1

    circles = []
    for k, v in sorted(dDict.items(), key=lambda i: -i[1]):
        x, y, r = k
        
        if v / steps >= thresh and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))

    return circles



def gausianVector(img: np.ndarray, variance: float) -> np.ndarray:
    # For applying gaussian function for each element in matrix.
    sigma = math.sqrt(variance)
    cons = 1 / (sigma * math.sqrt(2 * math.pi))
    return cons * np.exp(-((img / sigma) ** 2) * 0.5)


def get_slice(img: np.ndarray, x: int, y: int, kerSize: int) -> np.ndarray:
    halfKer = kerSize // 2
    return img[x - halfKer: x + halfKer + 1, y - halfKer: y + halfKer + 1]


def creatGaussKer(kerSize: int, spatialVar: float) -> np.ndarray:

    arr = np.zeros((kerSize, kerSize))
    for i in range(0, kerSize):
        for j in range(0, kerSize):
            arr[i, j] = math.sqrt(abs(i - kerSize // 2) ** 2 + abs(j - kerSize // 2) ** 2)
    return gausianVector(arr, spatialVar)


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    res = np.zeros(in_image.shape)
    gaussianKernel = creatGaussKer(k_size, sigma_space)
    sizeX, sizeY = in_image.shape
    for i in range(k_size // 2, sizeX - k_size // 2):
        for j in range(k_size // 2, sizeY - k_size // 2):
            imgSlice = get_slice(in_image, i, j, k_size)
            imgI = imgSlice - imgSlice[k_size // 2, k_size // 2]
            weights = np.multiply(gaussianKernel, gausianVector(imgI, sigma_color))
            values = np.multiply(imgSlice, weights)
            val = np.sum(values) / np.sum(weights)
            res[i, j] = val

    imgOpenCv = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space, borderType=cv2.BORDER_REPLICATE)
    return imgOpenCv, imgOpenCv


