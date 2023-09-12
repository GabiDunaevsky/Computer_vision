"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208588392


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # as opencv loads in BGR format by default, we want to show it in RGB.
    if representation == 1:
        imageGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return imageGray.astype('float32')
    if representation == 2:
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return imageRGB.astype('float32')

    return None


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)
    plt.imshow(image, cmap='gray')
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])

    # from shape (x,y,3) reshape to (3,x,y)
    imgRGB = imgRGB.transpose(2, 0, 1)
    # from shape (3,x,y) to (3,x*y)
    img_shape_after_trans = imgRGB.shape
    imgRGB = imgRGB.reshape(3, -1)

    # make the transformation by multiplication
    YIQ = np.dot(yiq_from_rgb, imgRGB)

    # from shape (3,x*y) back to (3,x,y)
    YIQ = YIQ.reshape(img_shape_after_trans)
    # from shape (3,x,y) reshape back to (x,y,3)
    YIQ = YIQ.transpose(1, 2, 0)
    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    rgb_from_yiq = np.linalg.inv(yiq_from_rgb)

    # from shape (x,y,3) reshape to (3,x,y)
    imgYIQ = imgYIQ.transpose(2, 0, 1)
    # from shape (3,x,y) to (3,x*y)
    img_shape_after_trans = imgYIQ.shape
    imgYIQ = imgYIQ.reshape(3, -1)

    # make the transformation by multiplication
    RGB = np.dot(rgb_from_yiq, imgYIQ)

    # from shape (3,x*y) back to (3,x,y)
    RGB = RGB.reshape(img_shape_after_trans)
    # from shape (3,x,y) reshape back to (x,y,3)
    RGB = RGB.transpose(1, 2, 0)
    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if len(imgOrig.shape) == 3:
        YIQim = transformRGB2YIQ(imgOrig)
        intensityIM = YIQim[:, :, 0].copy()
    else:
        intensityIM = imgOrig.copy()

    intensityIM = (intensityIM * 255).astype('int32')
    # make original vistogram:
    hist, bins_edges = np.histogram(intensityIM, bins=256)

    pdf = hist / sum(hist)
    cdf = np.cumsum(pdf)

    intensity_map = (cdf * 255).astype('int32')

    new_intensityIM = intensityIM.copy()

    for i in range(256):
        new_intensityIM[intensityIM == i] = intensity_map[i]

    new_intensityIM = new_intensityIM.reshape(intensityIM.shape)
    new_intensityIMSCALED = 255 * (new_intensityIM - new_intensityIM.min()) / (
                new_intensityIM.max() - new_intensityIM.min())

    hist_new, bins_edges = np.histogram(new_intensityIMSCALED, bins=256)
    pdf_new = hist_new / sum(hist_new)

    #     print(YIQim)
    new_intensityIMSCALED = (new_intensityIMSCALED / 255).astype('float32')

    if len(imgOrig.shape) == 3:
        #         img = imgOrig.copy()
        YIQim[:, :, 0] = new_intensityIMSCALED
        new_intensityIMSCALED = transformYIQ2RGB(YIQim)

    # Display cumsum
    cumsum = np.cumsum(pdf)
    cumsumEq = np.cumsum(pdf_new)

    #     plt.gray()
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')
    plt.show()

    return new_intensityIMSCALED.astype('float32'), pdf, pdf_new


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 2:  # single channel (grey channel)
        return oneChanelQuantize(imOrig.copy(), nQuant, nIter)

        # transfer img from RGB to YIQ
        # following quantization procedure should only operate on the Y channel
    yiqImg = transformRGB2YIQ(imOrig)
    qImage_, mse = oneChanelQuantize(yiqImg[:, :, 0].copy(), nQuant, nIter)  # y channel = yiqImg[:, :, 0].copy()
    qImage = []
    for img in qImage_:
        # convert the original img back from YIQ to RGB
        qImage_i = transformYIQ2RGB(np.dstack((img, yiqImg[:, :, 1], yiqImg[:, :, 2])))
        qImage.append(qImage_i)

    return qImage, mse


from sklearn.metrics import mean_squared_error


# Quantized an single channel image (grey channel) in to **nQuant** colors
# :param imOrig: The original image (RGB or Gray scale)
# :param nQuant: Number of colors to quantize the image to
# :param nIter: Number of optimization loops
# :return: (List[qImage_i],List[error_i])
def oneChanelQuantize(imOrig: np.ndarray, nQuant: int, nIter: int):
    # to return
    qImages = []
    error_i = []

    imOrig = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imOrig_flat = imOrig.ravel().astype(int)
    histOrg, edges = np.histogram(imOrig_flat, bins=256)
    z_board = np.zeros(nQuant + 1, dtype=int)  # k + 1 boards

    # split the boarder to even parts.
    # the first board is 0 the last board is 255
    for i in range(nQuant + 1):
        z_board[i] = i * (255.0 / nQuant)

    # num of loops
    for i in range(nIter):
        # vector of weighted avg
        x_bar = []
        # calc mean weighted avg for every part
        for j in range(nQuant):
            intense = histOrg[z_board[j]:z_board[j + 1]]
            idx = range(len(intense))  # new arr in len intense
            weightedMean = (intense * idx).sum() / np.sum(intense)
            # add to x_bar mean between two boards
            x_bar.append(z_board[j] + weightedMean)

        qImage_i = np.zeros_like(imOrig)

        # overriding old color and update the mean color for every part
        # there is nQuant means
        for k in range(len(x_bar)):
            qImage_i[imOrig > z_board[k]] = x_bar[k]

        mse = mean_squared_error(imOrig, qImage_i)
        if (len(error_i) > 0) and (mse == error_i[len(error_i) - 1]):
            break;
        else:
            error_i.append(mse)
        qImages.append(qImage_i / 255.0)  # back to range [0, 1]
        for k in range(len(x_bar) - 1):
            z_board[k + 1] = (x_bar[k] + x_bar[k + 1]) / 2  # move (k-2) middle boards -> b_i by x_bar's mean

    return qImages, error_i
