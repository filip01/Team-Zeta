import numpy as np
import cv2
import sys
from os.path import expanduser
from matplotlib import pyplot as plt
from sklearn import svm

def calc_hist(img, channel, hist_range):
    # construct histogram
    hist_size = 256
    hist = cv2.calcHist(imgHSV, [channel], None, [hist_size], hist_range, accumulate=False)

    # normalize histogram
    hist_sum = np.sum(hist)
    hist = np.divide(hist, hist_sum)
    return hist

if __name__ == "__main__":
    home = expanduser("~")
    img = cv2.imread(home + '/Documents/test.jpg') # TODO: set general path

    # convert to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue_hist = calc_hist(imgHSV, 0, (0, 180))
    saturation_hist = calc_hist(imgHSV, 1, (0, 256))
    value_hist = calc_hist(imgHSV, 2, (0, 256))

    # visualize histograms
    plt.plot(hue_hist)
    plt.plot(saturation_hist)
    plt.plot(value_hist)
    plt.xlim([0,256])
    plt.show()
