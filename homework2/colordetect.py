import numpy as np
import cv2
import sys
import glob
from os.path import expanduser
from matplotlib import pyplot as plt
from sklearn import svm

def calc_hist(images, channel, hist_size):
    # construct histogram
    hist_range = (0, hist_size)

    # OpenCV function is faster than (around 40X) than np.histogram()
    hist = cv2.calcHist(images, [channel], None, [hist_size], hist_range, accumulate=False)

    # normalize histogram
    hist_sum = np.sum(hist)
    hist = np.divide(hist, hist_sum)
    return hist

if __name__ == "__main__":
    home = expanduser("~")
    images = [cv2.imread(file) for file in glob.glob(home + "/Documents/Colors/*")]
    # images are in BGR format!

    # convert to HSV
    imagesHSV = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

    # HSV ranges :: H: 0-179, S: 0-255, V: 0-255
    hue_hist = calc_hist([imagesHSV[0]], 0, 180)
    saturation_hist = calc_hist([imagesHSV[0]], 1, 256)
    value_hist = calc_hist([imagesHSV[0]], 2, 256)

    # visualize histograms
    plt.plot(hue_hist)
    plt.plot(saturation_hist)
    plt.plot(value_hist)
    plt.xlim([0,256])
    plt.show()
