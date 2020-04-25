import numpy as np
import cv2
import sys
import glob
import os
from os.path import expanduser
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def calc_hist(image, channel, hist_size):
    hist_range = (0, hist_size)

    # OpenCV function is faster (around 40X) than np.histogram()
    hist = cv2.calcHist([image], [channel], None, [hist_size], hist_range, accumulate=False)

    # normalize histogram
    hist_sum = np.sum(hist)
    hist = np.divide(hist, hist_sum)
    return hist

if __name__ == "__main__":
    # get current path
    path = os.path.dirname(os.path.realpath(__file__)) + '/pictures/'
    colors = ['red', 'green', 'blue', 'yellow', 'white', 'black']
    en_colors = enumerate(colors)
    images = []
    images_color = []
    for c in en_colors:
        p = path + c[1] + '/*'
        imgs = ([cv2.imread(file) for file in glob.glob(p)])
        # images are in BGR format!
        images.extend(imgs)
        num_imgs = len(imgs)
        images_color.extend(num_imgs * [c[0]]) # list of encoded color labels (e.g. 0 for 'red')

    # convert to HSV
    imagesHSV = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

    # HSV ranges :: H: 0-179, S: 0-255, V: 0-255
    # for testing: calc HSV hists for the 1. picture
    # hue_hist = calc_hist(imagesHSV[0], 0, 180)
    # saturation_hist = calc_hist(imagesHSV[0], 1, 256)
    # value_hist = calc_hist(imagesHSV[0], 2, 256)

    # visualize histograms
    # plt.plot(hue_hist)
    # plt.plot(saturation_hist)
    # plt.plot(value_hist)
    # plt.xlim([0,256])
    # plt.title(colors[images_color[0]])
    # plt.show()

    # histogram calculation for every image
    hue_hists = []
    saturation_hists = []
    value_hists = []
    for img in imagesHSV:
        h = np.concatenate(calc_hist(img, 0, 180))
        hue_hists.append(h)
        saturation_hists.append(calc_hist(img, 1, 256))
        value_hists.append(calc_hist(img, 2, 256))
    
    # training based on hue
    X, y = hue_hists, images_color
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    # TODO: research GridSearchCV for additional configuration options
    clf = GridSearchCV(
        SVC(), tuned_parameters
    )
    clf.fit(X_train, y_train)
    print clf.cv_results_