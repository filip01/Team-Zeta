import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import seaborn as sn
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier

def calc_hist(image, channel, hist_size):
    hist_range = (0, hist_size)

    # OpenCV function is faster (around 40X) than np.histogram()
    hist = cv2.calcHist([image], [channel], None, [hist_size], hist_range, accumulate=False)

    # normalize histogram
    hist_sum = np.sum(hist)
    hist = np.divide(hist, hist_sum)
    return hist

def draw_conf_matrix(conf_mat, title):
    df_cm = DataFrame(conf_mat, index=colors, columns=colors)
    ax = sn.heatmap(df_cm, cmap="YlGnBu", annot=True, fmt="d")
    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    # get current path
    path = os.path.dirname(os.path.realpath(__file__)) + '/pictures/'
    colors = ['red', 'green', 'blue', 'yellow', 'white', 'black']
    en_colors = enumerate(colors)
    images_color = []
    hue_hists = []
    saturation_hists = []
    value_hists = []

    rgbhists=[]
    hsvHists = []

    plt.close('all')

    for c in en_colors:
        p = path + c[1] + '/*'
        num_imgs = 0
        # imgs = ([cv2.imread(file) for file in glob.glob(p)])
        for file in glob.glob(p):
        
            img = cv2.imread(file)
            # images are in BGR format!
            # convert to HSV
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
            print num_imgs
            # histogram calculation
            # HSV ranges :: H: 0-179, S: 0-255, V: 0-255
            hh = np.concatenate(calc_hist(imgHSV, 0, 180)) 
            hue_hists.append(hh)
            sh = np.concatenate(calc_hist(imgHSV, 1, 256))
            saturation_hists.append(sh)
            vh = np.concatenate(calc_hist(imgHSV, 2, 256))
            value_hists.append(vh)
               
            hsvHists.append(np.concatenate([hh,sh,vh])) 

            R= np.concatenate(calc_hist(imgRGB, 0, 256)) 
            G= np.concatenate(calc_hist(imgRGB, 1, 256)) 
            B= np.concatenate(calc_hist(imgRGB, 2, 256)) 

            rgbhists.append(np.concatenate([R,G,B])) 
            

            # TODO: check model perf. for alternative number of bins

            num_imgs += 1

        images_color.extend(num_imgs * [c[0]]) # list of encoded color labels (e.g. 0 for 'red')
   
    # visualize histograms
    #plt.plot(hue_hists[0])
    #plt.plot(saturation_hists[0])
    #plt.plot(value_hists[0])
    #plt.xlim([0,256])
    #plt.title(colors[images_color[0]])
    #plt.show()

    # training based on hue
    X, y = hue_hists, images_color
    
    Xrgb = rgbhists
    Xhsv = hsvHists
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

    X_trainRGB, X_testRGB, y_trainRGB, y_testRGB = train_test_split(Xrgb, y, test_size=0.4, stratify=y)

    X_trainHSV, X_testHSV, y_trainHSV, y_testHSV = train_test_split(Xhsv, y, test_size=0.4, stratify=y)

    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    # TODO: research GridSearchCV for additional configuration options
    # TODO: determine score function
    clf = GridSearchCV(
        SVC(), tuned_parameters
    )
    knn = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X_train, y_train)
    print 'Best parameters: ', clf.best_params_


    knn.fit(X_train, y_train)
    
    
    # model perf. on test data
    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    draw_conf_matrix(conf_mat, 'scv')


    y_predKnn = knn.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_predKnn)
    draw_conf_matrix(conf_mat, 'knn')

    clf.fit(X_trainRGB, y_trainRGB)

    y_predRGB = clf.predict(X_testRGB)
    conf_mat = confusion_matrix(y_testRGB, y_predRGB)
    draw_conf_matrix(conf_mat, 'scv: RGB')

    clf.fit(X_trainHSV, y_trainHSV)

    y_predHSV = clf.predict(X_testHSV)
    conf_mat = confusion_matrix(y_testHSV,y_predHSV)
    draw_conf_matrix(conf_mat, 'scv: HSV all')


    # save best model to file
    # dump(clf.best_estimator_, 'svc.joblib')

    # TODO: training/testing based on other data derived from HSV (+RGB and maybe other stuff?)
    # TODO: try another classifier
   