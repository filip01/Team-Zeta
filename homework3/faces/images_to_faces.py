from __future__ import print_function
import numpy as np
import cv2
import sys
import glob
import os

if __name__ == '__main__':
    min_size = (30,30)
    max_size = (60,60) 
    haar_scale = 1.2
    min_neighbors = 3
    haar_flags = 0

    face_cascade = cv2.CascadeClassifier('/opt/ros/kinetic/share/OpenCV-3.3.1-dev/haarcascades/haarcascade_frontalface_default.xml')
    path = os.path.dirname(os.path.realpath(__file__)) + '/'
    for directory in glob.glob(path + '*'):
        index = 0
        dir_name = directory.split('/')
        dir_name = dir_name[len(dir_name)-1]
        print(dir_name, ':')

        for file in glob.glob(directory + '/*'):
            file_name = file.split('/')
            file_name = file_name[len(file_name)-1]
            print(file_name[:4])
            if file_name[:4] == 'face':
                continue

            print(file)
            image = cv2.imread(file)
            # Preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0)

            # Detecting faces
            # todo: add min and max_size parameters
            faces = face_cascade.detectMultiScale(gray, haar_scale, min_neighbors, haar_flags)
            for (x,y,w,h) in faces:
                crop = image[y:y+h, x:x+w]
                cv2.imwrite(directory + '/' + dir_name + '_' + str(index) + '.jpg', crop)
                index += 1