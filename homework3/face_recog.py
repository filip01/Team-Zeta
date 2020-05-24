import torch
import torch.nn
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from torchvision.transforms import ToTensor
from PIL import Image

# import pretrained model:
from facenet_pytorch import InceptionResnetV1

# generate claases
face_classes = []
for i in range(0,20):
    face_classes.append('face_' + str(i))

# load pretrained face recognition model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# load data
path = os.path.dirname(os.path.realpath(__file__)) + '/faces/'

X_train = []
y_train = []
X_test = []
y_test = []

for f_c in face_classes:
    training_path = path + 'training_data/' + f_c + '/*'
    for file in glob.glob(training_path):
        img = Image.open(file)
        img = img.resize((160, 160))
        img = ToTensor()(img)
        # calculate embeddings
        img_embedding = resnet(img.unsqueeze(0))

        X_train.append(img_embedding)
        y_train.append(f_c)

    test_path = path + 'test_data/' + f_c + '/*'
    for file in glob.glob(test_path):
        img = Image.open(file)
        img = img.resize((160, 160))
        img = ToTensor()(img)
        # calculate embeddings
        img_embedding = resnet(img.unsqueeze(0))

        X_test.append(img_embedding)
        y_test.append(f_c)

# setup SVC

# train SVC on embeddings

# test SVC

# plot confusion matrix

# save model