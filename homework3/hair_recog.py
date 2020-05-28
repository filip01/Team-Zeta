import torch
import torch.nn
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from torchvision.transforms import ToTensor
from PIL import Image

import face_to_hair

# import pretrained model:
from facenet_pytorch import InceptionResnetV1

# claases
hair_colors = ['light', 'dark']
hair_lengths = ['short', 'long']

# load pretrained face recognition model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# load data
path = os.path.dirname(os.path.realpath(__file__)) + '/faces/'

face_to_emb = {}
color_to_face = {'light' : [], 'dark' : []}
length_to_face = {'long' : [], 'short' : []}

for i in range(1,21):
    face = 'face_' + str(i)

    color = face_to_hair.color[face]
    length = face_to_hair.length[face]

    color_to_face[color].append(face)
    length_to_face[length].append(face)
    embeddings = []

    for file in glob.glob(path + face + '/*'):
        img = Image.open(file)
        img = img.resize((160, 160))
        img = ToTensor()(img)

        # calculate embeddings
        img_embedding = resnet(img.unsqueeze(0))

        embeddings.append(img_embedding.detach().numpy().ravel())

    face_to_emb[face] = embeddings

def train_SVC(X_train, X_test, y_train, y_test, classes, model_name):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # convert labels to integers
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # train SVC on embeddings
    # model = SVC(kernel='linear')
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # test SVC + plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    # save model

get_num_emb = lambda f: len(face_to_emb[f])

# =hair color=
num_l = len(color_to_face['light'])
num_d = len(color_to_face['dark'])
L = math.ceil(num_l / 2) 
D = math.ceil(num_d / 2)

# data split
face_train = color_to_face['light'][:L] + color_to_face['dark'][:D]
face_test = color_to_face['light'][L:] + color_to_face['dark'][D:]

X_train = [face_to_emb[f] for f in face_train]
X_train = [x for sub in X_train for x in sub]
X_test = [face_to_emb[f] for f in face_test]
X_test = [x for sub in X_test for x in sub]

y_train = [[face_to_hair.color[f]] * get_num_emb(f) for f in face_train]
y_train = [y for sub in y_train for y in sub]
y_test = [[face_to_hair.color[f]] * get_num_emb(f) for f in face_test]
y_test = [y for sub in y_test for y in sub]

train_SVC(X_train, X_test, y_train, y_test, hair_colors, 'hair_color_recog')

# =hair length=