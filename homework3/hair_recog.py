import torch
import torch.nn
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from joblib import dump, load

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
faces = []
colors = []
lengths = []

for i in range(1,21):
    face = 'face_' + str(i)

    color = face_to_hair.color[face]
    length = face_to_hair.length[face]

    color_to_face[color].append(face)
    length_to_face[length].append(face)
    embeddings = []

    faces.append(face)
    colors.append(color)
    lengths.append(length)

    for file in glob.glob(path + face + '/*'):
        img = Image.open(file)
        img = img.resize((160, 160))
        img = ToTensor()(img)

        # calculate embeddings
        img_embedding = resnet(img.unsqueeze(0))

        embeddings.append(img_embedding.detach().numpy().ravel())

    face_to_emb[face] = embeddings

def train_model(X_train, X_test, y_train, y_test, classes, model_name):
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
    '''
    print('training SVC')
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    model = GridSearchCV(SVC(), tuned_parameters, verbose=1)
    '''

    grid_params = {
        'n_neighbors' : [1,3,5,11,19],
        'weights' : ['uniform', 'distance']
    }
    
    model = GridSearchCV(
        KNeighborsClassifier(),
        grid_params,
        verbose=1
        )

    model.fit(X_train, y_train)

    # test SVC + plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    # save model

faces_train, faces_test, _, _ = train_test_split(
    faces, colors, stratify=colors
)

get_num_emb = lambda f: len(face_to_emb[f])

X_train = [face_to_emb[f] for f in faces_train]
X_train = [x for sub in X_train for x in sub]
X_test = [face_to_emb[f] for f in faces_test]
X_test = [x for sub in X_test for x in sub]

y_train = [[face_to_hair.color[f]] * get_num_emb(f) for f in faces_train]
'''
print('Sanity check:')
for i in range(0, len(faces_train)):
    print(faces_train[i], ':', y_train[i])
'''
y_train = [y for sub in y_train for y in sub]
y_test = [[face_to_hair.color[f]] * get_num_emb(f) for f in faces_test]
y_test = [y for sub in y_test for y in sub]

train_model(X_train, X_test, y_train, y_test, hair_colors, 'hair_color_recog')

# =hair length=