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
from PIL import ImageFilter
from PIL import ImageOps

import face_to_hair

# import pretrained model:
from facenet_pytorch import InceptionResnetV1

def train_model(X_train, X_test, y_train, y_test, classes, model, model_name):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model.fit(X_train, y_train)
    
    # test SVC + plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    # save model
    if input('Save model? (y/n)') == 'y':
        dump(model.best_estimator_, model_name + '.joblib')

def main():
    # claases
    hair_lengths = ['short', 'long']
    length_to_num = lambda l: hair_lengths.index(l)

    # load pretrained face recognition model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # load data
    path = os.path.dirname(os.path.realpath(__file__)) + '/faces/'

    face_to_emb = {}
    faces = []
    lengths = []

    for i in range(1,21):
        face = 'face_' + str(i)

        embeddings = []

        length = face_to_hair.length[face]

        faces.append(face)
        lengths.append(length_to_num(length))

        for file in glob.glob(path + face + '/*'):
            img = Image.open(file)
            img = img.resize((160, 160))
            #img2 = img.filter(ImageFilter.GaussianBlur(radius=1))
            img = ToTensor()(img)
            #img2 = ToTensor()(img2)

            # calculate embeddings
            img_embedding = resnet(img.unsqueeze(0))
            embeddings.append(img_embedding.detach().numpy().ravel())

            '''
            img_embedding = resnet(img2.unsqueeze(0))
            embeddings.append(img_embedding.detach().numpy().ravel())
            '''

        face_to_emb[face] = embeddings

    print(lengths)

    get_num_emb = lambda f: len(face_to_emb[f])
    unnest_list = lambda l: [x for sub in l for x in sub]

    # train SVC on embeddings
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    model_SVC = GridSearchCV(SVC(), tuned_parameters, verbose=1)

    # train k-NN on embeddings
    grid_params = {
        'n_neighbors' : [1,3,5,11,19],
        'weights' : ['distance']
    }
    
    model_kNN = GridSearchCV(
        KNeighborsClassifier(),
        grid_params,
        verbose=1
        )

    faces_train, faces_test, _, _ = train_test_split(
        faces, lengths, stratify=lengths, test_size=0.25
    )

    X_train = [f for f_t in faces_train for f in face_to_emb[f_t]]
    X_test = [f for f_t in faces_test for f in face_to_emb[f_t]]

    y_train = [[length_to_num(face_to_hair.length[f])] * get_num_emb(f) for f in faces_train]
    y_train = unnest_list(y_train)
    y_test = [[length_to_num(face_to_hair.length[f])] * get_num_emb(f) for f in faces_test]
    y_test = unnest_list(y_test)

    train_model(X_train, X_test, y_train, y_test, hair_lengths, model_SVC, 'hair_length_recog')
    train_model(X_train, X_test, y_train, y_test, hair_lengths, model_kNN, 'hair_length_recog')

if __name__ == '__main__':
    main()