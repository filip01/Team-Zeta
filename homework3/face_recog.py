import torch
import torch.nn
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from torchvision.transforms import ToTensor
from PIL import Image

# import pretrained model:
from facenet_pytorch import InceptionResnetV1

# generate claases
face_classes = ['face_' + str(i) for i in range(0,21)]

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

        X_train.append(img_embedding.detach().numpy().ravel())
        y_train.append(f_c)

    test_path = path + 'test_data/' + f_c + '/*'
    for file in glob.glob(test_path):
        img = Image.open(file)
        img = img.resize((160, 160))
        img = ToTensor()(img)
        # calculate embeddings
        img_embedding = resnet(img.unsqueeze(0))

        X_test.append(img_embedding.detach().numpy().ravel())
        y_test.append(f_c)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# convert labels to integers
le = preprocessing.LabelEncoder()
le.fit(face_classes)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# train SVC on embeddings
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# test SVC + plot confusion matrix
plot_confusion_matrix(model, X_test, y_test)
plt.show()

# save model