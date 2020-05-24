import torch
import torch.nn
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from torchvision.transforms import ToTensor
from PIL import Image

# import pretrained model:
from facenet_pytorch import InceptionResnetV1

'''
image = Image.open(img_path)
image = ToTensor()(image).unsqueeze(0) # unsqueeze to add artificial first dimension
'''

# load data

# load pretrained face recognition model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# calculate embeddings

# setup SVC

# train SVC on embeddings

# test SVC

# plot confusion matrix

# save model