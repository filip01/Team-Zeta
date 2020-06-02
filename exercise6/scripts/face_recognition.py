import torch
import torch.nn
import cv2
import numpy as np
import pickle
import socket
import sys
import os
import time
from joblib import load
from PIL import Image
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from torchvision.transforms import ToTensor
from facenet_pytorch import InceptionResnetV1

class face_recognition:
    def __init__(self):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.face_model = load('todo')
        self.hair_model = load('todo')
        self.length_model = load('todo')

        path = os.environ['XDG_RUNTIME_DIR']
        server_address = path + '/uds_socket'

        try:
            os.unlink(server_address)
        except OSError:
        if os.path.exists(server_address):
            raise

        # Create a UDS socket
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Bind the socket to the port
        print('starting up on', server_address)
        self.sock.bind(server_address)

        # Listen for incoming connections
        self.sock.listen()

        while True:
            self.connection, self.client_address = sock.accept()
            print('Connected!')

            packets = []
            while True:
                packet = connection.recv(4096)
                if packet[-3:] == b'End':
                    packets.append(packet[:-3])
                    break
                packets.append(packet)

            data = b"".join(packets)
            img_face, img_features = pickle.loads(data, encoding='latin1')

    def get_embedding(self, img):