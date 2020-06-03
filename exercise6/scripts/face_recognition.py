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

        ml_path = os.path.dirname(os.path.realpath(__file__)) + '/ml_models/'
        self.face_model = load(ml_path + 'face_model.joblib')
        self.color_model = load(ml_path + 'hair_color_recog.joblib')
        self.length_model = load(ml_path + 'hair_length_recog.joblib')

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
            connection, client_address = self.sock.accept()
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
            face_emb = self.get_embedding(img_face)
            feature_emb = self.get_embedding(img_features)

            # perform recognition
            face_id = self.recognition(self.face_model, feature_emb)
            color = self.recognition(self.color_model, feature_emb)
            length = self.recognition(self.length_model, feature_emb)

            # pack results into a list
            result = [face_id[0], color[0], length[0]]
            data = str.encode(' '.join(str(x) for x in result))

            # send result back
            connection.send(data)

            # close connection
            # connection.shutdown()
            connection.close()

    def get_embedding(self, img_cv):
        # convert cv2 image to PIL Image
        img = Image.fromarray(img_cv)

        img = img.resize((160, 160))
        img.show()
        img = ToTensor()(img)

        # calculate embeddings
        emb = self.resnet(img.unsqueeze(0))
        return [emb.detach().numpy().ravel()]

    def recognition(self, model, emb):
        return model.predict(emb)

if __name__ == '__main__':
    face_recognition()