from __future__ import print_function
import numpy as np
import pickle
import socket
import sys
import os
import time

path = os.environ['XDG_RUNTIME_DIR']
server_address = path + '/uds_socket'

try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Bind the socket to the port
print('starting up on', server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen()

connection, client_address = sock.accept()
print('Connected!')

packets = []
while True:
    packet = connection.recv(4096)
    if packet[-3:] == b'End':
        packets.append(packet[:-3])
        break
    packets.append(packet)

connection.send(b'Kappa')
data = b"".join(packets)

a = pickle.loads(data, encoding='latin1')
print(a.shape)
connection.close()
sock.close()
os.unlink(server_address)