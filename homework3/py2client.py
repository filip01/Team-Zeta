from __future__ import print_function
import numpy as np
import pickle
import socket
import sys
import os
import time

path = os.environ['XDG_RUNTIME_DIR']
server_address = path + '/uds_socket'

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

print('connecting to', server_address)
try:
    sock.connect(server_address)
except socket.error, msg:
    print(msg)
    sys.exit(1)

print('Sending array')
a = np.random.rand(700,700)
data = pickle.dumps(a, protocol=2)
sock.sendall(data)
sock.send('End')
print('Sent!')
time.sleep(1)
x = sock.recv(4096)
sock.close()
print('recv : ', x)
print('Data sent')