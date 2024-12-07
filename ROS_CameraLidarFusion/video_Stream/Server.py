import cv2
import socket
import pickle
import struct

# Capture video
cap = cv2.VideoCapture(0)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '168.5.158.86'  # Server's IP address
port = 9000  # Port to listen on
socket_address = (host_ip, port)

# Bind and listen
server_socket.bind(socket_address)
server_socket.listen(5)
print("Listening at:", socket_address)

# Accept connection
client_socket, addr = server_socket.accept()
print('Got Connection from:', addr)

while cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape)
    if not ret:
        break
    # Serialize frame
    data = pickle.dumps(frame)
    # Send message length first
    message = struct.pack("Q", len(data)) + data
    # Then data
    client_socket.sendall(message)

cap.release()
client_socket.close()
