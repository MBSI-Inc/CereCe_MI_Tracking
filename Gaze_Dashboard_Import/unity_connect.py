import socket
import PIL.Image as Image
import numpy as np
import cv2

UNITY_FRAME_PORT = 8052
UNITY_COMMAND_PORT = 8051


def sendValueToUnity(value):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_ip = "127.0.0.1"
    data = bytes(str(value),'utf-8')
    sock.sendto(data, (unity_ip, UNITY_COMMAND_PORT))

def sendToUnity_targetSpeedAndDirection(speed,direction):
    sendValueToUnity("d [{},{}]".format(speed, direction))
    return

def sendToUnity_MoveCommand():
    print("Sent move command to Unity")
    sendValueToUnity("c move")
    return

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# def fetchFrameFromUnity():
#     HOST = "127.0.0.1"  # The server's hostname or IP address
#     PORT = 8052  # The port used by the server

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect((HOST, PORT))
#         s.sendall(b"FETCHFRAME")
#         #data = recvall(s,16)
#         data = s.recv(131072)
#         print(f"Received frame")
#         return data
#         #return bytearray(data)
    
def fetchFrameUnity_byteArray():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', UNITY_FRAME_PORT))
    client_socket.sendall(b"FETCHFRAME")
    image_data = bytearray()
    lengthHeader = client_socket.recv(4)
    #print(lengthHeader)
    length = int.from_bytes(lengthHeader,byteorder='little')
    #print(length)
    #data = client_socket.recvmsg()
    while True:
        #print("receiving packets")
        data = client_socket.recv(1024)
        if not data:
            #print("not data")
            break
        image_data += data
        #print(f"Received {len(data)} bytes ({len(image_data)} total)")
        if len(image_data)>=length:
            break
    #print("returning image data")
    return image_data

def fetchFrameUnity():
    unity_frame_dat = fetchFrameUnity_byteArray()
    unity_frame_array = np.frombuffer(unity_frame_dat, dtype=np.uint8)
    u_frame = cv2.imdecode(unity_frame_array, cv2.IMREAD_UNCHANGED)
    return u_frame

### the fetchFrameUnity returns a non-UMat frame, this converts it to a UMat frame
def fetchFrameUnity_UMat():
    u_frame = fetchFrameUnity()
    print(u_frame.shape)
    u_frame = u_frame[:,:,:3]
    print(u_frame.shape)
    u_frame = cv2.UMat(u_frame)
    return u_frame

def is_unity_connected():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)  # Set timeout to avoid long wait
            sock.connect(("127.0.0.1", UNITY_FRAME_PORT))
        return True  # Connection successful
    except (socket.error, socket.timeout):
        return False  # Connection failed