## TCP Client to obtain gaze data from GazePointer

# Imports and classes
import socket
import re
from networking.networking_helpers import byteSendString

# Constants
RESULT_FORMAT = "xml"
APP_KEY = "AppKeyTrial"

class GazeClient:
    def __init__(self, inputSock=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Create socket object
        self.connected = False
        return

    ###
    # connect function
    # Connect the defined socket to localhost (127.0.0.1) with port 43333, as defined in GazePointer samples
    # It serves as a TCP client that connects to GazePointer program (which is the corresponding server)
    ###
    def connect(self, host="127.0.0.1", port=43333): 
        # Establish socket connection
        print(self.sock)
        self.sock.connect((host, port))

        # Define response type and specs
        # Send ResultFormat as XML in Byte array
        bResultFormat = bytes(RESULT_FORMAT, encoding='utf-8')
        self.sock.sendall(bResultFormat)

        byteSendString(self.sock, APP_KEY)

        # Get initial connection details from server
        for i in range(5):  # Retry 5 times max
            data = self.sock.recv(1024)
            messageString = data.decode('utf-8')
            print(f"Received data: {messageString}")
            if messageString[0:2] == 'ok':
                # ok message means connection success, break loop and return true
                self.connected = True
                return True
        # TODO: Handling of failed connections

    def getGazeData(self):
        if (not self.connected):
            return None

        # Receive stream data from socket
        streamData = self.sock.recv(1024)

        # Splitting stream data because GazePointer would sometime return multiple GazeData object in the same stream
        # Delimited by 2 bytes '<any byte>\x02'
        splitStreamData = streamData.split(b'\x02')
        # Expected outcomes: 
        #   [b'\xA8' (or any random byte), b'<xml ....  </GazeData>'] <- When there is one object returned
        #   [b'\xA8' (or any random byte), b'<xml ....  </GazeData>\xA8', b'<xml .... </GazeData>'] <- When there are > 1 objects returned

        try:
            if (len(splitStreamData) > 1): 
                # Using try catch to handle the two scenarios upon splitting the stream
                # Note: We're not choosing to use splitStreamData[-1] despite it never needing to handle the last byte.
                #       This is because there is a chance that the last object is not returned fully (i.e. missing some part of the xml) due to packet size limit
                try:
                    message = splitStreamData[2].decode('utf-8') # Decoding from byteArray to string
                except:
                    message = splitStreamData[2][:-1].decode('utf-8')

                # Cleaner attempt using dictionaries and regex
                data = [float(x[1:(len(x)-1)]) for x
                        in re.findall(r">[-]?(?:\d*\.\d+|\d+)<", message)]
                LABELS = [x[1:(len(x)-1)] for x
                        in re.findall(r"<\w+>", message)]
                store = {}
                for i in range(len(data)):
                    store[LABELS[i]] = data[i]
                return store

        except Exception as err:
            print(err)
