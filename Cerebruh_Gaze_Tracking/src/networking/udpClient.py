## UDP Client to send data to wheelchair

# Imports and classes
import socket

# Constants
UDP_IP = "192.168.4.1"
UDP_PORT = 4210

class WheelchairClient:
    def __init__(self, inputSock=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create socket object
        self.current_location = (0,0) # Tuple of int to send as the starting location to API
        return

    ###
    # sendMessage function
    # Sends input string to wheelchair via defined UDP host and port.
    ###
    def sendMessage(self, input, host=UDP_IP, port=UDP_PORT): 
        # Console output to show integration
        print(f"Sending input: {input}")
        # Send message (See Wheelchair Integration Team documentation for details)
        self.sock.sendto(input.encode(), (host, port))
