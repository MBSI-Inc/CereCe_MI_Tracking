import socket
import math

class Wheelchair:
    LOCAL_UDP_IP = "192.168.4.11"
    SHARED_UDP_PORT = 3545

    def __init__(self):
        self.wifi_ssid = "ESP32"
        self.wifi_password = "12345678"
        self.BUFFER_SIZE = 2048
        self.LOCAL_UDP_IP = "192.168.4.11"
        self.SHARED_UDP_PORT = 3545
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
        self.sock.bind((self.LOCAL_UDP_IP, self.SHARED_UDP_PORT))
        self.addr = {}

    # this method is used to  establish UDP connection between wheelchair and the laptop
    # the method returns the addr its talking to, if needed for some other function
    # current implementation doesnt use it
    def connect(self):
        while True:
            print("waiting for connection...")
            data, self.addr = self.sock.recvfrom(self.BUFFER_SIZE)
            print(self.addr)
            if self.addr:
                print("sending")
                self.sock.sendto("connected".encode(), self.addr)
                return self.addr

    #  "╔═══════════════════╗",
    #  "║0,4|1,4|2,4|3,4|4,4║",
    #  "║-------------------║",
    #  "║0,3|1,3|2,3|3,3|4,3║",
    #  "║-------------------║",
    #  "║0,2|1,2|2,2|3,2|4,2║",
    #  "║-------------------║",
    #  "║0,1|1,1|2,1|3,1|4,1║",
    #  "║-------------------║",
    #  "║0,0|1,0|2,0|3,0|4,0║",
    #  "╚═══════════════════╝"
    # pick a start coordinate and end coordinate
    # wheelchair will assume its currently at begin coordinate and move towards end coordinate on this grid
    # the unit can be adjusted in the esp32 code.
    def send_coordinate_vector(self, begin, end):
        message = "l" + begin
        self.sock.sendto(message.encode(), self.addr)
        message = "l" + end
        self.sock.sendto(message.encode(), self.addr)
        #self.wait_for_response()

    # move (dist : some units)
    # this method is used to move the wheelchair forward and backward
    def move(self, dist):
        message = "m" + str(dist)
        self.sock.sendto(message.encode(), self.addr)
        #self.wait_for_response()

    # rotate(angle : degrees)
    # method used to rotate the wheelchair by angle in degrees
    # note the wheelchair itself accepts angles in radians, so this method converts angles to radians and sends
    # to wheelchair
    def rotate(self, angle):
        angle = math.radians(angle)
        message = "r" + str(angle)
        self.sock.sendto(message.encode(), self.addr)
        #self.wait_for_response()

    # function used to get response from wheelchair that it has processed the instruction
    def wait_for_response(self):
        data = {}
        while not data:
            data, self.addr = self.sock.recvfrom(self.BUFFER_SIZE)
            if data.decode() != "reached destination":
                if data.decode() == "no path":
                    print("NO PATH FOUND")
                    break
                data = {}
        print("DESTINATION REACHED")

    # method used for debugging udp connection, dont touch
    def routine(self):
        while True:
            angle = float(input("Enter coordinate in the format xxxx: "))
            self.move(90)
            self.rotate(angle)
            #self.wait_for_response()

# this is used for testing
# ignore if you are using the library above
def main():
    conn = Wheelchair()
    addr = conn.connect()
    conn.routine()

#main()