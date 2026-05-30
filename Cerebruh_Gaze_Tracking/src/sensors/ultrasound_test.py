import serial
import time
from  ultrasoundReader import ultrasound

# Replace with your actual port
port = '/dev/cu.usbmodem11301'  # ← Update this!
usreader = ultrasound(port = port, baud_rate = 9600)

try:
    while True:
        distance = usreader.get_distance()
        if distance is not None:
            print(f"Distance: {distance:.2f} cm")
        else:
            print("Distance none")    
            
except KeyboardInterrupt:
    print("\nStopped by user")
