import struct
import time
import datetime

def byteSendString(socket, string):
    # 1. Start by encoding the string into byte array
    bString = bytes(string, encoding='utf-8')
    # print(struct.pack('B', len(bString)))

    # 2. Send the byte array length as a separate message to stream
    socket.sendall(struct.pack('B', len(bString)))
    # print(struct.pack(f'{len(bString)}s', bString))

    # 3. Send the actual string formatted with struct pack
    socket.sendall(struct.pack(f'{len(bString)}s', bString))

# Directly countdown to buffer input
def countdown(h, m, s):
 
    # Calculate the total number of seconds
    total_seconds = h * 3600 + m * 60 + s
 
    # While loop that checks if total_seconds reaches zero
    # If not zero, decrement total time by one second
    while total_seconds > 0:
 
        # Timer represents time left on countdown
        timer = datetime.timedelta(seconds = total_seconds)
        
        # Prints the time left on the timer
        # print(timer, end="\r")
 
        # Delays the program one second
        time.sleep(1)
 
        # Reduces total time by one second
        total_seconds -= 1
 
    # print("Bzzzt! The countdown is at zero seconds!")

