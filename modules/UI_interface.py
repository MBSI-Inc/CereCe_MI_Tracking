import tkinter as tk
from Bluetooth_UltrasonicSensor import ultrasonic_result
import asyncio
import threading
# --- SETTINGS ---
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 300

#How to integrate: Must take in an input of distance & the respective sensor direction.

# Simulated distance (replace this with your variable)
distance = [120]  #['left','right','back'] -> Change to three repeats & values later when you have 3 modules
#To test variations use 30,70,120.
repeats = 1;

# --- CREATE WINDOW ---
root = tk.Tk()
root.title("Ultrasonic Parking Display")

canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

# --- DRAW CAR --- #To replace with an image of a wheelchair 
car_x1 = 50
car_x2 = 110
car_y1 = 120
car_y2 = 240

canvas.create_rectangle(car_x1, car_y1, car_x2, car_y2, fill="black")

# --- COLOR LOGIC ---
def get_color(distance): #Returns a matrix of colours based on the distances
    colours = []
    for lengths in range(repeats):
        if  distance[lengths]> 100:
            colours.append("green") 
        elif distance[lengths] > 50:
            colours.append("yellow")
        else:
            colours.append("red")
    return colours #(e.g. ["red","red","red"])

def get_wave_number(distance): #Return a matrix of wave number
    wave_number = []
    for lengths in range(repeats):
        if distance[lengths] > 100:
            wave_number.append(3)
        elif distance[lengths] > 50:
            wave_number.append(2)
        else:
            wave_number.append(1)
    return wave_number 

# --- DRAW WAVES FUNCTION ---
def draw_waves(distance):
    canvas.delete("waves")  # clear old waves
    color = get_color(distance)
    waves = get_wave_number(distance)
    
    start_x = (car_x2 + car_x1)//2
    start_y = (car_y1 + car_y2) // 2
    angle = [300,120,210]
    
    for i in range(repeats): #Draw arc that depends on number of waves and color for left, right and back
     for wave_number in range(waves[i]): #Draw arcs 
        offset = wave_number * 15
        x_position_start = [start_x + offset,start_x - offset, start_x - offset - 40]
        y_position_start = [start_y - 40 - offset, start_y - 40 - offset, start_y + offset]
        x_position_end = [start_x + 60 + offset, start_x - 60 - offset, start_x + offset + 40]
        y_position_end = [start_y+ 40 + offset, start_y+ 40 + offset, start_y + offset + 60]
        canvas.create_arc( #Defines the rectangle that contains the waves
        x_position_start[i], #Starting x
        y_position_start[i], #Starting y
        x_position_end[i], #Ending x
        y_position_end[i], #Ending y
        start=angle[i], #Starts at 300 degrees (Based on unit circle)
        extent=120, #Anticlockwise extension
        style="arc",
        outline=color[i],
        width=3,
        tags="waves",
        )

# --- UPDATE LOOP ---
def on_ble_update(LHS, RHS):
    global distance #update distance variable
    avg = (LHS + RHS) / 2
    distance = [avg]
    root.after(0, draw_waves, distance) #runs draw_waves on main thread

def start_ble_listener():
    asyncio.run(ultrasonic_result(on_ble_update)) #whenever an update occur, on_ble_update will be run

#By adding a thread, you allow things to occur simultaneously
#main thread is Tkinter window, ble thread send BLE data
ble_thread = threading.Thread(target=start_ble_listener, daemon=True)
ble_thread.start()

root.mainloop()

if __name__ == "__main__":
    run_display()

#if you want to use this in main.py
#import UI_interface
#UI_interface.run_display()
