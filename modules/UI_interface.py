import tkinter as tk
from itertools import cycle
from PIL import Image, ImageTk
from Bluetooth_UltrasonicSensor import ultrasonic_result
import asyncio
import threading
# --- SETTINGS ---
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

#How to integrate: Must take in an input of distance & the respective sensor direction.

# Simulated distance (replace this with your variable)
distance = [30,70,120]  #['left','right','back'] -> Change to three repeats & values later when you have 3 modules
#To test variations use 30,70,120.
repeats = 3;

# --- CREATE WINDOW ---
root = tk.Tk()
root.title("Ultrasonic Parking Display")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.resizable(False, False)
root.attributes("-topmost", True)  # Forces window to appear over everything

canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

# --- ANIMATION ---
DELAY = 100  # milliseconds

def animate():
  canvas.itemconfig(img_on_canvas, image=next(frames_cycle))
  root.after(DELAY, animate)

# Load GIF using Pillow
gif_image = Image.open("chair_gif.gif")
frames = []

# the gif is too big
NEW_SIZE = (200, 200) 

for frame_index in range(gif_image.n_frames):
  gif_image.seek(frame_index)
  # Convert frame to RGBA to preserve transparency channels
  rgba_frame = gif_image.convert("RGBA")
  resized_frame = rgba_frame.resize(NEW_SIZE, Image.Resampling.LANCZOS)
  photo_img = ImageTk.PhotoImage(resized_frame)
  frames.append(photo_img)

frames_cycle = cycle(frames)

img_on_canvas = canvas.create_image(
    200, 185, anchor="center", image=next(frames_cycle) # its slightly off-centre because the animation goes higher than it goes lower
)

animate()

# --- COLOR LOGIC ---
def get_color(distance): #Returns a matrix of colours based on the distances
    colours = []
    for lengths in range(repeats):
        if  distance[lengths] > 70:
            colours.append("#00B38A")  # green
        elif distance[lengths] > 40:
            colours.append("#F2AC42")  # yellow
        else:
            colours.append("#EA324C")  # red
    return colours #(e.g. ["red","red","red"])

def get_wave_number(distance): #Return a matrix of wave number
    wave_number = []
    for lengths in range(repeats):
        if distance[lengths] > 70:
            wave_number.append(3)
        elif distance[lengths] > 40:
            wave_number.append(2)
        else:
            wave_number.append(1)
    return wave_number 

# --- DRAW WAVES FUNCTION ---
def draw_waves(distance):
    canvas.delete("waves")  # clear old waves
    color = get_color(distance)
    waves = get_wave_number(distance)
    
    start_x = WINDOW_WIDTH // 2
    start_y = WINDOW_HEIGHT // 2
    angle = [300,120,210]
    
    for i in range(repeats): #Draw arc that depends on number of waves and color for left, right and back
     for wave_number in range(waves[i]): #Draw arcs 
        offset = wave_number * 15
        space_1  = 70
        space_2  = 90
        x_position_start = [start_x + offset,start_x - offset, start_x - offset - space_1]
        y_position_start = [start_y - space_1 - offset, start_y - space_1 - offset, start_y + offset]
        x_position_end = [start_x + space_2 + offset, start_x - space_2 - offset, start_x + offset + space_1]
        y_position_end = [start_y+ space_1 + offset, start_y+ space_1 + offset, start_y + offset + space_2]
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
#haven't exactly changed this to work for many modules
def on_ble_update(LHS, RHS):
    global distance #update distance variable
    distance = [LHS,0, RHS]
    root.after(0, draw_waves, distance) #runs draw_waves on main thread

def start_ble_listener():
    asyncio.run(ultrasonic_result(on_ble_update)) #whenever an update occur, on_ble_update will be run

#By adding a thread, you allow things to occur simultaneously
#main thread is Tkinter window, ble thread send BLE data
ble_thread = threading.Thread(target=start_ble_listener, daemon=True)
ble_thread.start()

draw_waves(distance)  # Initial call to draw_waves

root.mainloop()
#if you want to use this in main.py
#import UI_interface
#UI_interface.run_display()
