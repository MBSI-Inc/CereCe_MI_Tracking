import cv2
import numpy as np
from sys import platform
import pandas as pd
import mediapipe as mp
import json
from timeit import default_timer as timer
from facemesh_gazetracker import GazeTracker
import collections
import matplotlib.pyplot as plt




#### Displays real-time blink signal
if __name__ == '__main__':
    nblinks = 0
    ndblinks = 0
    gaze_tracker = GazeTracker("config.json")
    data = None
    plt.ion()
    
    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    x_axis_window = 10
    y_max = 100
    y_min = -100
    x = [0,x_axis_window]
    y = [y_min,y_max]   
    x_axis_start = 0
    x_axis_end = x_axis_window
    line1, = ax.plot(x, y)
    line2, = ax.plot(x, y)
    #line2, = ax.plot(x,[e+2 for e in y])
    


    # setting title
    plt.title("Left eye blink signal", fontsize=20)
    
    # setting x-axis label and y-axis label
    plt.xlabel("time (s)")
    plt.ylabel("% Eye_slit_h / nosebridge_h")
    
    start_time = 0
    end_time = 0
    start_2 = 0
    end_2 = 0
    temp = 0
    blinkwindow = 0.2
    double_blink_window = 0.5
    blinklabel = ax.text(0.85, 0.9, "Blinks: "+str(nblinks), transform=ax.transAxes, fontsize=12, ha='center')
    doubleblink = ax.text(0.85, 0.85, "Double Blinks: "+str(ndblinks), transform=ax.transAxes, fontsize=12, ha='center')

    while True:
        raw_data = gaze_tracker.get_gaze_raw_data_from_camera()
        if raw_data == None or raw_data["has_data"]==False:
            continue
        frame = raw_data["frame"]
        cv2.imshow("Main", frame)
        timestamp = raw_data["frame_timestamp"] 
        eyeslit_h_r = raw_data["eyeslit_h_r"] 
        eyeslit_h_l = raw_data["eyeslit_h_l"] 
        nosebridge_h = raw_data["nosebridge_h"]
        ratio_r = eyeslit_h_r/nosebridge_h *100.0
        ratio_l = eyeslit_h_l/nosebridge_h *100.0
        #print("ratio",ratio_r)
        first_derivative = 0
        row_data = [timestamp, ratio_l,ratio_r, eyeslit_h_l,eyeslit_h_r,nosebridge_h,first_derivative]
        if data is None:
            data = pd.DataFrame([row_data])
        else:
            # Append the new row to the DataFrame
            data=pd.concat([data, pd.DataFrame([row_data])], ignore_index=True)
        
        if len(data)>=2:
            data.iat[-1,-1] = (data.iat[-1,1]-data.iat[-2,1])/(data.iat[-1,0]-data.iat[-2,0])

        


        # Set the maximum number of rows you want to index (e.g., 10)
        max_rows_n = 500
        # Calculate the number of rows in the DataFrame
        total_rows = len(data)
        # Index at most the last 10 rows
        start_index = max(0, total_rows - max_rows_n)
        data_window = data.iloc[start_index:]
        x = data_window[0].tolist()
        y = data_window[1].tolist()

        if timestamp>x_axis_window-2:
            #print("Axis should update now!")
            x_axis_end = timestamp+2
            x_axis_start = x_axis_end-x_axis_window
            plt.axis([x_axis_start, x_axis_end, y_min, y_max])
        
        dydx = data_window.iloc[:, -1].tolist()
        
        
        if dydx[-1]<(-50):
            start_time = timestamp
            
        if dydx[-1]>(50):
            end_time = timestamp
            
            elapsed_time = end_time-start_time
            
            if elapsed_time<blinkwindow:
                blinklabel.remove()
                nblinks+=1
                
                if temp!=nblinks:
                    end_2 = timestamp
                    elapsed_2 = end_2-start_2

                    if elapsed_2<double_blink_window:
                        doubleblink.remove()
                        ndblinks+=1
                        doubleblink = ax.text(0.85, 0.85, "Double Blinks: "+str(ndblinks), transform=ax.transAxes, fontsize=12, ha='center')

                start_2 = timestamp
                temp = nblinks
                blinklabel = ax.text(0.85, 0.9, "Blinks: "+str(nblinks), transform=ax.transAxes, fontsize=12, ha='center')
                
                

        #ax.clear()
        line1.set_xdata(x)
        line1.set_ydata(y)
        line2.set_xdata(x)
        line2.set_ydata(dydx)
        
        # line2.set_xdata(x)
        # line2.set_ydata([e+2 for e in y])
        # drawing updated values
        figure.canvas.draw()
        

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        # figure.canvas.flush_events()


        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    gaze_tracker.deinit()
