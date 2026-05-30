import cv2 # Load OpenCV module
from sys import platform
import numpy as np
# Setting for the camera output resolution. Change this if the window
# open up too small or too large.
CAM_WIDTH = 1280
CAM_HEIGHT = 720

def main():
    # Some setting that make OpenCV startup faster
    if platform == "win32":
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # This loop will run forever
    while True:
        # Fetch a frame image from camera
        success, frame = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Mirror the image
        frame = cv2.flip(frame, 1)
        # Show the frame on a new window called "Camera"
        
        projectionFunction(frame)
        #cv2.imshow("Step 3 Camera", frame)

        # Break the loop if user press Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Make sure OpenCV quit gracefully
    cv2.destroyAllWindows()
    cam.release()

def projectionFunction(frame):
    # If it's a color image, convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ###calculate IPF (mean integral projection function)
    width = gray.shape[1]
    height = gray.shape[0]
    # Calculate vertical projection (sum over rows, along the y-axis)
    mean_vertical_projection = np.sum(gray, axis=0)/ height
    # Calculate horizontal projection (sum over columns, along the x-axis)
    mean_horizontal_projection = np.sum(gray, axis=1)/width
    # Normalize the projections (IPF) to fit on the image
    vertical_projection_normalized = normalize_to_image (gray, mean_vertical_projection, 0)
    horizontal_projection_normalized = normalize_to_image( gray, mean_horizontal_projection,1)
    
    # Vertical derivative of IPF_v (forward difference)
    vertical_IPF_derivative = firstDeriv(vertical_projection_normalized)
    # Horizontal derivative of IPF_h (forward difference)
    horizontal_IPF_derivative = firstDeriv(horizontal_projection_normalized)

    
    ### Calculate Variance projection
    # Vertical Variance Projection (VPF_v)
    vertical_variance_projection = np.zeros(width)
    for x in range(width):
        vertical_variance_projection[x] = (np.sum((gray[:, x] - mean_vertical_projection[x])**2) / height) ** 0.5
    # Horizontal Variance Projection (VPF_h)
    horizontal_variance_projection = np.zeros(height)
    for y in range(height):
        horizontal_variance_projection[y] = (np.sum((gray[y, :] - mean_horizontal_projection[y])**2) / width) ** 0.5

    print ("max horizontal mean {0}, max hori variance {1}".format(np.max(mean_horizontal_projection), np.max(horizontal_variance_projection)))
    # Normalize the variance projections to fit the image size
    vertical_variance_normalized = normalize_to_image(gray, vertical_variance_projection, 0)
    horizontal_variance_normalized = normalize_to_image(gray, horizontal_variance_projection ,1)
    
    # Vertical derivative of VPF_v (forward difference)
    vertical_vpf_derivative = firstDeriv(vertical_variance_normalized)

    # Horizontal derivative of VPF_h (forward difference)
    horizontal_vpf_derivative = firstDeriv(horizontal_variance_normalized)

    ###Calculate generalized projection
    alpha = 0.6
    vertical_generalized_projection = -(1-alpha)* vertical_IPF_derivative  + alpha * vertical_vpf_derivative
    horizontal_generalized_projection = -(1-alpha)* horizontal_IPF_derivative+ alpha * horizontal_vpf_derivative
    vertical_generalized_normalized = normalize_to_image(gray,vertical_generalized_projection, axis = 0)
    horizontal_generalized_normalized = normalize_to_image(gray,horizontal_generalized_projection, axis = 1)
    
    ### Overlay lines
    output_image = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    #vertical component
    output_image = drawLine(output_image,normalize_to_image(gray,vertical_IPF_derivative,0), 0, (100,50,0) )
    output_image = drawLine(output_image,normalize_to_image(gray,vertical_vpf_derivative,1), 0, (100,0,50) )
    output_image = drawLine(output_image,vertical_generalized_normalized, 0, (255,0,0) )
    #horizontal component
    output_image = drawLine(output_image,horizontal_IPF_derivative, 1, (0,100,50) )
    output_image = drawLine(output_image,horizontal_vpf_derivative, 1, (50,100,0) )
    output_image = drawLine(output_image,horizontal_generalized_normalized, 1, (0,255,0) )
    # Display the image using OpenCV
    cv2.imshow('Image with Vertical and Horizontal Projections', output_image)

def normalize_to_image(frame, array, axis):
    ###normalizes values of an np array by height (vertical, axis = 0) or width (horizontal, axis = 1) to be overlayed on top of an image
    array = array/np.max(array)*frame.shape[axis]
    return array
def drawLine(frame, array, axis, color = (0,0,0), lineWidth = 2):
    ### vertical projection axis = 0, horizontal projection axis = 1
    step = 1
    for i in range(len(array)-step):
        if axis == 0:
            value = int(frame.shape[axis] - array[i])
            value_next = int(frame.shape[axis] - array[i+step])
            cv2.line(frame, (i, value), (i+step, value_next), color, lineWidth)
        elif axis ==1:
            value = int(array[i])
            value_next = int(array[i+step])
            cv2.line(frame, (value, i), (value_next, i+step), color, lineWidth)
    return frame

def firstDeriv(array):
    array = np.diff(array, n=1)
    # To keep the same length, append a zero at the end of the derivative array
    array = np.append(array, 0)
    return array
    
if __name__ == '__main__':
    main()