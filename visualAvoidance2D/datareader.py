import cv2
import numpy as np

# Path to the video file
video_path = 'visualAvoidance2D/data/xplane_data/0001/output.mp4'
# Path to the numpy file
npy_path = 'visualAvoidance2D/data/xplane_data/0001/bearing_info.npy'

# Load the numpy file
bearing_info = np.load(npy_path)
# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)
print(bearing_info.shape)
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and display the video frames
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the frame to half its size
    # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Get the frame dimensions
    height, width, _ = frame.shape

    # Calculate the center of the frame
    center_x = width // 2

    # Ensure the frame index is within the bounds of bearing_info
    if frame_index < len(bearing_info):
        bearing = bearing_info[frame_index]
        # Ensure bearing[0][0] is a scalar value
        bearing_angle = np.degrees(float(bearing[0][0]))
        
        print(bearing_angle)
        # Calculate the x position based on the bearing angle and FOV
        x_pos = int(center_x + (bearing_angle / 60.0) * width)
        # Draw a line from top to bottom at the calculated x position
        cv2.line(frame, (x_pos, 0), (x_pos, height), (0, 255, 0), 1)

        bearing_sizes = np.degrees(float(bearing[0][1]))
        bearing_sizes_x_pos = x_pos + int((bearing_sizes / 60.0) * width)
        cv2.line(frame, (bearing_sizes_x_pos, 0), (bearing_sizes_x_pos, height), (0, 0, 255), 1)


    cv2.imshow('Video', frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(0) & 0xFF == 27:  # 27 is the ASCII code for the Esc key
        break

    frame_index += 1

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()