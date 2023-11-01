import dronekit
import cv2
import shutil
# Connect to the drone
vehicle = dronekit.connect('tcp://191.168.1.1')
# Start the video stream
vehicle.start_video_stream()
# Capture the video frames
cap = cv2.VideoCapture(vehicle.video_feed)
# Save the video frames to disk
counter = 0
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'frame_{counter}.jpg', frame)
        counter += 1
    else:
        break
# Copy the video frames to the PyCharm directory
shutil.copytree('.', '/path/to/PyCharm/directory')
