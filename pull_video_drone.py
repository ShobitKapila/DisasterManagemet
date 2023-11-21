import dronekit
import cv2
import shutil
vehicle = dronekit.connect('tcp://191.168.1.1')
vehicle.start_video_stream()
cap = cv2.VideoCapture(vehicle.video_feed)
counter = 0
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'frame_{counter}.jpg', frame)
        counter += 1
    else:
        break
shutil.copytree('.', '/path/to/PyCharm/directory')
