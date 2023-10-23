import cv2
import tensorflow as tf
import numpy as np

# Load the four trained models
fire_detection_model = tf.keras.models.load_model('fire_detection_model.h5')
traffic_accident_detection_model = tf.keras.models.load_model('traffic_accident_detection_model.h5')
flood_area_detection_model = tf.keras.models.load_model('flood_area_detection_model.h5')
collapsed_building_detection_model = tf.keras.models.load_model('collapsed_building_detection_model.h5')

# Preprocess the input video
def preprocess_video(video_file, resize_shape=(224, 224), format='MP4', normalize=True):
    """Pre-processes a video file.
    Args:
        video_file (str): The path to the video file.
        resize_shape (tuple, optional): The shape to resize the video to. Default: (224, 224).
        format (str, optional): The format to convert the video to. Default: 'MP4'.
        normalize (bool, optional): Whether to normalize the pixel values. Default: True.

    Returns:
        np.ndarray: The pre-processed video.
    """
    # Load the video file
    video = cv2.VideoCapture(video_file)

    # Resize the video
    video = cv2.resize(video, resize_shape)

    # Convert the video to a specific format
    video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)

    # Normalize the pixel values
    if normalize:
        video = video / 255.0

    return video

# Feed the preprocessed video to each of the four models
fire_predictions = fire_detection_model.predict(video_dataset)
traffic_accident_predictions = traffic_accident_detection_model.predict(video_dataset)
flood_area_predictions = flood_area_detection_model.predict(video_dataset)
collapsed_building_predictions = collapsed_building_detection_model.predict(video_dataset)

# Predict the label of the video for each model
fire_label = 'fire' if fire_predictions[0][0] > 0.5 else 'no fire'
traffic_accident_label = 'traffic accident' if traffic_accident_predictions[0][0] > 0.5 else 'no traffic accident'
flood_area_label = 'flood area' if flood_area_predictions[0][0] > 0.5 else 'no flood area'
collapsed_building_label = 'collapsed building' if collapsed_building_predictions[0][0] > 0.5 else 'no collapsed building'

# If the predicted label is "fire", "traffic_accident", "flood area", or "collapsed building" for any of the models, then the video contains that type of event.
event_label = fire_label or traffic_accident_label or flood_area_label or collapsed_building_label

# Extract the detected part of the video for the event that was detected.
def extract_detected_part_of_video(video, bounding_box):
    """Extracts the detected part of a video.
    Args:
        video (np.ndarray): The video.
        bounding_box (tuple): The bounding box of the detected object.

    Returns:
        np.ndarray: The extracted video.
    """
    # Crop the video to the bounding box
    cropped_video = video[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]

    return cropped_video
# Show the output of the detected part of the video.
import cv2
# Load the detected video
detected_video = cv2.VideoCapture('detected_video.mp4')
# Display the detected video
while detected_video.isOpened():
    ret, frame = detected_video.read()
    if ret:
        cv2.imshow('Detected Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture
detected_video.release()

# Close all windows
cv2.destroyAllWindows()

video_file = 'input_video.mp4'
preprocessed_video = preprocess_video(video_file)

bounding_box = (10, 20, 100, 120)

# Extract the detected part of the video
extracted_video = extract_detected_part_of_video(video, bounding_box)

# Save the extracted video as a new file
cv2.imwrite('extracted_video.mp4', extracted_video)

