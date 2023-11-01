import tensorflow as tf
import cv2
import numpy as np
from twilio.rest import Client
import os
from urllib.parse import quote

# Set up Twilio credentials
twilio_account_sid = 'AC1f619ff115f7554d37bbc186fe7f45e3'
twilio_auth_token = 'bd19fe33e59e5a4a9d82c7e776c10999'
client = Client(twilio_account_sid, twilio_auth_token)
# Define your phone numbers
sender_phone_number = '+12564019504'
receiver_phone_number = '+91 9985506841'

message_body = 'Emergency detected! View the video footage.'

target_size = (224,224)

# def send_twilio_message(video_path):
#     message = client.messages.create(
#         body=message_body,
#         from_=sender_phone_number,
#         to=receiver_phone_number,
#         media_url=[video_path]
#     )
def send_twilio_message(message_body):
    message = client.messages.create(
        body=message_body,
        from_=sender_phone_number,
        to=receiver_phone_number
    )
def preprocess_frame(frame, target_size):
    # Resize the frame to the target size
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0  # Normalize the pixel values (if your model expects values in [0, 1])
    return frame

collapsed_building_model = tf.keras.models.load_model('collapsed_building_detection_model.h5')
fire_model = tf.keras.models.load_model('fire_detection_model.h5')
flood_model = tf.keras.models.load_model('flood_area_detection_model.h5')
traffic_accident_model = tf.keras.models.load_model('traffic_accident_detection_model.h5')

cap = cv2.VideoCapture(0) # 0 corresponds to the default camera (usually the webcam)

while True:
    ret, frame = cap.read()
    # Preprocess the frame (resize, normalize, etc.) as needed for your models
    preprocessed_frame = preprocess_frame(frame, target_size)
    # Run each model for detection
    is_collapsed_building = collapsed_building_model.predict(np.expand_dims(preprocessed_frame, axis=0))
    is_fire = fire_model.predict(np.expand_dims(preprocessed_frame, axis=0))
    is_flood = flood_model.predict(np.expand_dims(preprocessed_frame, axis=0))
    is_traffic_accident = traffic_accident_model.predict(np.expand_dims(preprocessed_frame, axis=0))

    # Define detection thresholds and actions to take based on detection results
    collapsed_building_threshold = 0.7
    fire_threshold = 0.7
    flood_threshold = 0.7
    traffic_accident_threshold = 0.5

    # Display the result on the frame
    if is_collapsed_building > collapsed_building_threshold:
        x, y, w, h = 100, 100, 200, 200
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Collapsed Building', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite('collapsed_building_detection.jpg', frame)
        # image_path = os.path.abspath('collapsed_building_detection.jpg')
        # encoded_image_path = quote(image_path)
        send_twilio_message("collapsed building accident detected")
    if is_fire > fire_threshold:
        x, y, w, h = 100, 100, 200, 200
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'fire detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite('fire_detection.jpg', frame)
        # image_path = os.path.abspath('fire_detection.jpg')
        # encoded_image_path = quote(image_path)
        send_twilio_message("fire accident detected")

    if is_flood > flood_threshold:
        x, y, w, h = 100, 100, 200, 200
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'flood detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite('flood_detection.jpg', frame)
        # image_path = os.path.abspath('flood_detection.jpg')
        # encoded_image_path = quote(image_path)
        send_twilio_message("flood accident detected")

    if is_traffic_accident > traffic_accident_threshold:
        x, y, w, h = 100, 100, 200, 200
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'traffic accident', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite('traffic_accident_detection.jpg', frame)
        # image_path = os.path.abspath('traffic_accident_detection.jpg')
        # encoded_image_path = quote(image_path)
        send_twilio_message("traffic accident detected")

    # Show the frame with annotations
    cv2.imshow('Live Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()






