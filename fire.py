# importing librarires
import cv2
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the fire and normal images
fire_image_files = os.listdir('fire')
normal_image_files = os.listdir('normal')

# Create a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the TensorFlow model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the TensorFlow model
train_image_dataset = []
for image_file in fire_image_files:
    image = cv2.imread(f'fire/{image_file}')
    image = cv2.resize(image, (224, 224))
    train_image_dataset.append(image)

train_image_dataset = np.array(train_image_dataset)
model.fit(train_image_dataset, np.ones(len(train_image_dataset)), epochs=10)

# Save the TensorFlow model
model.save('fire_detection_model.h5')

# Load the trained TensorFlow model
model = tf.keras.models.load_model('fire_detection_model.h5')

# Make predictions on the test image dataset
test_image_dataset = []
for image_file in normal_image_files:
    image = cv2.imread(f'normal/{image_file}')
    image = cv2.resize(image, (224, 224))
    test_image_dataset.append(image)

test_image_dataset = np.array(test_image_dataset)
predictions = model.predict(test_image_dataset)

# Calculate the confusion matrix
true_labels = [0 if image_file.startswith('normal') else 1 for image_file in normal_image_files]
labels = [0, 1]

# Calculate the confusion matrix using a dictionary
confusion_matrix = {}
for label in labels:
    confusion_matrix[label] = {}
    for prediction in labels:
        confusion_matrix[label][prediction] = 0

for i in range(len(true_labels)):
    confusion_matrix[true_labels[i]][predictions[i][0]] += 1

# Print the confusion matrix
print(confusion_matrix)



# Capture the video stream
# video_stream = cv2.VideoCapture(0)
# # Loop over the video frames
# while True:
#     # Read the next video frame
#     ret, frame = video_stream.read()

#     # If the video frame is empty, break the loop
#     if not ret:
#         break

#     # Resize the video frame to the model input size
#     frame = cv2.resize(frame, (224, 224))

#     # Make a prediction on the video frame
#     prediction = model.predict(np.array([frame]))

#     # If the prediction is greater than 0.5, then the video frame is classified as fire
#     if prediction[0] > 0.5:
#         print('Fire detected!')

# # Release the video stream
# video_stream.release()
