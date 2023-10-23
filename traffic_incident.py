import cv2
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the traffic accident and normal image datasets
accident_image_files = os.listdir('traffic_incidents')
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

# Train the TensorFlow model on the traffic accident image dataset
accident_image_dataset = []
for image_file in accident_image_files:
    image = cv2.imread(f'traffic_incidents/{image_file}')
    image = cv2.resize(image, (224, 224))
    accident_image_dataset.append(image)

accident_image_dataset = np.array(accident_image_dataset)
model.fit(accident_image_dataset, np.ones(len(accident_image_dataset)), epochs=10)

# Save the trained model
model.save('traffic_accident_detection_model.h5')

# Load the trained model and resize the test images to the same size as the input shape of the model
test_image_dataset = []
for image_file in normal_image_files:
    image = cv2.imread(f'normal/{image_file}')
    image = cv2.resize(image, (224, 224))
    test_image_dataset.append(image)

test_image_dataset = np.array(test_image_dataset)

# Predict the labels for the test images
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
