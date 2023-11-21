import cv2
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
import numpy as np

collapsed_building_image_files = os.listdir('collapsed_building')
normal_image_files = os.listdir('normal')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the TensorFlow model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the TensorFlow model on the collapsed building image dataset
collapsed_building_image_dataset = []
for image_file in collapsed_building_image_files:
    image = cv2.imread(f'collapsed_building/{image_file}')
    image = cv2.resize(image, (224, 224))
    collapsed_building_image_dataset.append(image)

collapsed_building_image_dataset = np.array(collapsed_building_image_dataset)
model.fit(collapsed_building_image_dataset, np.ones(len(collapsed_building_image_dataset)), epochs=10)

model.save('collapsed_building_detection_model.h5')

test_image_dataset = []
for image_file in normal_image_files:
    image = cv2.imread(f'normal/{image_file}')
    image = cv2.resize(image, (224, 224))
    test_image_dataset.append(image)

test_image_dataset = np.array(test_image_dataset)

predictions = model.predict(test_image_dataset)

true_labels = [0 if image_file.startswith('normal') else 1 for image_file in normal_image_files]
labels = [0, 1]

confusion_matrix = {}
for label in labels:
    confusion_matrix[label] = {}
    for prediction in labels:
        confusion_matrix[label][prediction] = 0

for i in range(len(true_labels)):
    confusion_matrix[true_labels[i]][predictions[i][0]] += 1

print(confusion_matrix)
