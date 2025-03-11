import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, confusion_matrix

#image size
image_size = 128
labels = ['nonpd', 'pd']

#Load training data
X_train, Y_train = [], []
for label in labels:
    folderPath = os.path.join('/media/rohith/windows/vscode/Parkinson_Disease/parkinson_dataset/train', label)
    for filename in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, filename))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(0 if label == 'nonpd' else 1) 

#Load testing data
X_test, Y_test = [], []
for label in labels:
    folderPath = os.path.join('/media/rohith/windows/vscode/Parkinson_Disease/parkinson_dataset/test', label)
    for filename in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, filename))
        img = cv2.resize(img, (image_size, image_size))
        X_test.append(img)
        Y_test.append(0 if label == 'nonpd' else 1)

# Convert to numpy arrays
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)

#training data
X_train, Y_train = shuffle(X_train, Y_train, random_state=101)

X_train, X_test = X_train.astype('float32') / 255.0, X_test.astype('float32') / 255.0

#CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
history = model.fit(X_train, Y_train, batch_size=32, epochs=15, validation_data=(X_test, Y_test))

#Evaluate model
score, acc = model.evaluate(X_test, Y_test)
print(f"Loss: {(score * 100):.2f}% | Accuracy: {(acc * 100):.2f}%")

#Predict and calculate accuracy
Y_pred = (model.predict(X_test) > 0.5).astype("int32")
precision, recall = precision_score(Y_test, Y_pred), recall_score(Y_test, Y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

#Save trained model
model.save('parkinson.h5')

#Plot Accuracy and Loss
plt.figure(figsize=(14, 7))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper left')
plt.title('Training and Validation Loss')
plt.show()
