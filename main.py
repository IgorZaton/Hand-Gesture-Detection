import data_manager as dm
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import random


# dm.get_data("yolo", 500)


CATEGORIES = ["openhand", "fist", "peace"]
no_classes = len(CATEGORIES)
data = dm.load_data("/home/igor/PycharmProjects/ComesHandy/data", CATEGORIES)

# print(data[0])

IMG_WIDTH = 50
IMG_HEIGHT = 37
IMG_SIZE = 50


random.shuffle(data)


X = []
y = []

#
# for cell in range(len(data)):
#     np.append(X, data[cell][0])
#     np.append(y, data[cell][1])
#

for cell in data:
    X.append(cell[0])
    y.append(cell[1])

X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
y = np.array(y)

# print(X[1])


X = X/255.0
# print(X[1])

# model = Sequential()
# model.add(Conv2D(4, (3, 3), padding="same", input_shape=X.shape[1:], activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
# model.add(Flatten())
# # model.add(Dense(8))
# model.add(Dense(no_classes))
#
# model.add(Activation("sigmoid"))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=3, epochs=2, validation_split=0.10)

dm.predict_camera_input(model)
