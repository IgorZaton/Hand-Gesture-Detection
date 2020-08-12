import data_manager
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import random

CATEGORIES = ["openhand", "fist"]
data = data_manager.load_data("/home/igor/PycharmProjects/itComesHandy/data", CATEGORIES)

# print(data[0])

IMG_WIDTH = 21
IMG_HEIGHT = 28
IMG_SIZE = 28

random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
y = np.array(y)

# print(X[1])


X = X/255.0
# print(X[1])

model = Sequential()
model.add(Conv2D(16, (3, 3), padding="same", input_shape=X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(32))
model.add(Dense(1))

model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=3, epochs=2, validation_split=0.10)

data_manager.get_image_from_camera(model, IMG_SIZE)