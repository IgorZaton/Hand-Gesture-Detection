import data_manager as dm
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import random


# dm.get_data("phone", 500)


CATEGORIES = ["openhand", "fist"]
data = dm.load_data("/home/igor/PycharmProjects/ComesHandy/data", CATEGORIES)

# print(data[0])

IMG_WIDTH = 21
IMG_HEIGHT = 28
IMG_SIZE = 50

random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# print(X[1])


X = X/255.0
# print(X[1])

model = Sequential()
model.add(Conv2D(4, (3, 3), padding="same", input_shape=X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(8))
model.add(Dense(1))

model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=3, epochs=20, validation_split=0.10)

dm.predict_camera_input(model, IMG_SIZE)
