import data_manager as dm
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import random
import os


class ConvNet:

    def __init__(self):
        if os.path.exists('hand_gesture_detection_model'):
            self.model = load_model('hand_gesture_detection_model')
        else:
            CATEGORIES = ["openhand", "fist", "peace", "yolo"]
            no_classes = len(CATEGORIES)
            data = dm.load_data("/home/igor/PycharmProjects/ComesHandy/data", CATEGORIES)

            IMG_WIDTH = 50
            IMG_HEIGHT = 37
            IMG_SIZE = 50

            random.shuffle(data)

            X = []
            y = []

            for cell in data:
                X.append(cell[0])
                y.append(cell[1])

            X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
            y = np.array(y)

            X = X / 255.0

            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(no_classes, activation='softmax'))

            self.model.compile(loss="categorical_crossentropy",
                               optimizer="adam",
                               metrics=['accuracy'])

            self.model.fit(X, y, batch_size=3, epochs=2, validation_split=0.10)

            self.model.save('hand_gesture_detection_model')

    def predict_camera(self, hsv_lower, hsv_upper):
        dm.predict_camera_input(self.model, hsv_lower_values=hsv_lower, hsv_upper_values=hsv_upper)

    def set_camera(self):
        return dm.set_camera()
