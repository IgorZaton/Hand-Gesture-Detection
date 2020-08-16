import cv2
from imutils.video import VideoStream
import time
import imutils
import os
import SlidersC
import numpy as np
import copy


def get_data(dset_name, imgnum=500, time_between_shots=0.1, IMG_SIZE=50):
    vs = VideoStream(src=0).start()
    time.sleep(3.0)
    sliders = SlidersC.Sliders()

    while sliders.is_started() is False:
        frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=600, height=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        sliders.update_slider()
        mask = cv2.inRange(hsv, sliders.get_lower_values(), sliders.get_upper_values())
        mask = cv2.erode(mask, (3, 3), iterations=2)
        mask = cv2.dilate(mask, (3, 3), iterations=10)
        mask = imutils.resize(mask, width=IMG_SIZE, height=IMG_SIZE)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            exit()

    time.sleep(2.0)
    print("[INFO] Data acquisition started.")

    for i in range(imgnum):
            frame = vs.read()

            if frame is None:
                break

            frame = imutils.resize(frame, width=600, height=600)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, sliders.get_lower_values(), sliders.get_upper_values())
            mask = cv2.erode(mask, (3, 3), iterations=2)
            mask = cv2.dilate(mask, (3, 3), iterations=10)
            mask = imutils.resize(mask, width=IMG_SIZE, height=IMG_SIZE)
            cv2.imshow("mask", mask)

            path = os.path.join("/home/igor/PycharmProjects/itComesHandy/data", dset_name)
            if os.path.exists(path):
                cv2.imwrite("data/" + dset_name + "/" + dset_name + "{}".format(i) + ".png", mask)
            else:
                os.mkdir(path)
                cv2.imwrite("data/" + dset_name + "/" + dset_name + "{}".format(i) + ".png", mask)
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break
            time.sleep(time_between_shots)
    vs.stop()
    cv2.destroyAllWindows()


def predict_camera_input(model, IMG_SIZE=50):
    vs = VideoStream(src=0).start()
    time.sleep(3.0)
    sliders = SlidersC.Sliders()

    while sliders.is_started() is False:
        frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=600, height=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        sliders.update_slider()
        mask = cv2.inRange(hsv, sliders.get_lower_values(), sliders.get_upper_values())
        mask = cv2.erode(mask, None, iterations=2)
        mask = imutils.resize(mask, width=IMG_SIZE, height=IMG_SIZE)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            exit()

    while (True):

        frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=600, height=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, sliders.get_lower_values(), sliders.get_upper_values())
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        cv2.imshow("mask", mask)

        mask = np.array(mask).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        mask = mask/255.0

        prediction = model.predict(mask)
        if prediction >=0.5:
            print("fist")
        else:
            print("open hand")
        os.system('clear')
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


def load_data(path_to_data_dir, CATEGORIES):  # pass path to your data directory

    data = []
    for category in CATEGORIES:
        path = os.path.join(path_to_data_dir, category)
        class_num = CATEGORIES.index(category)
        print(category + " " + str(class_num))
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            data.append([img_array, class_num])
    return data
    # path = os.path.join(path_to_data_dir)
    # if path:
    #     dirs = os.listdir(path + "/")
    #     for d in dirs:
    #         p = os.path.join(path + "/" + d)
    #         for img in os.listdir(p):
    #             if img.endswith('.png'):
    #                 i = Image.open("data/"+d+"/"+img)
    #                 matrix = np.asarray(i)
    #                 temp_list.append(matrix)
    #         t = copy.deepcopy(temp_list)
    #         data[d] = t
    #         temp_list.clear()
    #     return data
    # else:
    #     ValueError("No data directory")


def add_category(data):  # data is dict
    y = {}
    temp = []
    i = -1
    for k in data:
        t = [0 for i in data.keys()]
        i += 1
        for d in data[k]:
            t[i] = 1
            temp.append(t)
        y[k] = copy.deepcopy(temp)
        temp.clear()
    return y


def split_data(data, y, factor=0.75):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for k in data:
        d = 0
        while d < len(data[k]) * factor:
            train_x.append(data[k][d])
            train_y.append(y[k][d])
            d += 1
        while d < len(data[k]):
            test_x.append(data[k][d])
            test_y.append(y[k][d])
            d += 1
    return train_x, train_y, test_x, test_y


def normalize_data(data):  # data is a numpy array
    for k in range(len(data)):
        data[k] = data[k].astype(np.float32)
        data[k] /= 255
    return data
