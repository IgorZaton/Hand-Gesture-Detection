import cv2
from imutils.video import VideoStream
import time
import imutils
import os
import SlidersC


def get_data(dset_name, imgnum=500):
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

        # sl = (0, 58, 50)
        # sh = (30, 255, 255)
        sliders.update_slider()
        mask = cv2.inRange(hsv, sliders.get_lower_values(), sliders.get_upper_values())
        # mask = cv2.inRange(hsv, sl, sh)
        #cv2.imshow("mask", mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = imutils.resize(mask, width=28, height=28)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    for i in range(imgnum):
        frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=600, height=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # sl = (0, 58, 50)
        # sh = (30, 255, 255)
        # sliders.update_slider()
        mask = cv2.inRange(hsv, sliders.get_lower_values(), sliders.get_upper_values())
        # mask = cv2.inRange(hsv, sl, sh)
        #cv2.imshow("mask", mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = imutils.resize(mask, width=28, height=28)
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
        time.sleep(0.2)

    vs.stop()
    cv2.destroyAllWindows()