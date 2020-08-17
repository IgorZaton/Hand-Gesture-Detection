import convnet as cnn
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty

# net.predict_camera()


class Output(BoxLayout):
    pass

class mainApp(App):
    def build(self):
    #     self.net = cnn.ConvNet()
    #     self.net.set_camera()
    #     self.net.predict_camera()
        return Output()

    def get_open_hand_value(self, val=0):
        return val
    def get_fist_value(self, val=0):
        return val
    def get_peace_value(self, val=0):
        return val
    def get_yolo_value(self, val=0):
        return val

