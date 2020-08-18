from tkinter import *
from tkinter.ttk import *


class Bar:
    def __init__(self):
        self.root = Tk()
        self.root.title("PREDICTIONS")
        self.root.geometry("210x170")
        self.openhand_label = Label(self.root, text="OPEN HAND:")
        self.openhand = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')
        self.fist_label = Label(self.root, text="FIST:")
        self.fist = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')
        self.peace_label = Label(self.root, text="PEACE:")
        self.peace = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')
        self.yolo_label = Label(self.root, text="YOLO:")
        self.yolo = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')
        self.openhand_label.pack()
        self.openhand.pack()
        self.fist_label.pack()
        self.fist.pack()
        self.peace_label.pack()
        self.peace.pack()
        self.yolo_label.pack()
        self.yolo.pack()

    def update(self, p1, p2, p3, p4):
        self.openhand['value'] = p1
        self.fist['value'] = p2
        self.peace['value'] = p3
        self.yolo['value'] = p4
        self.root.update_idletasks()

