from tkinter import *
from tkinter.ttk import *


class Bar:
    def __init__(self):
        self.root = Tk()
        self.bar1 = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')
        self.bar2 = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')
        self.bar3 = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')
        self.bar4 = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate')

    def update(self, p1, p2, p3, p4):
        self.bar1['value'] = p1
        self.bar2['value'] = p2
        self.bar3['value'] = p3
        self.bar4['value'] = p4
        self.root.update_idletasks()
