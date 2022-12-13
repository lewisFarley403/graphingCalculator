import tkinter as tk
from tkinter import ttk


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Calculator')
        self.geometry('300x500')
        self.label = ttk.Label(self, text='test label')
        self.label.pack()
        self.buttons = []
        for i in range(50):
            b = ttk.Button(self, text=str(i))
            b.pack()
            self.buttons.append(b)


a = App()
if __name__ == '__main__':
    a.mainloop()
