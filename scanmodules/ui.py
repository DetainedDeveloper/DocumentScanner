import tkinter as tk
from tkinter.filedialog import askopenfilename
import webbrowser
import cv2
from scanmodules import scan, utils

scanner = scan.Scanner()


class UI(tk.Frame):

    def __init__(self, master=None):

        super().__init__(master)
        self.master = master
        self.grid()

        self.cam = tk.Button(master=master, text='WebCam', command=self.startWebCam)
        self.cam.grid(row=0, column=0, sticky='WE', padx=4)

        self.file = tk.Button(master=master, text='Choose an image', command=self.openfile)
        self.file.grid(row=1, column=0, sticky='WE', padx=4)

        self.git = tk.Button(master=master, text='GitHub', command=self.openGitHub)
        self.git.grid(row=2, column=0, sticky='WE', padx=4)

        self.quit = tk.Button(master=master, text='Quit', command=self.quit)
        self.quit.grid(row=3, column=0, sticky='WE', padx=4)

    # Start scan process using WebCam feed
    @staticmethod
    def startWebCam():
        scanner.cam(0)

    # Start scan process using static image
    @staticmethod
    def openfile():

        # Get file path
        file_path = askopenfilename(title='Choose an image',
                                          filetypes=[('image', '.jpg'),
                                                     ('image', '.jpeg'),
                                                     ('image', '.png')])

        # Check if file path is empty
        if file_path and not file_path.isspace():
            scanned = scanner.scan(cv2.imread(file_path))
            result = utils.ScanUtils.displayAllImages(scanned, 0.75, True)
            cv2.imshow('Scanner', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def openGitHub():
        webbrowser.open('https://github.com/DetainedDeveloper?tab=repositories')


def launch():

    ui = tk.Tk()

    ui.title('Select an option')
    ui.geometry('180x140')
    ui.resizable(0, 0)

    for i in range(4):
        ui.rowconfigure(i, weight=1)

    ui.columnconfigure(0, weight=1)

    UI(ui).grid()
    ui.mainloop()
