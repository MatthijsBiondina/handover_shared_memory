import tkinter as tk


class View(tk.Tk):
    def __init__(self, canvas_width, canvas_height):
        tk.Tk.__init__(self)
        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, borderwidth=0, highlightthickness=0,
                                bg="#FFFFFF")
        self.canvas.pack(side="top", fill="both", expand="true")
        self.delay = 1  # Delay between redrawing in ms

    def redraw(self):
        raise NotImplementedError

    def update_view(self):
        self.redraw()
        self.update_idletasks()
        self.update()
