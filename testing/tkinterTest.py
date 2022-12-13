import numpy as np
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

xdata = np.random.random([2, 10])

# split the data into two parts
xdata1 = xdata[0, :]
xdata2 = xdata[1, :]

# sort the data so it makes clean curves
xdata1.sort()
xdata2.sort()

ydata1 = xdata1 ** 2
ydata2 = 1 - xdata2 ** 3



import tkinter as tk
root = tk.Tk()
fig = Figure()

canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(fill="both", expand=True)

im = np.array(np.random.rand(10,10,10))

ax = fig.subplots(1,1)

axmax  = fig.add_axes([0.25, 0.01, 0.65, 0.03])
ax.plot(xdata1, ydata1, color='tab:blue')

smax = Slider(axmax, 'Max', 0, np.max(im), valinit=50)
# tracker = IndexTracker(ax, im)

# canvas.mpl_connect('scroll_event', tracker.onscroll)
# canvas.mpl_connect('button_release_event', tracker.contrast) #add this for contrast change
root.mainloop()