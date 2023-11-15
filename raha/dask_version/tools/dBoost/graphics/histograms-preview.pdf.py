#!/usr/bin/env python3
import matplotlib
from utils import TANGO, filename
from matplotlib import pyplot
from numpy import linspace

matplotlib.rcParams['lines.linewidth'] = 2

def histogram_plt():
    y = [0.1, 1, 5, 6, 0.2, 1, 10, 8, 0]
    x = linspace(0, 8, num = 9)
    T = 0.5

    FILLS   = (TANGO["red"][0], TANGO["green"][1])
    STROKES = (TANGO["red"][1], TANGO["green"][2])
    colors = [FILLS[yy > T] for yy in y]
    edgecolors = [STROKES[yy > T] for yy in y]

    pyplot.clf()
    pyplot.bar(x, y, width=1)

    ax = pyplot.axes(frameon = False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(-1, 10)

    pyplot.bar(x, y, width = 1, color = colors, edgecolor = edgecolors, linewidth = 2)
    pyplot.tight_layout()

batch_mode, fname = filename("models-plots.pdf")

if batch_mode:
    histogram_plt()
    pyplot.savefig(fname, transparent = True)
